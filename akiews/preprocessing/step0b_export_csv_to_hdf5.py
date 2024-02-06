#!/usr/bin/env python
import os
import gc
import pandas as pd
import numpy as np

dtype_dict = {'Value': np.float64, 'Rate': np.float64, 
              'CumulDose': np.float64, 'GivenDose': np.float64,
              'PatientID': np.int64, 'VariableID': np.int64,
              'ResultID': np.int64, 'InfusionID': np.int64,
              'PharmaID': np.int64, 'Route': np.int64, 
              'Datetime': np.str_, 'DateTime': np.str_,
              'EnterTime': np.str_, 'Entertime': np.str_,
              'SampleTime': np.str_, 'AdmissionTime': np.str_,
              'birthYear': np.int64, 'Sex': np.str_,
              'Status': np.int32, 'PatGroup': np.int32}
class CsvLoaderMain:
    def __init__(self, data_path, output_path, consent_pids_file):
        self.data_path = data_path
        self.output_path = output_path
        self.consent_pids_file = consent_pids_file

    def LoadCsv2HDF5(self, tbl, write_to_disk=False):
        if tbl in ['monvals', 'comprvals', 'dervals']: 
            fields = ['PatientID', 'VariableID', 'Datetime', 
                      'Entertime' if tbl=='monvals' else 'EnterTime', 
                      'Value', 'Status']

        elif tbl == 'generaldata':
            fields = ['PatientID', 'AdmissionTime', 'birthYear','Sex', 
                      'PatGroup', 'Status']

        elif tbl == 'labres':
            fields = ['PatientID', 'VariableID', 'ResultID', 'SampleTime', 
                      'EnterTime', 'Value', 'Status', ]

        elif tbl == 'observrec':
            fields = ['PatientID', 'DateTime', 'EnterTime', 'VariableID', 
                      'Value', 'Status']

        elif tbl == 'pharmarec':
            fields = ['PatientID', 'PharmaID', 'InfusionID', 'Route', 
                      'DateTime', 'EnterTime', 'CumulDose', 'GivenDose', 
                      'Rate', 'Status']
        else:
            raise Exception('Wrong table name.')

        dtype = {key: dtype_dict[key] for key in fields}
        input_path = os.path.join(self.data_path, 'expot-%s.csv'%tbl)
        if write_to_disk:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            output_path = os.path.join(self.output_path, '%s.h5'%tbl)

        pID_consent = pd.read_csv(self.consent_pids_file).PatientID.values

        if tbl in ['monvals', 'comprvals', 'dervals']:
            chunksize = 10 ** 7
            iter_csv = pd.read_csv(input_path, usecols=fields, dtype=dtype, 
                                   sep=';', chunksize=chunksize, low_memory=True, 
                                   encoding='cp1252', na_values='(null)')
            lst_vID = []
            lst_pID = []
            idx_row = 0
            for i, df in enumerate(iter_csv):
                print(i)
                df.rename(columns={'DateTime': 'Datetime', 
                                   'SampleTime': 'Datetime',
                                   'EnterTime': 'Entertime'}, inplace=True)
                for col in ['Datetime', 'Entertime']:
                    df.loc[:,col] = pd.to_datetime(df[col]).dt.floor(freq='s')
                idx2drop = df.index[~np.isin(df.PatientID.values, pID_consent)]
                df.drop(idx2drop, inplace=True)
                gc.collect()

                lst_vID.extend(df.VariableID.unique().tolist())
                lst_pID.extend(df.PatientID.unique().tolist())


                if write_to_disk:
                    df.set_index(np.arange(idx_row, idx_row+len(df)), 
                                 drop=True, inplace=True)
                    idx_row += len(df)
                    df.to_hdf(output_path, 'raw_import', append=True, 
                              complevel=5, complib='zlib', 
                              data_columns=True, format='table')

            pID_set = np.sort(np.unique(lst_pID))
            vID_set = np.sort(np.unique(lst_vID))
        else:
            df = pd.read_csv(input_path, usecols=fields, dtype=dtype, 
                             sep=';', low_memory=True, 
                             encoding='cp1252', na_values='(null)')

            df.rename(columns={'DateTime': 'Datetime', 
                               'SampleTime': 'Datetime',
                               'EnterTime': 'Entertime'}, inplace=True)
            for col in ['Datetime', 'Entertime', 'AdmissionTime']:
                if col not in df.columns:
                    continue 
                df.loc[:,col] = pd.to_datetime(df[col]).dt.floor(freq='s')

            idx2drop = df.index[~np.isin(df.PatientID.values, pID_consent)]
            df.drop(idx2drop, inplace=True)
            gc.collect()
 
            if write_to_disk:
                df.reset_index(drop=True, inplace=True)
                df.to_hdf(output_path, 'raw_import', 
                          complevel=5, complib='zlib', 
                          data_columns=True, format='table')

            pID_set = np.sort(df.PatientID.unique())
            if tbl == 'pharmarec':                
                vID_set = np.sort(df.PharmaID.unique())
            elif tbl != 'generaldata':
                vID_set = np.sort(df.VariableID.unique())
           
        if write_to_disk:
            print('%s is created.'%output_path)

        print('Number of patients = %d'%len(pID_set))
        pID_set = pd.DataFrame(pID_set.reshape((-1,1)), columns=['PatientID'])
        if tbl == 'generaldata':
            return pID_set, None
        else:
            print('Number of variables = %d'%len(vID_set))
            vID_set = pd.DataFrame(vID_set.reshape((-1,1)), columns=['VariableID'])
            return pID_set, vID_set

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-tbl')
    parser.add_argument('-input_path')
    parser.add_argument('-output_path')
    parser.add_argument('-id_path')
    parser.add_argument('-consent_pids_file')
    parser.add_argument('--write_to_disk', action='store_true')

    args = parser.parse_args()
    
    configs = vars(args)

    l = CsvLoaderMain(configs['input_path'], 
                      configs['output_path'], 
                      configs['consent_pids_file'])

    pIDs, vIDs = l.LoadCsv2HDF5(configs['tbl'], 
                                write_to_disk=configs['write_to_disk'])

    pIDs.to_csv(os.path.join(configs['id_path'], 'PID_consent_%s.csv'%configs['tbl']), index=False)
    if configs['tbl']!='generaldata':
        vIDs.to_csv(os.path.join(configs['id_path'], 'VID_%s.csv'%configs['tbl']), index=False)
