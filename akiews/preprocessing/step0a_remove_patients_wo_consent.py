import os
import gc
import sys
import pandas as pd
import numpy as np


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-data_path')
parser.add_argument('-pid_excl_files', nargs='+')
parser.add_argument('-output_file')
args = parser.parse_args()
data_path = args.data_path
pid_excl_files = args.pid_excl_files
output_file = args.output_file

pID_set = set()
for tbl in ['generaldata', 'labres', 'observrec', 'pharmarec', 'dervals', 'monvals']:
    if 'csv_exports' in data_path:
        if tbl in ['monvals', 'dervals']:
            chunksize = 10 ** 7
            csv_iters = pd.read_csv(os.path.join(data_path, 'expot-%s.csv'%tbl),
                                    sep=';', usecols=['PatientID'], 
                                    chunksize=chunksize)
            pID_tmp = set()
            for df in csv_iters:
                pID_tmp |= set(df.PatientID)

        elif tbl == 'observrec':
            df = pd.read_csv(os.path.join(data_path, 'expot-%s.csv'%tbl), sep=';', 
                             low_memory=True, encoding='cp1252', na_values='(null)', 
                             usecols=['PatientID', 'VariableID', 'Value'], 
                             dtype={'PatientID': np.int64, 'VariableID': np.int64,
                                    'Value': np.float64})
            pID_tmp = set(df.PatientID)
            pID_excl_gc = set(df[(df.VariableID==15004651)&(df.Value>=4)].PatientID)
        else:
            df = pd.read_csv(os.path.join(data_path, 'expot-%s.csv'%tbl),
                             sep=';', usecols=['PatientID'])
            pID_tmp = set(df.PatientID)

        print('# patients in %s: %d'%(tbl, len(pID_tmp)))
        pID_set |= pID_tmp
    else:
        id_path = os.path.split(output_file)[0]
        pID_tmp = set()
        if tbl=='observrec':
            pID_excl_gc = set()
        tbl_path = os.path.join(data_path, 'p_%s'%tbl)
        lst_files = [f for f in os.listdir(tbl_path) if 'crc' not in f and 'parquet' in f]
        pid_f_map = []
        for i, f in enumerate(lst_files):
            if 'crc' in f or 'parquet' not in f:
                continue
            df = pd.read_parquet(os.path.join(tbl_path, f), engine='pyarrow')
            pID_batch = df.patientid.unique()
            pid_f_map.append(np.stack((pID_batch, [os.path.join(tbl_path, f)]*len(pID_batch)), axis=1))
            pID_tmp |= set(pID_batch)
            if tbl=='observrec':
                pID_excl_gc |= set(df[(df.variableid==15004651)&(df.value>=4)].patientid)
            sys.stdout.write('%2d/%2d files of %s are processed.\r'%(i+1, len(lst_files), tbl))
            sys.stdout.flush()
        pid_f_map = np.vstack(pid_f_map)
        pid_f_map = pd.DataFrame(pid_f_map, columns=['PatientID', 'FileName'])
        pid_f_map.loc[:,'PatientID'] = pid_f_map.PatientID.astype(np.int64)
        # pid_f_map.to_hdf(os.path.join(id_path, 'PID_File_Mapping_%s.h5'%tbl), 
        #                  'data', complevel=5, complib='blosc:lz4')
        print('')
        print('# patients in %s: %d'%(tbl, len(pID_tmp)))
        pID_set |= pID_tmp

print('# patients in raw data: %d'%len(pID_set))
print('# patients with value(15004651) >= 4 : %d'%len(pID_excl_gc))

pID_excl = set()
for f in pid_excl_files:
    print('exclude file', f, len(set(pd.read_csv(f).PatientID)))
    pID_excl |= set(pd.read_csv(f).PatientID)
print('# patients to exclude (provided): %d'%len(pID_excl))
pID_set -= pID_excl_gc
pID_set -= pID_excl
print('# consenting patients: %d'%(len(pID_set)))

pID_set = np.sort(list(pID_set))
pID_set = pd.DataFrame(pID_set.reshape((-1,1)), columns=['PatientID'])

# pID_set.to_csv(output_file.replace('PID_WithConsent.csv', 'PID_WithConsent_201103.csv'), index=False)
