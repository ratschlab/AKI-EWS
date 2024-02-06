#!/usr/bin/env python
import pandas as pd
import numpy as np
from time import time

import os
import ipdb
import sys
sys.path.append('../utils')
sys.path.append('./pharma')
import preproc_utils
from pharmarec_processing import pivot_pharma_table

def LoadRawHDF5Values(tbl_name, 
                      patient_id, 
                      variable_ids=None,
                      no_duplicates=True,
                      datetime_col='Datetime',
                      verbose=False):
    """
    Load data of selected patients from the HDF file of a table.

    Parameters:
    tbl_name: the name of the table (string)
    patient_id: the selected patient ID
    verbose: bool; while true, print the information of the returned dataframe

    Returns:
    df: a dataframe consisting the selected data
    """

    
    df = pd.read_hdf(input_path, where='PatientID = %d'%patient_id, mode='r')

    # If variables of interest are specified, then only select columns of interest and discard
    # the rest
    if variable_ids is not None:
        if tbl_name == 'pharmarec':
            variables_intersect = set(variable_ids) & set(df['PharmaID'].tolist())
        else:
            variables_intersect = set(variable_ids) & set(df['VariableID'].tolist())

            
        if len(variables_intersect) == 0:
            df = df.loc[[]]
        else:
            if tbl_name == 'pharmarec':
                df = df[df['PharmaID'].isin(variables_intersect)]
            else:
                df = df[df['VariableID'].isin(variables_intersect)]

    # Rename columns to the same name
    df = df.rename(columns={'PharmaID': 'VariableID', 'GivenDose': 'Value'})
    if tbl_name == 'pharmarec':
        df = df[['PatientID', datetime_col, 'VariableID', 'InfusionID', 'Rate', 'Value', 'Status']]
    else:
        df = df[['PatientID', datetime_col, 'VariableID', 'Value']]
    df.rename(columns={datetime_col: 'Datetime'}, inplace=True)

    if tbl_name not in ['labres', 'pharmarec']:
        if (variables_of_interest['Impute semantics']=='Attribute one grid point').sum()>0:
            vid_nan2one = variables_of_interest.index[variables_of_interest['Impute semantics']=='Attribute one grid point']
            if df.VariableID.isin(vid_nan2one).sum() > 0:
                df.loc[df.index[df.VariableID.isin(vid_nan2one)], 'Value'] = 1

        if (variables_of_interest['VariableUnit']=='binary [timestamp present with no value]').sum()>0:
            vid_nan2one = variables_of_interest.index[variables_of_interest['VariableUnit']=='binary [timestamp present with no value]']
            if df.VariableID.isin(vid_nan2one).sum() > 0:
                df.loc[df.index[df.VariableID.isin(vid_nan2one)], 'Value'] = 1

    if verbose:
        if len(df) > 0:
            print('patient ', patient_id, 
                  '; # records ', len(df), 
                  '; # variables ', len(df.VariableID.unique()))
        else:
            print('No data of interest was found for patient ', patient_id)
        
    return df
                    
def remove_invalid_chars(variable_name):
    """
    Remove chars that should not appear in the column name of a dataframe.
    """
    for x in [' ', '-', '(', ')', ',', '/', ':']:
        if x in variable_name:
            variable_name = variable_name.replace(x,'_')
    for x in ['.', '*', '+']:
        if x in variable_name:
            variable_name = variable_name.replace(x, '')
    return variable_name

def table2matrix(tbl_name, df_tbl, variables_of_interest):
    """
    Export data from table to a feature matrix, where each column represents a variable.
    
    Parameter:
    df_tbl: the dataframe containing these columns: Datetime, PatientID, VariableID and Value
    variables_of_interest: a table of the mapping between their variable IDs and variable 
    names of variables that we are interested in.
    
    Returns:
    df_mat: the feature matrix, whose columns are associated with variables.
    """

    # If we choose to use the original name of the variables instead of IDs as the column names,
    # we need to remove some of the invalid chars from the variable names. 
    voi_tmp = variables_of_interest.copy()
    voi_tmp.VariableName.apply(lambda x: remove_invalid_chars(x))

    df_tbl = df_tbl.join(voi_tmp, on='VariableID', how='inner')
    
    if tbl_name == 'pharmarec':
        df_mat = pivot_pharma_table(df_tbl, switch='steph')
    else:
        df_tbl.drop('VariableID', axis=1, inplace=True)
        df_mat = pd.pivot_table(df_tbl, values='Value', index=['PatientID', 'Datetime'],
                                columns=['VariableName'])
    # Add the variables that are among the variables of interest but were not measured for the patients
    # in df_tbl. 
    missing_columns = set(voi_tmp.VariableName.tolist()) - set(df_mat.columns)
    for col in missing_columns:
        if tbl_name == 'pharmarec':
            df_mat[col] = 0
        else:
            df_mat[col] = np.nan    
    
    df_mat.reset_index(inplace=True)
    
    return df_mat


def main():
    
    df_idx_start = 0
    for nn, pid in enumerate(pid_list):
        # if pid != 12522:
        #     continue
        t_total = 0        
        t = time()
        df_tbl = LoadRawHDF5Values(tbl_name, pid, variable_ids=vid_list, verbose=True)
        t_read = time() - t
        print('Read time', t_read, 'secs')
        t_total += t_read
        if len(df_tbl) > 0:
            t = time()
            df_mat = table2matrix(tbl_name, df_tbl, variables_of_interest)
            t_pivot = time() - t
            print('Table-to-matrix transform time', t_pivot, 'secs')
            t_total += t_pivot

            # if tbl_name != 'pharmarec':
            #     try:
            #         assert(len(df_mat)==len(df_tbl.drop_duplicates(['Datetime'])))
            #     except AssertionError:
            #         print('Timestamp number mismatches between the feature matrix and the table.')
            #         pass

            #     try:
            #         assert(len(df_tbl.VariableID.unique())==np.sum(np.sum(pd.notnull(df_mat.iloc[:,2:]), axis=0)>0))
            #     except AssertionError:
            #         print('Variable number mismatches between the feature matrix and the table')
            #         ipdb.set_trace()

            sum_tbl_value = np.sum(df_tbl.Value)
            if tbl_name == 'pharmarec':
                sum_mat_value = 0
                non_zero_drugs = df_mat.dropna(axis=1, how='all').columns[2:]
                for col in non_zero_drugs:
                    df_drug = df_mat.loc[:, ['Datetime', col]]
                    df_drug.set_index('Datetime', inplace=True)
                    df_drug.sort_index(inplace=True)
                    df_drug.dropna(inplace=True)
                    if df_drug.shape[0] < 2:
                        if df_drug.values.sum() == 0:
                            # carry on
                            continue
                        else:
                            print('instantaneous drug?')
                            ipdb.set_trace()
                    # this seems to be converting back from rate into given doses to compare that the contents of the dataframe are unchanged (in aggregate, at least)
                    # time_difference calculates a time difference in hours, convert to minutes
                    # TODO somethign is a bit weird here, but ignore it for now
                    tdiff = ((df_drug.index[1:] - df_drug.index[:-1]).astype('timedelta64[s]').values)/60
                    total_dose = np.dot(df_drug.values[:-1].reshape(-1, ), tdiff)
                    sum_mat_value += total_dose
            else:
                sum_mat_value = np.nansum(df_mat.iloc[:,2:])
            try:
                # NOTE: due to rescaling, this should not be true for many pharma records
                assert(np.abs(sum_tbl_value - sum_mat_value) < 1e-4)
            except AssertionError:
                # If the big difference was due to the large absolute values then we look at the the relative difference
                if sum_tbl_value != 0 : 
                    try:
                        assert(np.abs(sum_tbl_value - sum_mat_value) / sum_tbl_value < 1e-4)
                    except AssertionError:
                        print('The sum of values in the feature matrix does not match with the sum in the table.')
                        print('\t\t table:', sum_tbl_value)
                        print('\t\t matrix:', sum_mat_value)
                        if tbl_name == 'pharmarec':
                            print('... but this is expected behaviour due to drug rescaling')
                        else:
                            ipdb.set_trace()

            df_mat.loc[:,'PatientID'] = pid
            df_mat.sort_values('Datetime', inplace=True)
            df_mat.set_index(np.arange(df_idx_start, df_idx_start+len(df_mat)), inplace=True)
            df_idx_start += len(df_mat)
            for col in df_mat.columns[2:]:
                df_mat[col] = df_mat[col].astype(float)

            if write_to_disk:
                t = time()
                df_mat.to_hdf(output_path, 'pivoted', append=True, complevel=5, 
                              complib='blosc:lz4', data_columns=['PatientID'], format='table')
                t_write = time() - t
                print('Write time', t_write, 'secs')
                t_total += t_write
                
        sys.stdout.write('# patients processed: %3d / %3d; Total runtime = %g sec\r'%(nn+1, len(pid_list), t_total))
        sys.stdout.flush()
    return 1

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-tbl_name')    
    parser.add_argument('-input_path')
    parser.add_argument('-output_path')
    parser.add_argument('-chunking_info_file')
    parser.add_argument('-voi_path')
    parser.add_argument('-version')        
    parser.add_argument('-batch_id', type=int)
    parser.add_argument('--write_to_disk', action='store_true')
    args = parser.parse_args()
    tbl_name = args.tbl_name
    input_path = args.input_path
    output_path = args.output_path
    chunking_info_file = args.chunking_info_file
    voi_path = args.voi_path
    version = args.version
    batch_id = args.batch_id
    write_to_disk = args.write_to_disk


    chunking_info = pd.read_csv(chunking_info_file)
    chunking_info.rename(columns={'ChunkfileIndex': 'BatchID'}, inplace=True)
    if 'PatientID' in chunking_info.columns:
        pid_list = chunking_info.PatientID[chunking_info.BatchID==batch_id].values
    else:
        pid_list = chunking_info.index[chunking_info.BatchID==batch_id].values
    input_path = os.path.join(input_path, tbl_name)
    input_path = os.path.join(input_path, [f for f in os.listdir(input_path) if '%s_%d_'%(tbl_name, batch_id) in f][0])
    output_path = os.path.join(output_path, tbl_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, '%s_%d_%d--%d.h5'%(tbl_name, 
                                                               batch_id, 
                                                               np.min(pid_list), 
                                                               np.max(pid_list)))


    dict_type = dict(monvals='Monitored', dervals='Derived', observrec='Observed',
                     pharmarec='Pharma')
    variables_of_interest = pd.read_csv(os.path.join(voi_path, 'labref_excel_%s.tsv'%version if tbl_name=='labres' else 'varref_excel_%s.tsv'%version), 
                      sep='\t', encoding='cp1252')
    if tbl_name!='labres':
        variables_of_interest = variables_of_interest[variables_of_interest.Type==dict_type[tbl_name]]
    variables_of_interest.drop(variables_of_interest.index[(variables_of_interest.VariableID.isnull())|(variables_of_interest.VariableID=='Vm23')], inplace=True)
    variables_of_interest.drop_duplicates('VariableID', inplace=True)
    variables_of_interest.loc[:,'VariableID'] = variables_of_interest.VariableID.astype(np.int64)
    variables_of_interest.loc[:,'VariableName'] = variables_of_interest.VariableID.apply(lambda x: 'p%d'%x if tbl_name=='pharmarec' else 'v%d'%x)
    variables_of_interest.set_index('VariableID', inplace=True)
    vid_list = variables_of_interest.index.tolist()

    main()
