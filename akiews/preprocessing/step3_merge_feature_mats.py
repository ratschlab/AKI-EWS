#!/usr/bin/env python
import pandas as pd
import numpy as np
import os

import sys
sys.path.append('../utils')
import preproc_utils

def get_feature_mat(tbl_name, patient_id):
    input_path_tbl = os.path.join(input_path, tbl_name)
    filename = [f for f in os.listdir(input_path_tbl) if '_%d_'%batch_id in f][0]
    df = pd.read_hdf(os.path.join(input_path_tbl, filename), where='PatientID=%d'%patient_id, mode='r')        
    df.set_index(['PatientID', 'Datetime'], inplace=True)
    return df

def merge_single_patient_feature_mats(patient_id):
    df_mat = get_feature_mat('monvals', patient_id)
    df_mat = [df_mat] + [get_feature_mat(tbl_name, patient_id) for tbl_name in ['dervals', 'observrec', 'pharmarec', 'labres']]
    df_merged = pd.concat(df_mat, join='outer', axis=1)

    datetime_list = [set([x[1] for x in df_tmp.index]) for df_tmp in df_mat]
    datetime_set = set()
    for x in datetime_list:
        datetime_set |= x
    try:        
        assert(len(df_merged) == len(datetime_set))
    except AssertionError:
        print('Number of records in merged table does not match with the size of datetime union set of all tables.')
        import ipdb
        ipdb.set_trace()

    df_merged.reset_index(inplace=True)
    df_merged.sort_values(by=['Datetime'], inplace=True)

    try:
        assert(len(df_merged) == len(df_merged.Datetime.unique()))
    except AssertionError:
        print('There are duplicated records with the same datetime')
        import ipdb
        ipdb.set_trace()

    columns = np.sort(df_merged.columns[2:]).tolist()
    return df_merged[['PatientID', 'Datetime'] + columns]

def main():

    df_urine_sum = pd.read_hdf(os.path.join(urine_sum_path, 'urine_sum_%d_%d--%d.h5'%(batch_id, np.min(pid_list), np.max(pid_list))), 
                               'data', mode='r')
    df_urine_sum.rename(columns={'Value': 'vm24'}, inplace=True)
    
    df_idx_start = 0
    for nn, pid in enumerate(pid_list):
        df = merge_single_patient_feature_mats(pid)

        if len(df) == 0:
            print('Patient', pid, 'does not have data from any table.')
            continue
        df_urine_sum_tmp = df_urine_sum[df_urine_sum.PatientID==pid]
        df = df.merge(df_urine_sum_tmp[['Datetime', 'vm24']], how='left', left_on='Datetime', right_on='Datetime')

        if write_to_disk:
            df.set_index(np.arange(df_idx_start, df_idx_start+len(df)), inplace=True, drop=True)
            df['PatientID'] = df.PatientID.astype(int)
            try:
                df.to_hdf(output_path, 'fmat', append=True, complevel=5, 
                          complib='blosc:lz4', data_columns=['PatientID'], format='table')
            except:
                import ipdb
                ipdb.set_trace()
            df_idx_start += len(df)

        sys.stdout.write('# patients processed: %3d / %3d\r'%(nn+1, len(pid_list)))
        sys.stdout.flush()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_path')
    parser.add_argument('-output_path')
    parser.add_argument('-urine_sum_path')
    parser.add_argument('-chunking_info_file')
    parser.add_argument('-batch_id', type=int)
    parser.add_argument('--write_to_disk', action='store_true')
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    urine_sum_path = args.urine_sum_path
    chunking_info_file = args.chunking_info_file
    batch_id = args.batch_id
    write_to_disk = args.write_to_disk


    chunking_info = pd.read_csv(chunking_info_file)
    chunking_info.rename(columns={'ChunkfileIndex': 'BatchID'}, inplace=True)
    if 'PatientID' in chunking_info.columns:
        pid_list = chunking_info.PatientID[chunking_info.BatchID==batch_id].values
    else:
        pid_list = chunking_info.index[chunking_info.BatchID==batch_id].values

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, 'fmat_%d_%d--%d.h5'%(batch_id, 
                                                               np.min(pid_list), 
                                                               np.max(pid_list)))

    if write_to_disk and os.path.exists(output_path):
        print('Already os.path.exists: %s.'%output_path)
        print('Please delete it manually if you want to reproduce a new one.')
        exit(0)
    main()
