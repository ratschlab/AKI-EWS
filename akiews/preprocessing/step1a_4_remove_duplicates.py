#!/usr/bin/env python
import pandas as pd
import numpy as np

import os

import sys
sys.path.append('../utils')
import preproc_utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-tbl')
parser.add_argument('-input_path')
parser.add_argument('-output_path')
parser.add_argument('-std_path')
parser.add_argument('-chunking_info_file')
parser.add_argument('-voi_path')
parser.add_argument('-version')
parser.add_argument('-batch_id', type=int)
parser.add_argument('--write_to_disk', action='store_true')
args = parser.parse_args()
tbl = args.tbl
input_path = args.input_path
output_path = args.output_path
std_path = args.std_path
chunking_info_file = args.chunking_info_file
voi_path = args.voi_path
version = args.version
batch_id = args.batch_id
write_to_disk = args.write_to_disk

data_path = os.path.join(preproc_utils.datapath, '1a_hdf5_clean', version)

chunking_info = pd.read_csv(chunking_info_file)
chunking_info.rename(columns={'ChunkfileIndex': 'BatchID'}, inplace=True)
if 'PatientID' in chunking_info.columns:
    pid_list = chunking_info.PatientID[chunking_info.BatchID==batch_id].values
else:
    pid_list = chunking_info.index[chunking_info.BatchID==batch_id].values
input_path = os.path.join(input_path, tbl)
input_path = os.path.join(input_path, [f for f in os.listdir(input_path) if '%s_%d_'%(tbl, batch_id) in f][0])
output_path = os.path.join(output_path, tbl)
if not os.path.exists(output_path):
    os.makedirs(output_path)
output_path = os.path.join(output_path, '%s_%d_%d--%d.h5'%(tbl, 
                                                           batch_id, 
                                                           np.min(pid_list), 
                                                           np.max(pid_list)))

dict_type = dict(monvals='Monitored', dervals='Derived', observrec='Observed',
                 pharmarec='Pharma')
voi = pd.read_csv(os.path.join(voi_path, 'labref_excel_%s.tsv'%version if tbl=='labres' else 'varref_excel_%s.tsv'%version), 
                  sep='\t', encoding='cp1252')
if tbl!='labres':
    voi = voi[voi.Type==dict_type[tbl]]
voi.drop(voi.index[(voi.VariableID.isnull())|(voi.VariableID=='Vm23')], inplace=True)
voi.drop_duplicates('VariableID', inplace=True)
voi.loc[:,'VariableID'] = voi.VariableID.astype(np.int64)
voi.set_index('VariableID', inplace=True)


if tbl == 'pharmarec':
    for nn, pid in enumerate(pid_list):    
        df = pd.read_hdf(input_path, where='PatientID=%d'%pid)
        
        cols_of_interest = ['PharmaID', 'InfusionID', 'Datetime', 'Status']
        if len(df.drop_duplicates(cols_of_interest)) != len(df):
            idx = set(df.index.tolist()) - set(df.drop_duplicates(cols_of_interest).index.tolist())
            idx = list(idx)
            for tmp in idx:
                if tmp not in df.index:
                    continue
                df_tmp  = df[np.logical_and(df.PharmaID==df.loc[tmp]['PharmaID'], 
                                            df.InfusionID==df.loc[tmp]['InfusionID'])]
                iloc = np.where(df_tmp.index==tmp)[0][0]
                if df.loc[tmp]['Status'] == 776:
                    index2drop = df_tmp.index[np.logical_and(df_tmp.Status == df.loc[tmp]['Status'], df_tmp.GivenDose==0)]                        
                    df.drop(index2drop, axis=0, inplace=True)
                elif df.loc[tmp]['Status'] == 780:
                    index_dups = df_tmp.index[np.logical_and(df_tmp.Status == df.loc[tmp]['Status'], 
                                                             df_tmp.Datetime==df.loc[tmp]['Datetime'])]
                    df.loc[index_dups[-1], 'GivenDose'] = np.sum(df.loc[index_dups, 'GivenDose'])
                    df.loc[index_dups[-1], 'CumulDose'] = np.sum(df.loc[index_dups, 'CumulDose'])
                    df.drop(index_dups[:-1], axis=0, inplace=True)
                else:
                    index_dups = df_tmp.index[np.logical_and(df_tmp.Status == df.loc[tmp]['Status'], 
                                                             df_tmp.Datetime==df.loc[tmp]['Datetime'])]
                    df.loc[index_dups[-1], 'GivenDose'] = np.mean(df.loc[index_dups, 'GivenDose'])
                    df.loc[index_dups[-1], 'CumulDose'] = np.mean(df.loc[index_dups, 'CumulDose'])
                    df.drop(index_dups[:-1], axis=0, inplace=True)

        cols_of_interest = ['PharmaID', 'InfusionID', 'Datetime']
        if len(df.drop_duplicates(cols_of_interest)) != len(df):           
            idx = set(df.index.tolist()) - set(df.drop_duplicates(cols_of_interest).index.tolist())
            idx = list(idx)
            for tmp in idx:
                if tmp not in df.index:
                    continue
                df_tmp  = df[np.logical_and(df.PharmaID==df.loc[tmp]['PharmaID'], 
                                            df.InfusionID==df.loc[tmp]['InfusionID'])]
                index_dups = df_tmp[df_tmp.Datetime==df.loc[tmp]['Datetime']].sort_values(['Datetime', 'Entertime']).index
                df_dup_tmp = df_tmp.loc[index_dups]
                iloc = np.where(df_tmp.index==tmp)[0][0]
                if 544 in df_dup_tmp.Status.values:
                    if np.sum(df_dup_tmp[df_dup_tmp.Status==544].GivenDose==0) == np.sum(df_dup_tmp.Status==544):
                        print(df_tmp.loc[index_dups])
                        index2drop = df_dup_tmp.index[df_dup_tmp.Status==544]
                        df.drop(index2drop, axis=0, inplace=True)
                    else:
                        print(df_tmp.loc[index_dups])
                else:
                    print(df_tmp.loc[index_dups])
                    df.loc[index_dups[-1], 'GivenDose'] = np.sum(df.loc[index_dups, 'GivenDose'])
                    df.loc[index_dups[-1], 'CumulDose'] = np.sum(df.loc[index_dups, 'CumulDose'])
                    df.drop(index_dups[:-1], axis=0, inplace=True)
                     
        if write_to_disk:
            df.to_hdf(output_path, 'data', append=True, format='table', data_columns=True, complevel=5, complib='blosc:lz4')

        sys.stdout.write('# patients processed: %3d / %3d\r'%(nn+1, len(pid_list)))
        sys.stdout.flush()
    
else:
    # std_path = os.path.join(std_path, version)
    if tbl == 'monvals':
        mean_std = []
        for f in os.listdir(std_path):
            if tbl not in f:
                continue
            mean_std_info_path = os.path.join(std_path, f)
            mean_std.append(pd.read_csv(mean_std_info_path, sep='\t').set_index('VariableID'))
        mean_std = pd.concat(mean_std, axis=0)
    else:
        mean_std_info_path = os.path.join(std_path, tbl+'.tsv')
        mean_std = pd.read_csv(mean_std_info_path, sep='\t').set_index('VariableID')
    
    if tbl != 'labres':
        mean_std = mean_std.merge(voi[['VariableUnit']], how='left', left_index=True, right_index=True)

    for nn, pid in enumerate(pid_list):    
        df = pd.read_hdf(input_path, where='PatientID=%d'%pid)        
        df.loc[:,'Datetime'] = df.Datetime.dt.floor(freq='s')
        df.loc[:,'Entertime'] = df.Entertime.dt.floor(freq='s')

        df = df[df.VariableID.isin(voi.index)].copy()

        df.drop_duplicates(['VariableID', 'Datetime', 'Value'], inplace=True)
        for vid in df.VariableID.unique():
            df_tmp = df[df.VariableID==vid]
            dt_cnt = df_tmp.Datetime.value_counts()
            if dt_cnt.max() > 1:
                dt_dup = dt_cnt.index[dt_cnt > 1]
                for dt in dt_dup:
                    tmp = df_tmp[df_tmp.Datetime==dt]
                    if tbl == 'labres':
                        df.drop(tmp.index[tmp.Status.isin([9, 72, 136, -120])], inplace=True)
                        tmp.drop(tmp.index[tmp.Status.isin([9, 72, 136, -120])], inplace=True)

                    if len(tmp) > 1:
                        if voi.loc[vid].Datatype == 'Categorical':
                            df.drop(tmp.index, axis=0, inplace=True) 
                        elif tmp.Value.std() < 0.05 * mean_std.loc[vid].Std:
                            df.loc[tmp.index, 'Value'] = tmp.Value.mean()
                        else:
                            df.drop(tmp.index, axis=0, inplace=True)
        df.drop_duplicates(['Datetime', 'VariableID', 'Value'], inplace=True)
        if write_to_disk:
            df.to_hdf(output_path, 'data', append=True, format='table', data_columns=['PatientID', 'VariableID'], complevel=5, complib='blosc:lz4')

        sys.stdout.write('# patients processed: %3d / %3d\r'%(nn+1, len(pid_list)))
        sys.stdout.flush()
