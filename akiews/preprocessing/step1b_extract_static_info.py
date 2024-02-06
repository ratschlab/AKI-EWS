#!/usr/bin/env python
import pandas as pd
import numpy as np

import gc
import os

import sys
sys.path.append('../utils')
import preproc_utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-hdf_path')
parser.add_argument('-generaldata_path')
parser.add_argument('-height_path')
parser.add_argument('-chunking_info_file')
parser.add_argument('-version')

args = parser.parse_args()
hdf_path = args.hdf_path
generaldata_path = args.generaldata_path
height_path = args.height_path
chunking_info_file = args.chunking_info_file
version = args.version

# hdf_path = os.path.join(preproc_utils.datapath, '1a_hdf5_clean', version, 'datetime_fixed')

# VariableID and table information of the static variables
static_info = dict()
static_info.update(dict(generaldata=['PatientID', 'Sex', 'birthYear', 'AdmissionTime', 'PatGroup']))
static_info.update(dict(observrec=dict(Discharge=14000100, 
                                       Emergency=14000140, 
                                       Surgical=14000150)))
static_info.update(dict(dervals=dict(Euroscores=30010001, 
                                     APACHECode=30000100)))

# Map the variableID to its corresponding table
id_name_mapping = {val: key for key, val in static_info['observrec'].items()}
id_name_mapping.update({val: key for key, val in static_info['dervals'].items()})

# Load the generaldata table where most static variables are
if 'hirid2' not in generaldata_path:
    generaldata = pd.read_hdf(os.path.join(generaldata_path, 'generaldata.h5'), mode='r')
else:
    generaldata_file = [f for f in os.listdir(generaldata_path) if 'crc' not in f and 'parquet' in f][0]
    generaldata = pd.read_parquet(os.path.join(generaldata_path, generaldata_file), engine='pyarrow')
    generaldata = generaldata[['patientid', 'birthyear', 'sex', 'admissiontime', 'status', 'patgroup']]
    generaldata.rename(columns={'patientid': 'PatientID', 'birthyear': 'birthYear', 'sex': 'Sex',
                                'admissiontime': 'AdmissionTime', 'status': 'Status', 'patgroup': 'PatGroup'}, inplace=True)
generaldata['AdmissionTime'] = pd.to_datetime(generaldata['AdmissionTime']).dt.floor(freq='s')
generaldata['Age'] = generaldata.AdmissionTime.apply(lambda x: x.year) - generaldata.birthYear

chunking_info = pd.read_csv(chunking_info_file)
chunking_info.rename(columns={'ChunkfileIndex': 'BatchID'}, inplace=True)
if 'PatientID' in chunking_info.columns:
    pid_list = chunking_info.PatientID.values
    chunking_info.set_index('PatientID', inplace=True)
else:
    pid_list = chunking_info.index.values

import ipdb
ipdb.set_trace()

generaldata.drop(generaldata.index[~generaldata.PatientID.isin(pid_list)], inplace=True)

static = []
for tbl in ['observrec', 'dervals']:
    vid_list = [str(val) for _, val in static_info[tbl].items()]
    df = []
    for batch_id in np.sort(chunking_info.BatchID.unique()):
        tmp_pid_list = np.array(chunking_info.index[chunking_info.BatchID==batch_id])
        chunkfile_path = os.path.join(hdf_path, tbl, '%s_%d_%d--%d.h5'%(tbl, 
                                                                        batch_id, 
                                                                        np.min(tmp_pid_list), 
                                                                        np.max(tmp_pid_list)))
        df.append(pd.read_hdf(chunkfile_path, where='VariableID in (%s)'%(','.join(vid_list))))
    df = pd.concat(df, axis=0)

    # remove invalidated records based on the status
    status_set = df.Status.unique()
    status_binary = ['{0:11b}'.format(s)[::-1] for s in status_set]
    # 1: invalidated; 5: notified but not measured; 
    invalid_status_set = status_set[np.where( [(x[1]=='1') or (x[5]=='1') for x in status_binary])]
    if len(invalid_status_set) > 0:
        df.drop(df.index[df.Status.isin(invalid_status_set)], inplace=True)

    if tbl == 'observrec':
        # delete records with values that are not 0 or 1 for emergency variable
        tmp = df[df.VariableID==14000140]
        if len(tmp) > 0:
            df.drop(tmp.index[np.logical_and(tmp.Value!=0, tmp.Value!=1)], axis=0, inplace=True)
    static.append(df)
static = pd.concat(static, axis=0)
static['VariableName'] = static.VariableID.apply(lambda x: id_name_mapping[x])

# Pivot the static table to static feature matrix
# Keep all records with different values for now 
static = pd.pivot_table(static, 
                        values='Value', 
                        index=['PatientID', 'Datetime', 'Entertime'],
                        columns=['VariableName'])


# Read the clean height information
df_height = []
for f in os.listdir(height_path):
    df_height.append(pd.read_hdf(os.path.join(height_path, f)))
df_height = pd.concat(df_height, axis=0)
df_height = df_height[df_height.PatientID.isin(pid_list)]
df_height.rename(columns={'DateTime': 'Datetime'}, inplace=True)
df_height = pd.pivot_table(df_height, values='Value', index=['PatientID', 'Datetime', 'Entertime'], columns='VariableID')
df_height.rename(columns={10000450: 'Height'}, inplace=True)
gc.collect()
print(len(df_height.reset_index().PatientID.unique()))
# Add heights to the static feature matrix
static = static.merge(df_height, left_index=True, right_index=True, how='outer')
static.reset_index(inplace=True)


# Add generaldata feature matrix to static feature matrix
static = generaldata.merge(static, how='outer', left_on='PatientID', right_on='PatientID')
del generaldata                      
gc.collect()

# Remove duplicates and only keep the latest records (the latest record can be correct of the previous ones instead of being dynamic)
static = static.sort_values(by=['PatientID', 'Datetime'])
static.drop_duplicates(set(static.columns) - {'Entertime', 'Datetime'}, inplace=True, keep='last')
gc.collect()

# Forward fill and backward fill the values 
pid_cnt = static.PatientID.value_counts()
total_n = len(pid_cnt.index[pid_cnt>1])
for nn, pid in enumerate(pid_cnt.index[pid_cnt>1]):
    static.loc[static.index[static.PatientID==pid]] = static[static.PatientID==pid].fillna(method='ffill')
    static.loc[static.index[static.PatientID==pid]] = static[static.PatientID==pid].fillna(method='bfill')

static.drop_duplicates(set(static.columns) - {'Entertime', 'Datetime'}, inplace=True, keep='last')

for col in ['Surgical', 'Emergency', 'Discharge', 'APACHECode', 'Euroscores', 'Height']:
    if col not in static.columns:
        continue
    # After removing imputed duplicates, there still exist multiple values for the patients, and I cannot decide
    # which one to use. It's better to set them `NaN` than picking the wrong one
    pid_cnt = static[['PatientID', col]].drop_duplicates().PatientID.value_counts()
    if np.sum(pid_cnt>1) > 0:
        pid_to_drop = pid_cnt.index[pid_cnt>1]
        static.loc[static.index[static.PatientID.isin(pid_to_drop)], col] = float('NaN')
    sys.stdout.write('# patients processed: %6d / %6d\r'%(nn+1, total_n))
    sys.stdout.flush()
static.drop_duplicates(set(static.columns) - {'Entertime', 'Datetime'}, inplace=True)
# Alternative: only keep the latest static records
static.drop(['Datetime', 'Entertime'], axis=1, inplace=True)

hirid_path = '/cluster/work/grlab/clinical/hirid2/pg_db_export'
metagroup_mapping = pd.read_csv('/cluster/work/grlab/clinical/hirid2/research/faltysm/volume_challenge/misc/apache_group.csv').iloc[:,:4]
metagroup_mapping.loc[:,'metavariable'] = metagroup_mapping.metavariable.astype(int)
metagroup_mapping_II = metagroup_mapping[['II', 'metavariable', 'Name']].set_index('II')
metagroup_mapping_IV = metagroup_mapping[['IV', 'metavariable', 'Name']].set_index('IV')

df_dict = dict()
for tbl in ['p_codeditem', 's_coderef', 's_codegroupref', 's_codesystemref']:
    df = []
    for f in os.listdir(os.path.join(hirid_path, tbl)):
        if 'parquet' not in f or 'crc' in f:
            continue
        df.append(pd.read_parquet(os.path.join(hirid_path, tbl, f), engine='pyarrow'))
    df = pd.concat(df, axis=0).reset_index(drop=True)
    df_dict.update({tbl: df.copy()})
    del df 
    gc.collect()

table_a=df_dict["s_coderef"].merge(df_dict["p_codeditem"], on="codeid",how="inner",suffixes=["_scoderef", "_pcodeditem"])
table_b=table_a.merge(df_dict["s_codegroupref"], on="groupid", how="inner", suffixes=["_scoderef/pcodeditem", "_scodegroupref"])
table_c=table_b[table_b["codesysid"].isin([1,17])]
table_d=table_c.sort_values(by="entertime")
unique_patients=sorted(table_d.patientid.unique())
print("Number of unique patients: {}".format(len(unique_patients)))
group_ids=list(table_d.groupid.unique())

for group_id in group_ids:
    df_group=table_d[table_d["groupid"]==group_id]
    example_item=list(df_group["name_scodegroupref"].unique())[0]
    print("Group: {}, Example item: {}, #patients={}".format(group_id, example_item, len(df_group.patientid.unique())))

table_d.loc[table_d.index[table_d.codesysid==1],'APACHEPatGroup'] = table_d.loc[table_d.index[table_d.codesysid==1],'groupid'].apply(lambda x: metagroup_mapping_II.loc[x].metavariable)
table_d.loc[table_d.index[table_d.codesysid==17],'APACHEPatGroup'] = table_d.loc[table_d.index[table_d.codesysid==17],'groupid'].apply(lambda x: metagroup_mapping_IV.loc[x].metavariable)
table_d.drop(table_d.index[(table_d.entertime - table_d['archtime_scoderef/pcodeditem']) / np.timedelta64(1,'s') < 0], inplace=True) 
table_d.drop(table_d.index[(table_d['archtime_scoderef/pcodeditem'] - table_d['archtime_scodegroupref']) / np.timedelta64(1,'s') < 0], inplace=True)
table_d.sort_values(['patientid', 'starttime', 'entertime', 'archtime_scoderef/pcodeditem', 'archtime_scodegroupref'], inplace=True)
table_d.drop_duplicates(['patientid', 'starttime', 'entertime'], keep='last', inplace=True)
pid_tmp = table_d.patientid.value_counts().index[table_d.patientid.value_counts()>1].values
for pid in pid_tmp:
    table_d_tmp = table_d[table_d.patientid==pid]
    groupid_counts = table_d_tmp.groupid.value_counts()
    if table_d_tmp.iloc[0].groupid == groupid_counts.index[0]:
        table_d.drop(table_d_tmp.index[1:], inplace=True)
    else:
        groupid2keep = groupid_counts.index[0]
        table_d.drop(list(set(table_d_tmp.index.values) - set([table_d_tmp.index[np.where(table_d_tmp.groupid.values==groupid2keep)[0][0]]])), inplace=True)
        
    table_d_tmp = table_d[table_d.patientid==pid]
    try:
        assert(len(table_d_tmp)==1)
    except:
        import ipdb
        ipdb.set_trace()
        
static = static.merge(table_d[['patientid', 'APACHEPatGroup']], how='left', left_on='PatientID', right_on='patientid')
static.drop('patientid', axis=1, inplace=True)

# static.to_hdf('/cluster/work/grlab/clinical/hirid2/research/1a_hdf5_clean/v8/static.h5', 'data', complib='blosc:lz4', complevel=5, data_columns=True, format='table')

# # integrate the APACHE group information from Matthias. 
# tmp = open(os.path.join(preproc_utils.datapath, 'misc_derived', 'apachegroup_patients_180130.tsv'), 'r').readlines()
# df_apachegroup  = []
# for i, line in enumerate(tmp):
#     if i > 0:
#         df_apachegroup.append([float(x) for x in line.rstrip('\n').split('\t')])
# df_apachegroup = pd.DataFrame(df_apachegroup, columns=['PatientID', 'APACHEPatGroup'])
# df_apachegroup['PatientID'] = df_apachegroup.PatientID.astype(np.int64)
# df_apachegroup['APACHEPatGroup'] = df_apachegroup.APACHEPatGroup.astype(np.int64)
# df_apachegroup.set_index('PatientID', inplace=True)

# static.set_index('PatientID', inplace=True)
# static = static.merge(df_apachegroup, how='left', left_index=True, right_index=True)
# static.reset_index(inplace=True)
# static.to_hdf(os.path.join(preproc_utils.datapath, '1a_hdf5_clean', version, 'static.h5'), 'data', 
#               complevel=5, complib='blosc:lz4', data_columns=True, format='table')
    
