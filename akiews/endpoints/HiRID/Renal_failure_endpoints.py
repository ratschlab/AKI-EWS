import pandas as pd
import numpy as np
import datetime as dt
import glob
import os.path
import os
from os.path import join
import csv
# from tqdm import tqdm

from KDIGO import KDIGO
# from AKI import onset_AKI
from load_3_merged import extract_p_HiRID, extract_batch_HiRID
from load_0_parquet import get_catheter_times

import pickle
import multiprocessing
from functools import partial
from itertools import repeat
# import time

from paths import *


def drug_given_time(p_dfv):
    orig = p_dfv[[ep_drugs] + ['PatientID', time_hirid_key]]
    re_pid = orig.copy()
    # for var in ep_drugs:
    #     re_pid[var] = re_pid[var] > 0
    # re_pid['ep_drug_combined'] = np.any(re_pid[ep_drugs] > 0, axis=1)

    orig[time_hirid_key] = pd.to_datetime(orig[time_hirid_key])
    orig = orig.set_index(time_hirid_key).resample('1 h').sum()
    # print(orig)
    # orig[ep_drugs] = np.logical_and(orig[ep_drugs] == 0, orig[ep_drugs].cumsum() > 0)
    # print(orig)
    # orig[ep_drugs] = orig[ep_drugs].astype(int).diff() == 1
    # print(orig)
    times = orig[orig[ep_drugs]>0].index.values#orig[np.any(orig[ep_drugs].astype(int).diff() == 1, axis=1)].index.values
    # print(orig)
    print(times)
    re_pid['masked_urine'] = False
    if len(times) > 0:
        for time in times:
            re_pid['masked_urine'] = np.logical_or(re_pid['masked_urine'],np.logical_and(re_pid[time_hirid_key] >= time, re_pid[time_hirid_key] < time + pd.Timedelta('8 h')))
            print(np.logical_or(re_pid['masked_urine'],np.logical_and(re_pid[time_hirid_key] >= time, re_pid[time_hirid_key] < time + pd.Timedelta('8 h'))))

    # print(re_pid)
    # print()

    return re_pid


def fill_geq1(row):
    if np.any([row[i] == 1 for i in one_columns + two_columns + three_columns]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns]):
        return 0
    else:
        return -1


def fill_geq1_creatinine(row):
    if np.any([row[i] == 1 for i in one_columns_creatinine + two_columns_creatinine + three_columns_creatinine]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns_creatinine]):
        return 0
    else:
        return -1


def fill_geq1_urine(row):
    if np.any([row[i] == 1 for i in one_columns_urine + two_columns_urine + three_columns_urine]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns_urine]):
        return 0
    else:
        return -1


def merge_gaps(df, min_gap, column_name='geq1'):
    prev_ep_time = pd.NaT
    start_region = False
    start_time = pd.NaT
    for _, row in df.iterrows():
        index=row[time_hirid_key]
        if row[column_name] != 1:
            start_region = True
        if row[column_name] == 1:
            if start_region:
                if index - prev_ep_time < min_gap:
                    print('Merged')
                    df.loc[prev_ep_time:index, column_name] = 1
            else:
                prev_ep_time = index
            start_region = False
    return df


def fill_geq2(row):
    if np.any([row[i] == 1 for i in two_columns + three_columns]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns + two_columns]):
        return 0
    else:
        return -1


def fill_geq2_creatinine(row):
    if np.any([row[i] == 1 for i in two_columns_creatinine + three_columns_creatinine]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns_creatinine + two_columns_creatinine]):
        return 0
    else:
        return -1


def fill_geq2_urine(row):
    if np.any([row[i] == 1 for i in two_columns_urine + three_columns_urine]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns_urine + two_columns_urine]):
        return 0
    else:
        return -1


def fill_geq3(row):
    if np.any([row[i] == 1 for i in three_columns]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns + two_columns + three_columns]):
        return 0
    else:
        return -1


def fill_geq3_creatinine(row):
    if np.any([row[i] == 1 for i in three_columns_creatinine]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns_creatinine + two_columns_creatinine + three_columns_creatinine]):
        return 0
    else:
        return -1



def fill_geq3_urine(row):
    if np.any([row[i] == 1 for i in three_columns_urine]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns_urine + two_columns_urine + three_columns_urine]):
        return 0
    else:
        return -1


def fill_endpoint_status(row):
    if np.any([row[i] == 1 for i in three_columns]):
        return '3'
    elif np.any([row[i] == 1 for i in two_columns]):
        return '2'
    elif np.any([row[i] == 1 for i in one_columns]):
        return '1'
    elif np.all([row[i] == 0 for i in one_columns]):
        return '0'
    else:
        return 'unknown'


def same_ep(LOO_vector):
    return [i if i in [-1, 0, 1] else -2 for i in LOO_vector]


def process_KDIGO_LOO(pid, p_dfv, kdigo_stages, halflife, c_base, min_gap):
    print(pid)
    p_dfv_backup_to_LOO = p_dfv[p_dfv[time_hirid_key] > p_dfv.dropna(subset=[hr_hirid_key], how='any')[time_hirid_key].min()]
    p_dfv_backup_to_LOO = p_dfv_backup_to_LOO[p_dfv_backup_to_LOO[time_hirid_key] <= p_dfv_backup_to_LOO.dropna(subset=[hr_hirid_key], how='any')[time_hirid_key].max()]
    p_dfv_backup = p_dfv
    KDIGO_events = KDIGO(p_dfv, pid, halflife, c_base, min_gap, pid in pid_of_interest)

    p_dfv_drugs = drug_given_time(p_dfv_backup)
    print(pd.merge_asof(KDIGO_events, p_dfv_drugs.drop(columns=['PatientID']),on=time_hirid_key, tolerance=gridsize))
    KDIGO_events = pd.merge_asof(KDIGO_events, p_dfv_drugs.drop(columns=['PatientID']),on=time_hirid_key, tolerance=gridsize)

    KDIGO_events = KDIGO_events.convert_objects()
    KDIGO_geq1 = KDIGO_events.apply(fill_geq1, axis=1)
    KDIGO_geq2 = KDIGO_events.apply(fill_geq2, axis=1)
    KDIGO_geq3 = KDIGO_events.apply(fill_geq3, axis=1)

    KDIGO_geq1_creatinine = KDIGO_events.apply(fill_geq1_creatinine, axis=1)
    KDIGO_geq2_creatinine = KDIGO_events.apply(fill_geq2_creatinine, axis=1)
    KDIGO_geq3_creatinine = KDIGO_events.apply(fill_geq3_creatinine, axis=1)


    KDIGO_geq1_urine = KDIGO_events.apply(fill_geq1_urine, axis=1)
    KDIGO_geq2_urine = KDIGO_events.apply(fill_geq2_urine, axis=1)
    KDIGO_geq3_urine = KDIGO_events.apply(fill_geq3_urine, axis=1)


    if len(KDIGO_geq1) > 0:
        # LOO_geq1_ur = KDIGO_geq1
        # LOO_geq2_ur = KDIGO_geq2
        # patient = p_dfv_backup_to_LOO.dropna(subset=[urine_hirid_key], how='any')
        # print('ur', len(patient.index))
        # for i in patient.index:
        #     p_dfv_loo = p_dfv_backup.drop(i)
        #     # print(p_dfv)
        #     KDIGO_events_LOO = KDIGO(p_dfv_loo, pid, halflife, c_base, min_gap, pid in pid_of_interest)
        #     KDIGO_events_LOO = KDIGO_events_LOO.convert_objects()
        #     KDIGO_events_LOO_geq1 = KDIGO_events_LOO.apply(fill_geq1, axis=1)
        #     KDIGO_events_LOO_geq2 = KDIGO_events_LOO.apply(fill_geq2, axis=1)
        #     # add 0.5 if the entries do not match (for additional i do not add more 0.5s)
        #     if len(KDIGO_events_LOO_geq1) == len(KDIGO_geq1):
        #         LOO_geq1_ur = np.logical_and(np.logical_and((KDIGO_events_LOO_geq1 != KDIGO_geq1), (KDIGO_events_LOO_geq1 != -1)), (KDIGO_geq1 == LOO_geq1_ur)) / 2 + LOO_geq1_ur
        #         LOO_geq2_ur = np.logical_and(np.logical_and((KDIGO_events_LOO_geq2 != KDIGO_geq2), (KDIGO_events_LOO_geq2 != -1)), (KDIGO_geq2 == LOO_geq2_ur)) / 2 + LOO_geq2_ur

        # LOO_geq1_cr = KDIGO_geq1
        # LOO_geq2_cr = KDIGO_geq2
        # patient = p_dfv_backup_to_LOO.dropna(subset=[creatinine_hirid_key], how='any')
        # print('cr', len(patient.index))
        # for i in patient.index:
        #     p_dfv_loo = p_dfv_backup.drop(i)
        #     # print(p_dfv)
        #     KDIGO_events_LOO = KDIGO(p_dfv_loo, pid, halflife, c_base, min_gap, pid in pid_of_interest)
        #     KDIGO_events_LOO = KDIGO_events_LOO.convert_objects()
        #     KDIGO_events_LOO_geq1 = KDIGO_events_LOO.apply(fill_geq1, axis=1)
        #     KDIGO_events_LOO_geq2 = KDIGO_events_LOO.apply(fill_geq2, axis=1)
        #     # add 0.5 if the entries do not match (for additional i do not add more 0.5s)
        #     if len(KDIGO_events_LOO_geq1) == len(KDIGO_geq1):
        #         LOO_geq1_cr = np.logical_and(np.logical_and((KDIGO_events_LOO_geq1 != KDIGO_geq1), (KDIGO_events_LOO_geq1 != -1)), (KDIGO_geq1 == LOO_geq1_ur)) / 2 + LOO_geq1_ur
        #         LOO_geq2_cr = np.logical_and(np.logical_and((KDIGO_events_LOO_geq2 != KDIGO_geq2), (KDIGO_events_LOO_geq2 != -1)), (KDIGO_geq2 == LOO_geq2_ur)) / 2 + LOO_geq2_ur

        # LOO_geq1_rt = KDIGO_geq1
        # LOO_geq2_rt = KDIGO_geq2
        # patient = p_dfv_backup_to_LOO.dropna(subset=dialysis_hirid_keys, how='all')
        # print('rrt', len(patient.index))
        # for i in patient.index:
        #     p_dfv_loo = p_dfv_backup.drop(i)
        #     # print(p_dfv)
        #     KDIGO_events_LOO = KDIGO(p_dfv_loo, pid, halflife, c_base, min_gap, pid in pid_of_interest)
        #     KDIGO_events_LOO = KDIGO_events_LOO.convert_objects()
        #     KDIGO_events_LOO_geq1 = KDIGO_events_LOO.apply(fill_geq1, axis=1)
        #     KDIGO_events_LOO_geq2 = KDIGO_events_LOO.apply(fill_geq2, axis=1)
        #     # add 0.5 if the entries do not match (for additional i do not add more 0.5s)
        #     if len(KDIGO_events_LOO_geq1) == len(KDIGO_geq1):
        #         LOO_geq1_rt = np.logical_and(np.logical_and((KDIGO_events_LOO_geq1 != KDIGO_geq1), (KDIGO_events_LOO_geq1 != -1)), (KDIGO_geq1 == LOO_geq1_ur)) / 2 + LOO_geq1_ur
        #         LOO_geq2_rt = np.logical_and(np.logical_and((KDIGO_events_LOO_geq2 != KDIGO_geq2), (KDIGO_events_LOO_geq2 != -1)), (KDIGO_geq2 == LOO_geq2_ur)) / 2 + LOO_geq2_ur

        KDIGO_events['geq1'] = KDIGO_geq1
        KDIGO_events['geq2'] = KDIGO_geq2
        KDIGO_events['geq3'] = KDIGO_geq3
        KDIGO_events['geq1_creatinine'] = KDIGO_geq1_creatinine
        KDIGO_events['geq2_creatinine'] = KDIGO_geq2_creatinine
        KDIGO_events['geq3_creatinine'] = KDIGO_geq3_creatinine
        KDIGO_events['geq1_urine'] = KDIGO_geq1_urine
        KDIGO_events['geq2_urine'] = KDIGO_geq2_urine
        KDIGO_events['geq3_urine'] = KDIGO_geq3_urine
        KDIGO_events = merge_gaps(KDIGO_events, min_gap, 'geq1')
        KDIGO_events = merge_gaps(KDIGO_events, min_gap, 'geq2')
        KDIGO_events = merge_gaps(KDIGO_events, min_gap, 'geq1_creatinine')
        KDIGO_events = merge_gaps(KDIGO_events, min_gap, 'geq2_creatinine')
        KDIGO_events = merge_gaps(KDIGO_events, min_gap, 'geq1_urine')
        KDIGO_events = merge_gaps(KDIGO_events, min_gap, 'geq2_urine')
        KDIGO_events['endpoint_status'] = KDIGO_events.apply(fill_endpoint_status, axis=1)
        # KDIGO_events['LOO_ur_geq1'] = np.nan  # same_ep(LOO_geq1_ur)  # -2 if they do not match
        # KDIGO_events['LOO_ur_geq2'] = np.nan  # same_ep(LOO_geq2_ur)
        # KDIGO_events['LOO_cr_geq1'] = np.nan  # same_ep(LOO_geq1_cr)
        # KDIGO_events['LOO_cr_geq2'] = np.nan  # same_ep(LOO_geq2_cr)
        # KDIGO_events['LOO_rt_geq1'] = np.nan  # same_ep(LOO_geq1_rt)
        # KDIGO_events['LOO_rt_geq2'] = np.nan  # same_ep(LOO_geq2_rt)
        KDIGO_events = KDIGO_events.reset_index()
        KDIGO_events.loc[:, time_hirid_key] = pd.to_datetime(KDIGO_events[time_hirid_key].values)
        KDIGO_events[time_endpoint_key] = pd.to_datetime(KDIGO_events[time_hirid_key].values)
        KDIGO_events['cath'] = KDIGO_events.apply(fill_cath, axis=1)

        #print(KDIGO_events)
        kdigo_stages[pid] = KDIGO_events



        #print(kdigo_stages[pid])


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


pid_of_interest = [99119, 94544, 63842, 44201, 101241, 87794, 103398, 77037]


def process_KDIGO(pid, p_dfv, kdigo_stages, halflife, c_base, min_gap):
    KDIGO_events = KDIGO(p_dfv, pid, halflife, c_base, min_gap, pid in pid_of_interest)
    KDIGO_events['geq1'] = KDIGO_events.apply(fill_geq1, axis=1)
    KDIGO_events['geq2'] = KDIGO_events.apply(fill_geq2, axis=1)
    KDIGO_events['geq3'] = KDIGO_events.apply(fill_geq3, axis=1)
    KDIGO_events['geq1_creatinine'] = KDIGO_events.apply(fill_geq1_creatinine, axis=1)
    KDIGO_events['geq2_creatinine'] = KDIGO_events.apply(fill_geq2_creatinine, axis=1)
    KDIGO_events['geq3_creatinine'] = KDIGO_events.apply(fill_geq3_creatinine, axis=1)
    KDIGO_events['geq1_urine'] = KDIGO_events.apply(fill_geq1_urine, axis=1)
    KDIGO_events['geq2_urine'] = KDIGO_events.apply(fill_geq2_urine, axis=1)
    KDIGO_events['geq3_urine'] = KDIGO_events.apply(fill_geq3_urine, axis=1)
    KDIGO_events['endpoint_status'] = KDIGO_events.apply(fill_endpoint_status, axis=1)
    KDIGO_events['cath'] = KDIGO_events.apply(fill_cath, axis=1)
    kdigo_stages[pid] = KDIGO_events
    # to_print = p_dfv.dropna(how='all', subset=[urine_hirid_key, creatinine_hirid_key])
    # print(pid)
    # print(to_print)
    # print(KDIGO_events)


def fill_cath(row):
    pid = row['PatientID']
    cath_times = get_catheter_times(pid)
    cath_status = False
    for index, rows in cath_times.iterrows():
        if row[time_hirid_key] > pd.to_datetime(rows['starttime']) and not row[time_hirid_key] > pd.to_datetime(rows['endtime']):
            cath_status = True
    return cath_status


def process_KDIGO_batch(batch, to_file=True, halflife=8, c_base='literature', gap=6):
    kdigo_stages = {}
    pdf = extract_batch_HiRID(batch)
    path_to_save = base_ep  # + 'merging' + str(gap) + 'h/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    grouped = pdf.groupby('PatientID')
    dflist = []
    pdlist = []
    for name, group in grouped:
        dflist.append(group)
        pdlist.append(name)
    print('Number patients in data: ', len(pdlist))

    # manager = multiprocessing.Manager()
    # kdigo_stages = manager.dict()

    # chunksize = 50
    # pool_iterator = 0
    # while chunksize * (pool_iterator + 1) < len(dflist):
    #     print(chunksize, pool_iterator)
    #     with manager.Pool() as pool:
    #         pool.starmap(process_KDIGO_LOO, zip(pdlist[chunksize * pool_iterator:chunksize * (pool_iterator + 1)], dflist[chunksize * pool_iterator:chunksize * (pool_iterator + 1)], repeat(kdigo_stages, chunksize), repeat(halflife, chunksize), repeat(c_base, chunksize), repeat(int(gap) * 60 * reference_time, chunksize)))
    #     pool_iterator = pool_iterator + 1
    #     print('Pool loop', pool_iterator)  # , dict(kdigo_stages))
    # with manager.Pool() as pool:
    #     pool.starmap(process_KDIGO_LOO, zip(pdlist[chunksize * pool_iterator:], dflist[chunksize * pool_iterator:], repeat(kdigo_stages, len(dflist[chunksize * pool_iterator:])), repeat(halflife, len(dflist[chunksize * pool_iterator:])), repeat(c_base, len(dflist[chunksize * pool_iterator:])), repeat(int(gap) * 60 * reference_time, len(dflist[chunksize * pool_iterator:]))))
    # kdigo_stages = dict(kdigo_stages)

    kdigo_stages = {}
    i_pid = 0
    for pid in pdf['PatientID'].unique():
       i_pid += 1
       print(pid)
       p_dfv = pdf[pdf['PatientID'] == pid]
       process_KDIGO_LOO(pid, p_dfv, kdigo_stages, halflife, c_base, int(gap) * 60 * reference_time)
       if i_pid > 10:
           break

    df = pd.DataFrame(columns=[time_hirid_key, time_endpoint_key, 'PatientID'] + list(ep_types) + ['geq1', 'geq2', 'geq3','geq1_creatinine', 'geq2_creatinine', 'geq3_creatinine','geq1_urine', 'geq2_urine', 'geq3_urine', 'endpoint_status', 'cath'] + [ep_drugs] )
    print('Number patients in dictionary: ', len(kdigo_stages.keys()))
    for pid in list(kdigo_stages.keys()):
        if len(df) == 0:
            df = kdigo_stages[pid]
            # df = pd.merge(df, drug_given_time(batch, pid), on=['PatientID', time_hirid_key])
        else:

            df_p = kdigo_stages[pid] #pd.merge(kdigo_stages[pid], drug_given_time(batch, pid), on=['PatientID', time_hirid_key])
            df = pd.concat([df, df_p]).reset_index(drop=True)
        print(df)

    df['PatientID'] = df['PatientID'].astype('int32')
    print('Number patients in hdf: ', len(np.unique(df['PatientID'])))
    outfile = path_to_save + 'batch_' + str(batch) + '.h5'

    # df = pd.read_hdf(outfile, key='/endpoints')
    # Conversion of columns
    df['unknown_geq1'] = df['geq1'] == -1
    df['unknown_geq2'] = df['geq2'] == -1
    df['unknown_geq3'] = df['geq2'] == -1
    df['geq1'] = df['geq1'] == 1
    df['geq2'] = df['geq2'] == 1
    df['geq3'] = df['geq3'] == 1
    df['unknown_geq1_creatinine'] = df['geq1_creatinine'] == -1
    df['unknown_geq2_creatinine'] = df['geq2_creatinine'] == -1
    df['unknown_geq3_creatinine'] = df['geq2_creatinine'] == -1
    df['unknown_geq1_urine'] = df['geq1_urine'] == -1
    df['unknown_geq2_urine'] = df['geq2_urine'] == -1
    df['unknown_geq3_urine'] = df['geq2_urine'] == -1
    df['geq1_creatinine'] = df['geq1_creatinine'] == 1
    df['geq2_creatinine'] = df['geq2_creatinine'] == 1
    df['geq3_creatinine'] = df['geq3_creatinine'] == 1
    df['geq1_urine'] = df['geq1_urine'] == 1
    df['geq2_urine'] = df['geq2_urine'] == 1
    df['geq3_urine'] = df['geq3_urine'] == 1
    df['masked_urine'] = df['masked_urine'].astype(str)=='True'
    # df['uncertain_ur_geq1'] = df['LOO_ur_geq1'] != -2
    # df['uncertain_ur_geq2'] = df['LOO_ur_geq2'] != -2
    # df['uncertain_cr_geq1'] = df['LOO_cr_geq1'] != -2
    # df['uncertain_cr_geq2'] = df['LOO_cr_geq2'] != -2
    # df['uncertain_rt_geq1'] = df['LOO_rt_geq1'] != -2
    # df['uncertain_rt_geq2'] = df['LOO_rt_geq2'] != -2

    df = df.convert_objects()
    print(df)
    print(df.dtypes)
    to_drop = ['pm74', 'pm75','pm88','pm91',ep_drugs]
    for c in to_drop:
        if c in df.columns.values:
           df = df.drop(c,axis=1)
           print('dropped', c)
    print(df.dtypes)
    df = df.convert_objects()
    print(df['masked_urine'])

    if to_file:
        # df.to_hdf(outfile, 'endpoints', append=False, complevel=5, complib='blosc:lz4', data_columns=['PatientID'], format='fixed')
        df.to_hdf(outfile, 'endpoints', append=False, complevel=5, complib='blosc:lz4', data_columns=['PatientID'], format='table')
        reread = pd.read_hdf(outfile, key='/endpoints')
        print(reread)


import sys
from load_3_merged import get_batch

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Event based evaluation")
    parser.add_argument('--halflife_urine', type=int, default=8)  # help='ewma halflife in hours',
    parser.add_argument('--gap', type=int, default=6)  # help=merging of events in hours
    parser.add_argument('--baseline_creatinine', type=str, default='literature')
    parser.add_argument('--number', type=int, default=0)
    parser.add_argument('--output_to_disk', type=bool, default=True)
    args = parser.parse_args()

    process_KDIGO_batch(args.number, args.output_to_disk, args.halflife_urine, args.baseline_creatinine, args.gap)
