import pandas as pd
import numpy as np
import datetime as dt
import glob
import os.path
import os
from os.path import join
import csv

from load_mimic import get_creatinine_pid, get_urine_pid, get_static_pid, get_RRT_pid, get_all_p_icu
from KDIGO import KDIGO
# from AKI import onset_AKI

import pickle
import multiprocessing
from functools import partial
from itertools import repeat
# import time

from paths import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



def fill_geq2_creatinine(row):
    if np.any([row[i] == 1 for i in two_columns_creatinine + three_columns_creatinine]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns_creatinine + two_columns_creatinine]):
        return 0
    else:
        return -2


def fill_geq2_urine(row):
    if np.any([row[i] == 1 for i in two_columns_urine + three_columns_urine]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns_urine + two_columns_urine]):
        return 0
    else:
        return -2



def fill_geq3(row):
    if np.any([row[i] == 1 for i in three_columns]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns + two_columns + three_columns]):
        return 0
    else:
        return -2


def fill_geq3_creatinine(row):
    if np.any([row[i] == 1 for i in three_columns_creatinine]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns_creatinine + two_columns_creatinine + three_columns_creatinine]):
        return 0
    else:
        return -2



def fill_geq3_urine(row):
    if np.any([row[i] == 1 for i in three_columns_urine]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns_urine + two_columns_urine + three_columns_urine]):
        return 0
    else:
        return -2


def fill_geq1_creatinine(row):
    if np.any([row[i] == 1 for i in one_columns_creatinine + two_columns_creatinine + three_columns_creatinine]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns_creatinine]):
        return 0
    else:
        return -2


def fill_geq1_urine(row):
    if np.any([row[i] == 1 for i in one_columns_urine + two_columns_urine + three_columns_urine]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns_urine]):
        return 0
    else:
        return -2

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


def fill_geq1(row):
    if np.any([row[i] == 1 for i in one_columns + two_columns + three_columns]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns]):
        return 0
    else:
        return -2


def fill_geq2(row):
    if np.any([row[i] == 1 for i in two_columns + three_columns]):
        return 1
    elif np.all([row[i] == 0 for i in one_columns + two_columns]):
        return 0
    else:
        return -2


def process_KDIGO(pid, kdigo_stages, halflife, c_base, gap):
    # print(pid)
    stat = get_static_pid(pid)
    ethn, ag, gen, in_dt, out_dt = stat[0]
    cr_p = get_creatinine_pid(pid)
    ur_p = get_urine_pid(pid)
    rrt_p = get_RRT_pid(pid)
    KDIGO_events = KDIGO(pid, cr_p, ur_p, ethn, ag, gen, in_dt, out_dt, rrt_p, halflife, c_base, gap)
    kdigo_stages[pid] = KDIGO_events.reset_index().rename(columns={'index': time_key})


def process_KDIGO_batch(batch, to_file=True, halflife=0, c_base='min_i', gap=6):
    all_p = get_all_p_icu()
    if batch > -1:
        all_p = np.asarray(all_p[int((len(all_p) / 50) * batch):int((len(all_p) / 50) * (batch + 1))])
    path_to_save = base_ep + '/220821/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    manager = multiprocessing.Manager()
    kdigo_stages = manager.dict()

    chunksize = 100
    pool_iterator = 0
    while chunksize * (pool_iterator + 1) < len(all_p):
        print(chunksize, pool_iterator, all_p[chunksize * pool_iterator:chunksize * (pool_iterator + 1)])
        with manager.Pool() as pool:
            pool.starmap(process_KDIGO, zip(all_p[chunksize * pool_iterator:chunksize * (pool_iterator + 1)], repeat(kdigo_stages, chunksize), repeat(halflife, chunksize), repeat(c_base, chunksize), repeat(gap, chunksize)))
        pool_iterator = pool_iterator + 1
        print('Pool loop', pool_iterator)  # , dict(kdigo_stages))
    with manager.Pool() as pool:
        pool.starmap(process_KDIGO, zip(all_p[chunksize * pool_iterator:], repeat(kdigo_stages, len(all_p[chunksize * pool_iterator:])), repeat(halflife, len(all_p[chunksize * pool_iterator:])), repeat(c_base, len(all_p[chunksize * pool_iterator:])),repeat(gap, len(all_p[chunksize * pool_iterator:]))))
    kdigo_stages = dict(kdigo_stages)

    print('start concat')
    df = pd.DataFrame(columns=[time_key, patient_key] + list(ep_types))
    for pid in list(kdigo_stages.keys()):
        df = pd.concat([df, kdigo_stages[pid]])

    print('done concat')
    print(df)
    df = df.reset_index(drop=True)
    df = df.infer_objects()
    df[patient_key] = df[patient_key].astype('int32')
    print(df[time_key])
    print(df.columns.values.tolist())
    df[time_key] = pd.to_datetime(df[time_key].values)
    outfile = path_to_save + 'batch_'+str(batch)+'.h5'
    print(df[time_key])
    print(outfile)
    print(df)

    df['endpoint_status'] = df.apply(fill_endpoint_status, axis=1)
    df['geq1'] = df.apply(fill_geq1, axis=1)
    df['geq2'] = df.apply(fill_geq2, axis=1)
    df['geq3'] = df.apply(fill_geq3, axis=1)
    df['geq1_urine'] = df.apply(fill_geq1_urine, axis=1)
    df['geq2_urine'] = df.apply(fill_geq2_urine, axis=1)
    df['geq3_urine'] = df.apply(fill_geq3_urine, axis=1)
    df['geq1_creatinine'] = df.apply(fill_geq1_creatinine, axis=1)
    df['geq2_creatinine'] = df.apply(fill_geq2_creatinine, axis=1)
    df['geq3_creatinine'] = df.apply(fill_geq3_creatinine, axis=1)
    df['unknown_geq1'] = df['geq1'] == -1
    df['unknown_geq2'] = df['geq2'] == -1
    df['unknown_geq3'] = df['geq2'] == -1
    df['unknown_geq1_creatinine'] = df['geq1_creatinine'] == -1
    df['unknown_geq2_creatinine'] = df['geq2_creatinine'] == -1
    df['unknown_geq3_creatinine'] = df['geq2_creatinine'] == -1
    df['unknown_geq1_urine'] = df['geq1_urine'] == -1
    df['unknown_geq2_urine'] = df['geq2_urine'] == -1
    df['unknown_geq3_urine'] = df['geq2_urine'] == -1
    df = df.infer_objects()
    print(df)

    df.to_hdf(outfile, 'endpoints', append=False, complevel=5, complib='blosc:lz4', data_columns=[patient_key], format='table')
    reread = pd.read_hdf(outfile, key='/endpoints')
    print(reread)
    print(reread.columns.values.tolist())

    # with open(path_to_save + 'KDIGO_stages' + str(batch) + '.pkl', 'wb') as handle:
    #     pickle.dump(kdigo_stages, handle, protocol=pickle.HIGHEST_PROTOCOL)


import sys

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Event based evaluation")
    parser.add_argument('--halflife_urine', type=int, default=0)  # help='ewma halflife in hours',
    parser.add_argument('--gap', type=int, default=0)  # help=merging of events in hours
    parser.add_argument('--baseline_creatinine', type=str, default='min_i')
    parser.add_argument('--number', type=int, default=0)
    parser.add_argument('--output_to_disk', type=bool, default=True)
    args = parser.parse_args()

    process_KDIGO_batch(args.number, args.output_to_disk, args.halflife_urine, args.baseline_creatinine, args.gap)
