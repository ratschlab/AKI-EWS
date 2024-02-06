from paths import *
import numpy as np
import pandas as pd

ep_types = np.asarray(['1.u', '1.i', '1.b', '2.u', '2.b', '3.au', '3.u', '3.b', '3.i', '3.d'])
endpoint_status_mapping = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 3, 3])
ff_ep_types = np.asarray([24 * 60 * pd.Timedelta('1 min'), 24 * 60 * pd.Timedelta('1 min'), 24 * 60 * pd.Timedelta('1 min'), 24 * 60 * pd.Timedelta('1 min'), 24 * 60 * pd.Timedelta('1 min'), 24 * 60 * pd.Timedelta('1 min'), 24 * 60 * pd.Timedelta('1 min'), 24 * 60 * pd.Timedelta('1 min'), 24 * 60 * pd.Timedelta('1 min'), 4 * 7 * 24 * 60 * pd.Timedelta('1 min')])

import pickle


reference_time = pd.Timedelta(np.timedelta64(1, 'm'))
from glob import glob
KDIGO_dict = {}
for file_d in glob(base_ep_mimic + 'KDIGO_stages*.pkl'):
    KDIGO_dict = {**KDIGO_dict, **pickle.load(open(file_d, 'rb'))}
    # break
MIMIC_concept_KDIGO = pd.read_csv(base_ep_mimic + 'KDIGO_concept.csv', index_col=0)
print([KDIGO_dict[k] for k in list(KDIGO_dict.keys())[:50]])


def from_list_column(KDIGO_list, window_centre, which):
    condition = -1
    for i, ep in enumerate(KDIGO_list):
        if len(ep) > 2:
            # print(ep, which)
            if ep[2] == ep_types[which]:
                if ep[1] > window_centre - ff_ep_types[which] and ep[1] <= window_centre:
                    # overwrite invalid states but forward fill other states
                    if ep[-1] >= 0 and condition >= 0:
                        condition = ep[-1]
                    elif condition < 0:
                        condition = ep[-1]
            if ep[1] > window_centre:
                break
    # if condition == -1:
    #     condition = np.nan
    return condition


def map_pid(pid):
    '''
    given pid from endpointfiles (icustayid) , get subject id
    '''
    # print(pid, np.unique(MIMIC_concept_KDIGO[MIMIC_concept_KDIGO['icustay_id'] == pid]['subject_id'].values))
    try:
        return int(MIMIC_concept_KDIGO[MIMIC_concept_KDIGO['icustay_id'] == pid]['subject_id'].values[0])
    except:
        return 0


def fill_df(dfc):
    prev_pid = np.nan
    pid = np.nan
    for index, row in dfc.iterrows():
        prev_pid = pid
        time = row[time_hirid_key]
        pid = row['PatientID']
        new_pid = map_pid(pid)
        # print(pid, type(pid), pid in KDIGO_dict.keys(), new_pid, type(new_pid), type(list(KDIGO_dict.keys())[0]), new_pid in KDIGO_dict.keys())
        # break
        if new_pid in list(KDIGO_dict.keys()):
            KDIGO_list = KDIGO_dict[new_pid]
            for idx_ep_type in range(len(ep_types)):
                dfc.loc[index, ep_types[idx_ep_type]] = from_list_column(KDIGO_list, time, idx_ep_type)
            stages_false = endpoint_status_mapping[dfc.loc[index, ep_types].values == 0]
            stages_true = endpoint_status_mapping[dfc.loc[index, ep_types].values == 1]
            stages_invalid = endpoint_status_mapping[dfc.loc[index, ep_types].values < 0]

            # print(stages_true, stages_false)
            if len(stages_true) > 0:
                # If higher KIDGO stages are true, lower stages nan are not important
                ep_st = str(np.nanmax(stages_true))

                if len([s for s in stages_true if s > float(ep_st)]) + len([s for s in stages_false if s > float(ep_st)]) == len([s for s in endpoint_status_mapping if s > float(ep_st)]):
                    dfc.loc[index, 'endpoint_status'] = ep_st
                    if np.nanmax(stages_true) <= 1:
                        dfc.loc[index, 'leq1'] = 1
                        dfc.loc[index, 'leq2'] = 1
                        dfc.loc[index, 'leq0'] = 0
                    elif np.nanmax(stages_true) <= 2:
                        dfc.loc[index, 'leq2'] = 1
                        dfc.loc[index, 'leq1'] = 0
                        dfc.loc[index, 'leq0'] = 0
                    else:
                        dfc.loc[index, 'leq2'] = 0
                        dfc.loc[index, 'leq1'] = 0
                        dfc.loc[index, 'leq0'] = 0
                    # Smaller equal only possible if all states are known
                if np.nanmax(stages_true) <= 1:
                    dfc.loc[index, 'geq1'] = 1
                    dfc.loc[index, 'geq2'] = 0
                    dfc.loc[index, 'geq3'] = 0
                    dfc.loc[index, 'leq0'] = 0
                elif np.nanmax(stages_true) <= 2:
                    dfc.loc[index, 'geq1'] = 1
                    dfc.loc[index, 'leq1'] = 0
                    dfc.loc[index, 'geq2'] = 1
                    dfc.loc[index, 'geq3'] = 0
                    dfc.loc[index, 'leq0'] = 0
                elif np.nanmax(stages_true) <= 3:
                    dfc.loc[index, 'geq1'] = 1
                    dfc.loc[index, 'leq2'] = 0
                    dfc.loc[index, 'geq2'] = 1
                    dfc.loc[index, 'leq1'] = 0
                    dfc.loc[index, 'geq3'] = 1
                    dfc.loc[index, 'leq0'] = 0
            else:
                for ep_st, col_ in zip([1, 2], ['geq2', 'geq3']):
                    if len([s for s in stages_false if s > float(ep_st)]) == len([s for s in endpoint_status_mapping if s > float(ep_st)]):
                        dfc.loc[index, col_] = 0
                for ep_st, col_ in zip([1, 2], ['leq1', 'leq2']):
                    if len([s for s in stages_false if s > float(ep_st)]) == len([s for s in endpoint_status_mapping if s > float(ep_st)]):
                        dfc.loc[index, col_] = 1

                # might be too strict due to 1.u often unknown
                if len(stages_false) == len(endpoint_status_mapping):
                    dfc.loc[index, 'endpoint_status'] = '0'
                    dfc.loc[index, 'geq1'] = 0
                    dfc.loc[index, 'leq0'] = 1

    return dfc


def urine_grid_df(pids):
    dfu_p = []
    dfu_t = []
    for pid in pids:
        new_pid = map_pid(pid)
        time = np.nan
        prev_time = np.nan
        # break
        if new_pid in list(KDIGO_dict.keys()):

            KDIGO_list = KDIGO_dict[new_pid]
            for ep in KDIGO_list:
                ep = list(ep)
                if len(ep) > 2:
                    # print(new_pid, ep)
                    if len(str(ep[2])) > 0:
                        if str(ep[2][-1]) == 'u':
                            time = ep[1]
                            if pd.isnull(prev_time):
                                prev_time = ep[1]
                            if time - prev_time > ffill_horizon:
                                prev_time = time
                            while prev_time < time:
                                dfu_p.append(float(pid))
                                dfu_t.append(prev_time)
                                prev_time += pd.Timedelta('1 h')
    df_u = pd.DataFrame()
    df_u['PatientID'] = dfu_p
    df_u[time_hirid_key] = dfu_t
    df_u[time_hirid_key] = pd.to_datetime(df_u[time_hirid_key])
    return df_u


def handle_file(file, to_disk=True):
    print(file)
    ykey = '/endpoints'
    df = pd.concat([x[['PatientID', time_hirid_key]] for x in pd.read_hdf(file, ykey, mode='r', chunksize=1000)], ignore_index=True)
    dfu = urine_grid_df(list(df['PatientID'].unique()))
    print(len(df['PatientID'].unique()), len(df), len(dfu))
    df = pd.concat([df, dfu]).drop_duplicates().reset_index(drop=True)
    df = df.sort_values(by=['PatientID', time_hirid_key]).reset_index(drop=True)
    # drop rows that are less than 5 min from prev row
    print(len(df))
    iterator = 1
    while iterator < len(df):
        if df.loc[iterator - 1, 'PatientID'] == df.loc[iterator, 'PatientID']:
            if df.loc[iterator - 1, time_hirid_key] < df.loc[iterator, time_hirid_key] - pd.Timedelta('1 h'):
                df.drop(iterator, inplace=True)
                df = df.reset_index(drop=True)
                continue
        iterator += 1

    print(len(df))

    # print(list(df['PatientID'].unique()))
    df['endpoint_status'] = 'unknown'
    df['leq0'] = np.nan
    df['leq1'] = np.nan  # smaller equal 1
    df['leq2'] = np.nan  # smaller equal 1
    df['geq1'] = np.nan  # larger equal 2
    df['geq2'] = np.nan  # larger equal 2
    df['geq3'] = np.nan  # larger equal 2
    for ep_t in ep_types:
        df[ep_t] = np.nan
    # df = fill_df(df)

    # create as many processes as there are CPUs available
    num_processes = multiprocessing.cpu_count()
    # calculate the chunk size as an integer
    chunk_size = int(df.shape[0] / num_processes)
    # will work even if the length of the dataframe is not evenly divisible by num_processes

    chunks = [df.ix[df.index[i:i + chunk_size]] for i in range(0, df.shape[0], chunk_size)]
    # create our pool with `num_processes` processes
    pool = multiprocessing.Pool(processes=num_processes)
    # apply our function to each chunk in the list
    result = pool.map(fill_df, chunks)

    for i in range(len(result)):
        # since result[i] is just a dataframe
        # we can reassign the original dataframe based on the index of each chunk
        df.ix[result[i].index] = result[i]

    df.set_index('PatientID')
    df.loc[:, time_endpoint_key] = pd.to_datetime(df[time_hirid_key].values)
    # print(df.columns.values.tolist())
    # print(df)
    outfile = base_ep_mimic + file.split('/')[-1]
    print(outfile)
    # print(list(df['PatientID'].values))
    if to_disk:
        df.to_hdf(outfile, 'endpoints', append=False, complevel=5, complib='blosc:lz4', data_columns=['PatientID'], format='table')
        reread = pd.read_hdf(outfile, key='/endpoints')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(reread[['PatientID', 'AbsDatetime', 'endpoint_status', 'geq1'] + list(ep_types)])


import multiprocessing
import sys
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser("Event based evaluation")
    parser.add_argument('--number', type=int, default=-1)
    parser.add_argument('--output_to_disk', type=bool, default=True)
    args = parser.parse_args()

    if args.number >= 0:
        chunkfiles = glob(endpoints_dir_MIMIC + 'reduced_endpoints_' + str(args.number) + '_*h5')
        print(chunkfiles)
        if len(chunkfiles) > 0:
            handle_file(chunkfiles[0], args.output_to_disk)
    else:
        ep_per_pid(args.output_to_disk)
