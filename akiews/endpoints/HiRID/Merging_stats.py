import pandas as pd
import numpy as np
import datetime as dt
from glob import glob
import os.path
import os
from os.path import join
import csv
# from tqdm import tqdm

import pickle
import multiprocessing
# import time

from paths import *


def process_p(dfp, ur_time):
    dfp = dfp.sort_values(by=[time_key])
    dfp[transition] = dfp[transition] * 1
    idp = -1
    which_0 = np.nan
    if len(dfp[dfp[transition] == 0]) == 0:
        idp = 0
        if len(dfp[dfp[transition] == 1]) > 0:
            which_0 = 0
        elif len(dfp[dfp['1.u'] == -2]) > 0:
            which_0 = 1
        elif len(dfp[dfp['1.b'] == -2]) > 0:
            which_0 = 2
        else:
            if len(dfp[dfp['1.u'] == -1]) + len(dfp[pd.isnull(dfp['1.u'])]) == len(dfp) and len(dfp[dfp['1.b'] == -1]) + len(dfp[pd.isnull(dfp['1.b'])]) == len(dfp):
                which_0 = 2
            elif len(dfp[dfp['1.u'] == -1]) + len(dfp[pd.isnull(dfp['1.u'])]) == len(dfp):
                which_0 = 1
            elif len(dfp[dfp['1.b'] == -1]) + len(dfp[pd.isnull(dfp['1.b'])]) == len(dfp):
                which_0 = 2
            elif len(dfp[dfp['3.u'] == -1]) + len(dfp[pd.isnull(dfp['3.u'])]) == len(dfp):
                which_0 = 3
            else:
                print('Unknown reason for no transition')
                print(dfp)
                which_0 = 4
    elif len(dfp[dfp[transition] == 1]) == 0:
        idp = 1
    else:
        idp = 2

    if not 'eICU' in base_ep:
        los = pd.to_timedelta(dfp[time_key].values[-1] - dfp[time_key].values[0]) / pd.Timedelta('1 day')
        # if ur_time > pd.Timedelta('1 h'):
        #     idp = 0
        #     which_0 = 1
    else:
        los = (dfp[time_key].values[-1] - dfp[time_key].values[0]) / (60 * 24)
        # if ur_time > 60:
        #     idp = 0
        #     which_0 = 1
    ttrp = []
    ttrg = []
    ttre = []
    if idp == 2:
        dfp[transition] = dfp[transition].fillna(-2)
        dfp['diff'] = dfp[transition].diff()
        dfp_filtered = dfp[dfp['diff'] != 0]
        dfp_filtered['Delta'] = dfp_filtered[time_key].diff()
        print(dfp_filtered)

        if not 'eICU' in base_ep:
            ttrp = [i / pd.Timedelta('1 h') for i in pd.to_timedelta(dfp_filtered[dfp_filtered['diff'] == 1]['Delta'].values)]
            ttre = [i / pd.Timedelta('1 h') for i in pd.to_timedelta(dfp_filtered[dfp_filtered['diff'] == -1]['Delta'].values)]
        else:
            ttrp = [i / 60 for i in dfp_filtered[dfp_filtered['diff'] == 1]['Delta'].values]
            ttre = [i / 60 for i in dfp_filtered[dfp_filtered['diff'] == -1]['Delta'].values]

        if len(ttrp) > 1:
            ttrg = ttrp[1:]

        if len(ttrp) > 0:
            idp = 3

    return idp, ttrp, los, which_0, ttrg, ttre


import argparse

if __name__ == '__main__':

    # base_ep = '../../endpoints_hirid/'
    # base_ep = '../../MIMIC-III/endpoints/'
    # base_ep = '../../eICU/endpoints/'
    x_axis = [0, 1, 2, 4, 8, 12, 16, 24, 36, 48]
    aki_epidsodes_mean = []
    aki_epidsodes_median = []
    aki_epidsodes_nf = []
    gap_mean = []
    gap_median = []
    gap_nf = []
    event_mean = []
    event_median = []
    event_nf = []
    for gap_length in x_axis:
        path_to_ep = base_ep + 'merging' + str(gap_length) + 'h/'

        time_key = 'AbsDatetime'
        if 'eICU' in base_ep:
            time_key = 'offset'

        patient_group = [0, 0, 0, 0]  # Patients without endpoints,Patients with only zero state,Patients with transition but no stability before transition, Patients with valid endpoint
        los_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]  # save length of stays
        number_transitions = []
        ttr = []  # time to each transition
        tte = []
        transition = 'geq1'

        zeros = [0, 0, 0, 0, 0]

        pid_los_large = []
        pid_ur_sampling_regular = []
        pid_ur_sampling_max = []
        pid_with_ep = []

        median_ur = []
        max_ur = []
        print(path_to_ep)
        chunkfiles_ep = sorted(glob(path_to_ep + 'batch*h5'))
        chunkfiles_three = sorted(glob(base_h5_hirid + 'reduced_*h5'))
        los_creatinine = []
        print(chunkfiles_ep)
        print(chunkfiles_three)
        for outfile_ep, outfile_three in zip(chunkfiles_ep, chunkfiles_three):
            print(outfile_ep, outfile_three)
            if '100.h5' not in outfile_ep:
                dfread = pd.concat([x[['PatientID', time_key, transition, '1.u', '1.b', '1.i', '2.u', '2.b', '3.u', '3.b', '3.i', '3.d']] for x in pd.read_hdf(outfile_ep, '/endpoints', mode='r', chunksize=500)], ignore_index=True)
                dfread_three = pd.concat([x[['PatientID', 'Datetime', urine_hirid_key]].dropna(how='any') for x in pd.read_hdf(outfile_three, '/reduced', mode='r', chunksize=500)], ignore_index=True)
                dfread_three['Datetime'] = pd.to_datetime(dfread_three['Datetime'])

                grouped = dfread.groupby('PatientID')
                dflist = []
                pdlist = []
                for name, group in grouped:
                    dflist.append(group)
                    pdlist.append(name)
                # print(outfile_ep, outfile_three, len(pdlist))

                for pid, dfp in zip(pdlist, dflist):
                    pd_ur = dfread_three[dfread_three['PatientID'] == pid]
                    pd_ur = pd_ur.sort_values(by=['Datetime'])

                    ur_time = pd_ur['Datetime'].diff().median(skipna=True)
                    ur_time_max = pd_ur['Datetime'].diff().max(skipna=True)
                    idp, ttrp, los, which_0, ttrg, ttre = process_p(dfp, ur_time)

                    # print(pid, ur_time, ur_time_max, idp, los)
                    if not pd.isnull(ur_time):
                        median_ur.append(ur_time / pd.Timedelta('1 h'))
                        max_ur.append(ur_time_max / pd.Timedelta('1 h'))

                    if los > 1:
                        pid_los_large.append(pid)
                    if ur_time <= pd.Timedelta('2 h'):
                        pid_ur_sampling_regular.append(pid)
                    if ur_time_max < pd.Timedelta('2 h'):
                        pid_ur_sampling_max.append(pid)
                    if idp == 3:
                        pid_with_ep.append(pid)

                    patient_group[idp] += 1
                    if idp < 3:
                        los_list[idp].append(los)
                        if idp != 0:
                            number_transitions.append(0)
                        else:
                            zeros[which_0] += 1
                            if which_0 == 2:
                                los_creatinine.append(los)
                    else:
                        if len(ttrp) == 1:
                            los_list[3].append(los)
                        else:
                            los_list[4].append(los)
                        number_transitions.append(len(ttrp))

                    ttr = ttr + ttrg  # ttrp
                    tte = tte + ttre
                # break

        aki_epidsodes_mean.append(np.nanmean(number_transitions))
        aki_epidsodes_median.append(np.nanmedian(number_transitions))
        aki_epidsodes_nf.append(np.nanpercentile(number_transitions, 95))
        gap_mean.append(np.nanmean(ttr))
        gap_median.append(np.nanmedian(ttr))
        gap_nf.append(np.nanpercentile(ttr, 95))
        event_mean.append(np.nanmean(tte))
        event_median.append(np.nanmedian(tte))
        event_nf.append(np.nanpercentile(tte, 95))

    from matplotlib import pyplot as plt
    f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(4, 7), sharex=True)
    ax1.plot(x_axis, aki_epidsodes_mean, color='k', linestyle='-', label='mean')
    ax1.plot(x_axis, aki_epidsodes_median, color='k', linestyle='--', label='median')
    ax1.plot(x_axis, aki_epidsodes_nf, color='k', linestyle='-.', label='95th percentile')
    ax1.legend(loc='best')
    ax1.set_ylabel('KDIGO episodes [#]')
    ax2.plot(x_axis, gap_mean, color='k', linestyle='-', label='mean')
    ax2.plot(x_axis, gap_median, color='k', linestyle='--', label='median')
    ax2.plot(x_axis, gap_nf, color='k', linestyle='-.', label='95th percentile')
    ax2.set_ylabel('gap length [h]')
    ax3.plot(x_axis, event_mean, color='k', linestyle='-', label='mean')
    ax3.plot(x_axis, event_median, color='k', linestyle='--', label='median')
    ax3.plot(x_axis, event_nf, color='k', linestyle='-.', label='95th percentile')
    ax3.set_ylabel('event length [h]')
    ax3.set_xlabel('merging length [h]')
    ax3.set_xticks(x_axis)

    plt.tight_layout()
    plt.savefig('Merging_stats.pdf')
