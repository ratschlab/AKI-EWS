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


def process_p(dfp):
    dfp = dfp.sort_values(by=[time_key])
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
    else:
        los = (dfp[time_key].values[-1] - dfp[time_key].values[0]) / (60 * 24)

    # print(dfp[time_key].values[-1])
    # print(dfp[time_key].values[0])
    # print(dfp.columns.values.tolist())
    ttrp = []
    if idp == 2:
        dfp[transition] = dfp[transition].fillna(-2)
        print(dfp[transition])
        dfp['diff'] = dfp[transition].diff()
        dfp_filtered = dfp[dfp['diff'] != 0]
        dfp_filtered['Delta'] = dfp_filtered[time_key].diff()

        if not 'eICU' in base_ep:
            ttrp = [i / pd.Timedelta('1 h') for i in pd.to_timedelta(dfp_filtered[dfp_filtered['diff'] == 1]['Delta'].values)]
        else:
            ttrp = [i / 60 for i in dfp_filtered[dfp_filtered['diff'] == 1]['Delta'].values]

        if len(ttrp) > 0:
            idp = 3

    return idp, ttrp, los, which_0


import argparse
from tqdm import tqdm

if __name__ == '__main__':
    # base_ep = '../../endpoints_hirid/'
    # base_ep = '../../MIMIC-III/endpoints/'
    # base_ep = '../../eICU/endpoints/'

    path_to_ep = base_ep +  '/220821/'
    transition = 'geq1'

    time_key = 'charttime'
    if 'eICU' in base_ep:
        time_key = 'offset'

    patient_group = [0, 0, 0, 0]  # Patients without endpoints,Patients with only zero state,Patients with transition but no stability before transition, Patients with valid endpoint
    los_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]  # save length of stays
    number_transitions = []
    ttr = []  # time to each transition

    zeros = [0, 0, 0, 0, 0]

    pid_los_large = []
    pid_ur_sampling_regular = []
    pid_ur_sampling_max = []
    pid_with_ep = []
    valid_pid = []

    median_ur = []
    max_ur = []
    print(path_to_ep)
    chunkfiles_ep = sorted(glob(path_to_ep + 'batch*h5'))
    los_creatinine = []
    # print(chunkfiles_ep)
    if len(chunkfiles_ep)<50:
        print('not all batchfiles')
        exit()
    for i in range(len(chunkfiles_ep)):
        # if i > 1:
        #     break
        outfile_ep = chunkfiles_ep[i]
        dfread = pd.concat([x[[patient_key, time_key, transition, '1.u', '3.u', '1.b']] for x in pd.read_hdf(outfile_ep, '/endpoints', mode='r', chunksize=500)], ignore_index=True)
        print(dfread)

        grouped = dfread.groupby(patient_key)
        dflist = []
        pdlist = []
        for name, group in grouped:
            dflist.append(group)
            pdlist.append(name)

        for pid, dfp in zip(pdlist, dflist):
            idp, ttrp, los, which_0 = process_p(dfp)

            if los > 1:
                pid_los_large.append(pid)
            if idp == 3:
                pid_with_ep.append(pid)

            if idp != 0:
                valid_pid.append(pid)

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

            ttr = ttr + ttrp
        # break

    np.savetxt(path_to_ep + 'pid_los_day.txt', pid_los_large)
    print(len(pid_los_large))
    np.savetxt(path_to_ep + 'pid_with_tranistion.txt', pid_with_ep)
    print(len(pid_with_ep))
    np.savetxt(path_to_ep + 'valid_pid.txt', valid_pid)
    print(len(valid_pid))

    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    f.suptitle(transition + ' transitions')

    # print(patient_group)
    # print(zeros)
    ax1.bar([1, 2, 3], patient_group[1:], alpha=0.4)
    ax1.bar([0], zeros[0], color='darkred', alpha=0.4, label='unstable')
    ax1.bar([0], zeros[1], bottom=zeros[0], color='darkblue', alpha=0.4, label='urine')
    ax1.bar([0], zeros[3], bottom=zeros[0] + zeros[1], color='lightblue', alpha=0.4, label='short stay')
    ax1.bar([0], zeros[2], bottom=zeros[0] + zeros[1] + zeros[3], color='darkgreen', alpha=0.4, label='creatinine')
    ax1.bar([0], zeros[4], bottom=zeros[0] + zeros[1] + zeros[3] + zeros[2], color='magenta', alpha=0.4, label='other')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xlabel('patient type')
    ax1.set_ylabel('# stays')
    ax1.legend(loc='best')
    # ax1.set_yscale('log')
    # ax1.set_yticks([100, 1000, 10000])
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(['unknown', 'stable', 'invalid', 'endpoint'])

    if len(ttr) > 1:

        xmax = 24 * 5
        ax4.hist(ttr, bins=[i * 8 for i in range(5 * 3)], color='k', alpha=0.4, histtype='bar')
        ax4.axvline(np.nanmean(ttr), color='k', linestyle='-', label='mean = ' + str(np.nanmean(ttr)) + 'h', alpha=0.4)
        ax4.axvline(np.nanmedian(ttr), color='k', linestyle=':', label='median = ' + str(np.nanmedian(ttr)) + 'h', alpha=0.4)
        ax4.axvline(np.nanpercentile(ttr, 95), color='k', linestyle='--', label='95th percentile = ' + str(np.nanpercentile(ttr, 95)) + 'h', alpha=0.4)
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.set_ylabel('# events')
        ax4.set_xlim((0, xmax))
        ax4.set_yscale('log')
        ax4.legend(loc='best')
        ax4.set_xticks([0, 8, 24, 48, 72, 96, 120])
        ax4.set_xlabel('hours of stability before transition')

    for k, name_list, list_t, col in zip(range(50), ['unknown', 'stable', 'unstable invalid', 'single episode', 'more than one episode'], los_list, ['darkred', 'red', 'orange', 'darkblue', 'blue', 'yellow']):
        if len(list_t) > 0:
            ax3.hist(list_t, bins=[i + k / 5 - 0.35 for i in range(21)], width=0.18, color=col, label=name_list, alpha=0.8, histtype='bar')
    ax3.set_yscale('log')
    ax3.legend(loc='best')
    ax3.set_xlabel('LOS in days')
    ax3.set_ylabel('# stays')
    ax3.set_xticks(range(21))
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)

    ax2.hist(number_transitions, bins=range(10), alpha=0.4)
    ax2.axvline(np.mean(number_transitions), linestyle='-', label='mean = ' + str(np.round(np.mean(number_transitions), decimals=2)), alpha=0.4)
    ax2.axvline(np.median(number_transitions), linestyle=':', label='median = ' + str(np.round(np.median(number_transitions), decimals=2)), alpha=0.4)
    ax2.axvline(np.percentile(number_transitions, 95), linestyle='--', label='95th percentile = ' + str(np.round(np.percentile(number_transitions, 95), decimals=2)), alpha=0.4)
    ax2.legend(loc='best')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_yscale('log')
    ax2.set_xticks(range(11))
    ax2.set_xlabel('# AKI episodes')
    ax2.set_ylabel('# stays')

    plt.tight_layout()
    plt.savefig('./figures/stats.pdf')

    plt.clf()
    plt.close()
