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


def process_p(dfp, ur_time,masked_urine_treatment=False):
    dfp = dfp.sort_values(by=[time_key])
    if masked_urine_treatment:
        dfp[transition] = np.logical_or(dfp[transition],dfp['masked_urine'])
    else:
        dfp[transition] = np.logical_and(dfp[transition],np.logical_not(dfp['masked_urine']))
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

    los = pd.to_timedelta(dfp[time_key].values[-1] - dfp[time_key].values[0]) / pd.Timedelta('1 day')

    ttrp = []
    ttrg = []
    length_events = []
    if idp > 1:
        dfp[transition] = dfp[transition].fillna(-2)
        dfp['diff'] = dfp[transition].diff()
        dfp_filtered = dfp[dfp['diff'] != 0]
        dfp_filtered['Delta'] = dfp_filtered[time_key].diff()

        ttrp = [i / pd.Timedelta('1 h') for i in pd.to_timedelta(dfp_filtered[dfp_filtered['diff'] == 1]['Delta'].values)]
        length_events = [i / pd.Timedelta('1 h') for i in pd.to_timedelta(dfp_filtered[dfp_filtered['diff'] == -1]['Delta'].values)]

        if len(ttrp) > 1:
            ttrg = ttrp[1:]

        if len(ttrp) > 0:
            idp = 3

    return idp, ttrp, los, which_0, ttrg, length_events


import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Event based evaluation")
    parser.add_argument('--halflife_urine', type=int, default=8)  # help='ewma halflife in hours',
    parser.add_argument('--gap', type=int, default=6)  # help=merging of events in hours
    parser.add_argument('--baseline_creatinine', type=str, default='literature')
    parser.add_argument('--transition', type=str, default='geq1')
    parser.add_argument('--masked_urine_treatment', type=int, default=0)
    parser.add_argument('--name_plot', type=str, default='HiRID')
    args = parser.parse_args()

    # base_ep = '../../endpoints_hirid/'
    # base_ep = '../../MIMIC-III/endpoints/'
    # base_ep = '../../eICU/endpoints/'

    path_to_ep = base_ep  # + 'merging' + str(args.gap) + 'h/'
    transition = args.transition
    name_plot = args.name_plot
    masked_urine_treatment = args.masked_urine_treatment != 0
    print(masked_urine_treatment)

    time_key = 'AbsDatetime'
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

    levents = []

    median_ur = []
    max_ur = []
    # print(path_to_ep)
    chunkfiles_ep = sorted(glob(path_to_ep + 'batch*h5'))
    chunkfiles_three = sorted(glob(base_h5_hirid + 'reduced_*h5'))
    los_creatinine = []
    # print(chunkfiles_ep)
    # print(chunkfiles_three)
    counter = 0
    for outfile_ep, outfile_three in zip(chunkfiles_ep, chunkfiles_three):
        counter += 1
        # if counter > 3:
        #     break
        if '100.h5' not in outfile_ep:
            try:
                dfread = pd.concat([x[['PatientID', time_key, transition,'masked_urine', '1.u', '1.b', '1.i', '2.u', '2.b', '3.u', '3.b', '3.i', '3.d']] for x in pd.read_hdf(outfile_ep, '/endpoints', mode='r', chunksize=500)], ignore_index=True)
            except:
                dfread = pd.concat([x[['PatientID', time_key, transition, 'masked_urine','1.u', '1.b', '1.i', '2.u', '2.b', '3.u', '3.b', '3.i', '3.d']] for x in pd.read_hdf(outfile_ep, mode='r', chunksize=500)], ignore_index=True)
            dfread_three = pd.concat([x[['PatientID', 'Datetime', urine_hirid_key]].dropna(how='any') for x in pd.read_hdf(outfile_three, '/reduced', mode='r', chunksize=500)], ignore_index=True)
            # dfread = pd.read_hdf(outfile_ep, mode='r')[['PatientID', time_key, transition, '1.u', '1.b', '1.i', '2.u', '2.b', '3.u', '3.b', '3.i', '3.d']]
            # dfread_three = pd.read_hdf(outfile_three, '/reduced', mode='r',columns=['PatientID', 'Datetime'])
            dfread[time_key] = pd.to_datetime(dfread[time_key])
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
                print(masked_urine_treatment)
                idp, ttrp, los, which_0, ttrg, length_events = process_p(dfp, ur_time,masked_urine_treatment)

                levents.extend(length_events)

                # print(pid, ur_time, ur_time_max, idp, los)
                if not pd.isnull(ur_time):
                    median_ur.append(ur_time / pd.Timedelta('1 h'))
                    max_ur.append(ur_time_max / pd.Timedelta('1 h'))

                if idp!=0:
                    valid_pid.append(pid)

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
            # break

    np.savetxt(path_to_ep + 'pid_los_day.txt', pid_los_large)
    print(len(pid_los_large))
    np.savetxt(path_to_ep + 'pid_ur_sampling_regular.txt', pid_ur_sampling_regular)
    print(len(pid_ur_sampling_regular))
    np.savetxt(path_to_ep + 'pid_ur_sampling_max.txt', pid_ur_sampling_max)
    print(len(pid_ur_sampling_max))
    np.savetxt(path_to_ep + 'pid_with_tranistion.txt', pid_with_ep)
    print(len(pid_with_ep))
    np.savetxt(path_to_ep + 'valid_pid.txt', valid_pid)
    print(len(valid_pid))


    from matplotlib import pyplot as plt
    plt.hist(median_ur, bins=range(17), color='red', alpha=0.4, label='median urine sampling')
    plt.hist(max_ur, bins=range(17), color='blue', alpha=0.4, label='max urine interval')
    plt.xlabel('time [h]')
    plt.ylabel('frequency')
    plt.legend(loc='best')
    plt.semilogy()
    plt.savefig('./figures/urine_sampling.pdf')
    plt.close()

    f, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(3, 2, figsize=(10, 12))
    f.suptitle(transition + ' transitions')

    # print(patient_group)
    # print(zeros)
    ax1.hist(levents, bins=[i * 24 for i in range(10)], label='#episodes = ' + str(len(levents)), alpha=0.4)
    ax1.axvline(np.mean(levents), linestyle='-', label='mean = ' + str(np.round(np.mean(levents), decimals=2)), alpha=0.4)
    ax1.axvline(np.median(levents), linestyle=':', label='median = ' + str(np.round(np.median(levents), decimals=2)), alpha=0.4)
    ax1.axvline(np.percentile(levents, 95), linestyle='--', label='95th percentile = ' + str(np.round(np.percentile(levents, 95), decimals=2)), alpha=0.4)
    ax1.legend(loc='best')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_yscale('log')
    ax1.set_xticks([i * 24 for i in range(10)])
    ax1.set_xticklabels([str(i) for i in range(10)])
    ax1.set_ylabel('# AKI episodes')
    ax1.set_xlabel('duration [d]')

    ax4.bar([1, 2, 3], patient_group[1:], alpha=0.4)
    ax4.bar([0], zeros[0], color='darkred', alpha=0.4, label='unstable')
    ax4.bar([0], zeros[1], bottom=zeros[0], color='darkblue', alpha=0.4, label='no urine')
    ax4.bar([0], zeros[3], bottom=zeros[0] + zeros[1], color='lightblue', alpha=0.4, label='<6h urine')
    ax4.bar([0], zeros[2], bottom=zeros[0] + zeros[1] + zeros[3], color='darkgreen', alpha=0.4, label='creatinine')
    ax4.bar([0], zeros[4], bottom=zeros[0] + zeros[1] + zeros[2] + zeros[3], color='magenta', alpha=0.4, label='other')
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_xlabel('patient type')
    ax4.set_ylabel('# stays')
    ax4.legend(loc='best')
    # ax1.set_yscale('log')
    # ax1.set_yticks([100, 1000, 10000])
    ax4.set_xticks([0, 1, 2, 3])
    ax4.set_xticklabels(['unknown', 'stable', 'invalid', 'endpoint'])

    ax6.spines['right'].set_visible(False)
    ax6.spines['top'].set_visible(False)
    # df=pd.DataFrame()
    # df['time to transition']=[np.mean(ttr), np.median(ttr), np.percentile(ttr, 95), np.percentile(ttr, 5)]
    # df['events']=[]

    if len(ttr) > 0:
        xmax = 24 * 5
        ax5.hist(ttr, bins=[i * 8 for i in range(5 * 3)], color='k', alpha=0.4, histtype='bar')
        ax5.axvline(np.mean(ttr), color='k', linestyle='-', label='mean = ' + str(int(np.mean(ttr))) + 'h', alpha=0.4)
        ax5.axvline(np.median(ttr), color='k', linestyle=':', label='median = ' + str(int(np.median(ttr))) + 'h', alpha=0.4)
        ax5.axvline(np.percentile(ttr, 95), color='k', linestyle='--', label='95th percentile = ' + str(int(np.percentile(ttr, 95))) + 'h', alpha=0.4)
        ax5.spines['right'].set_visible(False)
        ax5.spines['top'].set_visible(False)
        ax5.set_ylabel('# events')
        ax5.set_xlim((0, xmax))
        ax5.set_yscale('log')
        ax5.legend(loc='best')
        ax5.set_xticks([0, 8, 24, 48, 72, 96, 120])
        ax5.set_xlabel('hours of stability before transition')

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
    plt.savefig('./figures/Classification_stats' + '_' + name_plot + '_' + transition + '_gap' + str(args.gap) + 'h_' +str(masked_urine_treatment)+'.pdf')
    print('./figures/Classification_stats' + '_' + name_plot + '_' + transition + '_gap' + str(args.gap) + 'h_' +str(masked_urine_treatment)+'.pdf')

    plt.clf()
    plt.close()

    # print(los_creatinine)
    # plt.hist(los_creatinine, bins=[i / 2 for i in range(10)], color=col, alpha=0.8, histtype='bar')
    # plt.axvline(np.mean(los_creatinine), label='mean=' + str(np.round(np.mean(los_creatinine), decimals=2)) + 'days', color='k')
    # plt.axvline(np.median(los_creatinine), label='median=' + str(np.round(np.median(los_creatinine), decimals=2)) + 'days', linestyle='--', color='k')
    # plt.ylabel('# patients w/o creatinine')
    # plt.xlabel('los [days]')
    # plt.semilogy()
    # plt.legend(loc='best')

    # plt.savefig('./figures/No_creatinine_los' + '_' + name_plot + '_' + transition + '.pdf')
