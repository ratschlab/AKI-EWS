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


def process_p(dfp, time_key, transition):
  dfp = dfp.sort_values(by=[time_key])
  dfp[transition] = dfp[transition] * 1
  idp = -1
  number_ep = len(dfp[dfp[transition] == 1])
  number_st = len(dfp[dfp[transition] == 0])
  if number_st == 0:
    idp = 0
  elif number_ep == 0:
    idp = 1
  else:
    idp = 2
  numbers = [len(dfp) - (number_ep + number_st), number_ep, number_st]

  ttrp = []
  if idp == 2:
    dfp[transition] = dfp[transition].fillna(-2)
    dfp['diff'] = dfp[transition].diff()
    dfp_filtered = dfp[dfp['diff'] != 0]
    dfp_filtered['Delta'] = dfp_filtered[time_key].diff()

    if not 'eICU' in base_ep:
      ttrp = [i / pd.Timedelta('1 h') for i in pd.to_timedelta(dfp_filtered[dfp_filtered['diff'] == 1]['Delta'].values)]
    else:
      ttrp = [i / 60 for i in dfp_filtered[dfp_filtered['diff'] == 1]['Delta'].values]

    if len(ttrp) > 0:
      idp = 3

  return idp, numbers


def process_variation(halflife_urine, baseline_creatinine, gap, transition, patient_group_dict, numbers_group_dict, pid_lists_dict):
  path_to_ep = base_ep  # + '/halflife' + str(halflife_urine) + 'h_baseline' + str(baseline_creatinine) + '_gap' + str(gap) + 'h/'

  time_key = 'AbsDatetime'
  patient_group_list = [0, 0, 0, 0]  # Patients without endpoints,Patients with only zero state,Patients with transition but no stability before transition, Patients with valid endpoint
  numbers_group_list = [0, 0, 0]  # Number of timepoints unknown/in endpoints/stable
  pid_valid = []
  pid_ep = []
  chunkfiles_ep = sorted(glob(path_to_ep + 'batch*h5'))
  print(chunkfiles_ep)
  for outfile_ep in chunkfiles_ep:
    print(outfile_ep)
    if '100.h5' not in outfile_ep:
      dfread = pd.concat([x[['PatientID', time_key, transition]] for x in pd.read_hdf(outfile_ep, '/endpoints', mode='r', chunksize=500)], ignore_index=True)

      print(dfread)

      grouped = dfread.groupby('PatientID')
      dflist = []
      pdlist = []
      for name, group in grouped:
        dflist.append(group)
        pdlist.append(name)

      for pid, dfp in zip(pdlist, dflist):
        idp, numbers = process_p(dfp, time_key, transition)
        if idp > 0:
          pid_valid.append(pid)
          if idp > 1:
            pid_ep.append(pid)
        numbers_group_list[0] += numbers[0]
        numbers_group_list[1] += numbers[1]
        numbers_group_list[2] += numbers[2]
        patient_group_list[idp] += 1
    # break

  print(path_to_ep, transition)
  print(patient_group_list)
  print(numbers_group_list)
  patient_group_dict[str(halflife_urine) + str(baseline_creatinine) + str(gap) + str(transition)] = patient_group_list
  numbers_group_dict[str(halflife_urine) + str(baseline_creatinine) + str(gap) + str(transition)] = numbers_group_list
  pid_lists_dict[str(halflife_urine) + str(baseline_creatinine) + str(gap) + str(transition)] = [pid_valid, pid_ep]


import argparse

if __name__ == '__main__':
  from paths import *

  manager = multiprocessing.Manager()
  patient_group_dict = {}  # manager.dict()
  numbers_group_dict = {}  # manager.dict()
  pid_lists_dict = {}  # manager.dict()
  pool = multiprocessing.Pool(processes=20)

  for baseline_creatinine in ['min_i']:  # 'literature_i', 'literature', 'min', 'min_i']:
    for halflife_urine in [0]:  # [0, 2]:
      for gap in [12]:  # [0, 12]:
        for transition in ['geq1', 'geq2']:
          process_variation(halflife_urine, baseline_creatinine, gap, transition, patient_group_dict, numbers_group_dict, pid_lists_dict)
          # pool.starmap(process_variation, [(halflife_urine, baseline_creatinine, gap, transition, patient_group_dict, numbers_group_dict, pid_lists_dict)])

  # pool.close()
  # pool.join()

  patient_group_dict = dict(patient_group_dict)
  numbers_group_dict = dict(numbers_group_dict)
  pid_lists_dict = dict(pid_lists_dict)

  pids_ep = []
  pids_valid = []

  for k in pid_lists_dict.keys():
    valid_setting, ep_setting = pid_lists_dict[k]
    if len(pids_ep) == 0:
      pids_valid = valid_setting
      pids_ep = ep_setting
    else:
      pids_ep = list(set(pids_ep) & set(ep_setting))
      pids_valid = list(set(pids_valid) & set(valid_setting))

    print(k)
    print(len(pids_ep), pids_ep)
    print(len(pids_valid), pids_valid)

  with open(base_ep + '/valid_pid.pkl', 'wb') as handle:
    pickle.dump(pids_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open(base_ep + '/ep_pid.pkl', 'wb') as handle:
    pickle.dump(pids_ep, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # print(patient_group_dict)
  # print(numbers_group_dict)

  # with open('./figures/summary_patient.pkl', 'wb') as handle:
  #     pickle.dump(patient_group_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  # with open('./figures/summary_patient.pkl', 'rb') as handle:
  #     patient_group_dict = pickle.load(handle)
  # with open('./figures/summary_timepoints.pkl', 'wb') as handle:
  #     pickle.dump(numbers_group_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  # with open('./figures/summary_timepoints.pkl', 'rb') as handle:
  #     numbers_group_dict = pickle.load(handle)

  # for gap in [0, 12]:
  #     for baseline_creatinine in ['literature', 'min', 'literature_i', 'min_i']:
  #         for halflife_urine in [0, 2]:
  #             pg_geq1 = patient_group_dict[str(halflife_urine) + str(baseline_creatinine) + str(gap) + 'geq1']
  #             pg_geq2 = patient_group_dict[str(halflife_urine) + str(baseline_creatinine) + str(gap) + 'geq2']
  #             ng_geq1 = numbers_group_dict[str(halflife_urine) + str(baseline_creatinine) + str(gap) + 'geq1']
  #             ng_geq2 = numbers_group_dict[str(halflife_urine) + str(baseline_creatinine) + str(gap) + 'geq2']

  #             pg_geq1 = [i / sum(pg_geq1) for i in pg_geq1]
  #             pg_geq2 = [i / sum(pg_geq2) for i in pg_geq2]
  #             ng_geq1 = [i / sum(ng_geq1) for i in ng_geq1]
  #             ng_geq2 = [i / sum(ng_geq2) for i in ng_geq2]
  #             if '_i' in baseline_creatinine:
  #                 print(str(halflife_urine) + 'h & '
  #                       'V' + ' & ' + baseline_creatinine[:-2] + ' & '
  #                       + str(gap) + 'h & '
  #                       + str(np.round(pg_geq1[0], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq1[1], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq1[2], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq1[3], decimals=2)) + ' & '
  #                       + str(np.round(ng_geq1[0], decimals=2)) + ' & '
  #                       + str(np.round(ng_geq1[1], decimals=2)) + ' & '
  #                       + str(np.round(ng_geq1[2], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq2[0], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq2[1], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq2[2], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq2[3], decimals=2)) + ' & '
  #                       + str(np.round(ng_geq2[0], decimals=2)) + ' & '
  #                       + str(np.round(ng_geq2[1], decimals=2)) + ' & '
  #                       + str(np.round(ng_geq2[2], decimals=2)) + ' \\\\')
  #             else:
  #                 print(str(halflife_urine) + 'h & '
  #                       'X' + ' & ' + baseline_creatinine + ' & '
  #                       + str(gap) + 'h & '
  #                       + str(np.round(pg_geq1[0], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq1[1], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq1[2], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq1[3], decimals=2)) + ' & '
  #                       + str(np.round(ng_geq1[0], decimals=2)) + ' & '
  #                       + str(np.round(ng_geq1[1], decimals=2)) + ' & '
  #                       + str(np.round(ng_geq1[2], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq2[0], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq2[1], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq2[2], decimals=2)) + ' & '
  #                       + str(np.round(pg_geq2[3], decimals=2)) + ' & '
  #                       + str(np.round(ng_geq2[0], decimals=2)) + ' & '
  #                       + str(np.round(ng_geq2[1], decimals=2)) + ' & '
  #                       + str(np.round(ng_geq2[2], decimals=2)) + ' \\\\')
