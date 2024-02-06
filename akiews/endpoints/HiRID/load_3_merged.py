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

import pickle
import multiprocessing
from functools import partial
from itertools import repeat
# import time

from paths import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_batch(pid):
    pid = int(pid)
    if pid >= 12432 and pid <= 13471:
        return '0'
    elif pid >= 49941 and pid <= 50971:
        return '18'
    elif pid >= 59582 and pid <= 60650:
        return '27'
    elif pid >= 71555 and pid <= 72949:
        return '36'
    elif pid >= 84842 and pid <= 86375:
        return '45'
    elif 40629 <= pid <= 41880:
        return '10'
    elif 50972 <= pid <= 52028:
        return '19'
    elif 60651 <= pid <= 61795:
        return '28'
    elif 72951 <= pid <= 74342:
        return '37'
    elif 86376 <= pid <= 88151:
        return '46'
    elif 13472 <= pid <= 14532:
        return '1'
    elif 52032 <= pid <= 53129:
        return '20'
    elif 61796 <= pid <= 63154:
        return '29'
    elif 74343 <= pid <= 75767:
        return '38'
    elif 88152 <= pid <= 90113:
        return '47'
    elif 41881 <= pid <= 43146:
        return '11'
    elif 14533 <= pid <= 15566:
        return '2'
    elif 63155 <= pid <= 64409:
        return '30'
    elif 75769 <= pid <= 77217:
        return '39'
    elif 90114 <= pid <= 91709:
        return '48'
    elif 43157 <= pid <= 44338:
        return '12'
    elif 53131 <= pid <= 54202:
        return '21'
    elif 15567 <= pid <= 16613:
        return '3'
    elif 77221 <= pid <= 78816:
        return '40'
    elif 91710 <= pid <= 93101:
        return '49'
    elif 44340 <= pid <= 45500:
        return '13'
    elif 54203 <= pid <= 55344:
        return '22'
    elif 64410 <= pid <= 65686:
        return '31'
    elif 16614 <= pid <= 17639:
        return '4'
    elif 17640 <= pid <= 18682:
        return '5'
    elif 45501 <= pid <= 46623:
        return '14'
    elif 55345 <= pid <= 56411:
        return '23'
    elif 65688 <= pid <= 66963:
        return '32'
    elif 78817 <= pid <= 80328:
        return '41'
    elif 18683 <= pid <= 19725:
        return '6'
    elif 46624 <= pid <= 47697:
        return '15'
    elif 56412 <= pid <= 57437:
        return '24'
    elif 66964 <= pid <= 68722:
        return '33'
    elif 80329 <= pid <= 81913:
        return '42'
    elif 19727 <= pid <= 30025:
        return '7'
    elif 47698 <= pid <= 48866:
        return '16'
    elif 57438 <= pid <= 58501:
        return '25'
    elif 68723 <= pid <= 70167:
        return '34'
    elif 81914 <= pid <= 83390:
        return '43'
    elif 30026 <= pid <= 31195:
        return '8'
    elif 48867 <= pid <= 49940:
        return '17'
    elif 58502 <= pid <= 59581:
        return '26'
    elif 70169 <= pid <= 71552:
        return '35'
    elif 83391 <= pid <= 84841:
        return '44'
    elif 31197 <= pid <= 40628:
        return '9'
    elif 239976 <= pid <= 241951:
        return '20'
    elif 205949 <= pid <= 207965:
        return '3'
    elif 286008 <= pid <= 288013:
        return '43'
    elif 200001 <= pid <= 201962:
        return '0'
    elif 241952 <= pid <= 243950:
        return '21'
    elif 264067 <= pid <= 266072:
        return '32'
    elif 288014 <= pid <= 290039:
        return '44'
    elif 219885 <= pid <= 221937:
        return '10'
    elif 203944 <= pid <= 205948:
        return '2'
    elif 266075 <= pid <= 268059:
        return '33'
    elif 290042 <= pid <= 291978:
        return '45'
    elif 221938 <= pid <= 223961:
        return '11'
    elif 243951 <= pid <= 246005:
        return '22'
    elif 268063 <= pid <= 269987:
        return '34'
    elif 291979 <= pid <= 293990:
        return '46'
    elif 201964 <= pid <= 203943:
        return '1'
    elif 246007 <= pid <= 248007:
        return '23'
    elif 269989 <= pid <= 271996:
        return '35'
    elif 293991 <= pid <= 295986:
        return '47'
    elif 223962 <= pid <= 225983:
        return '12'
    elif 248008 <= pid <= 249954:
        return '24'
    elif 271997 <= pid <= 273981:
        return '36'
    elif 295987 <= pid <= 297949:
        return '48'
    elif 225985 <= pid <= 227935:
        return '13'
    elif 249955 <= pid <= 251950:
        return '25'
    elif 273983 <= pid <= 275949:
        return '37'
    elif 297950 <= pid <= 299999:
        return '49'
    elif 227939 <= pid <= 229908:
        return '14'
    elif 251953 <= pid <= 253946:
        return '26'
    elif 275950 <= pid <= 277942:
        return '38'
    elif 209937 <= pid <= 211958:
        return '5'
    elif 229909 <= pid <= 231963:
        return '15'
    elif 253947 <= pid <= 255983:
        return '27'
    elif 277943 <= pid <= 279870:
        return '39'
    elif 211959 <= pid <= 213922:
        return '6'
    elif 231964 <= pid <= 233972:
        return '16'
    elif 255984 <= pid <= 257985:
        return '28'
    elif 279872 <= pid <= 281965:
        return '40'
    elif 213923 <= pid <= 215889:
        return '7'
    elif 233974 <= pid <= 235945:
        return '17'
    elif 257986 <= pid <= 260065:
        return '29'
    elif 281968 <= pid <= 283980:
        return '41'
    elif 215892 <= pid <= 217906:
        return '8'
    elif 235946 <= pid <= 237954:
        return '18'
    elif 260068 <= pid <= 262073:
        return '30'
    elif 207966 <= pid <= 209936:
        return '4'
    elif 217907 <= pid <= 219884:
        return '9'
    elif 237955 <= pid <= 239974:
        return '19'
    elif 262074 <= pid <= 264065:
        return '31'
    elif 283981 <= pid <= 286004:
        return '42'
    else:
        return -1


def extract_p_HiRID(pid):
    batchfile = glob(base_h5_hirid + '/reduced_fmat_' + str(int(get_batch(pid))) + '_*.h5')
    # batchfile = glob(base_h5_hirid + '/reduced_fmat_' + str(batch_number) + '*.h5')
    if len(batchfile) > 0:
        pdf = pd.concat([x[x.PatientID == pid][[creatinine_hirid_key, hr_hirid_key, 'PatientID', time_hirid_key, urine_hirid_key, weight_hirid_key] + [ep_drugs]] for x in pd.read_hdf(batchfile[0], '/reduced', mode='r', chunksize=500)], ignore_index=True)
        pdf = pdf.dropna(how='all', subset=[urine_hirid_key, creatinine_hirid_key, hr_hirid_key])
        pdf = pd.merge(pdf, static_hirid, how='left', on='PatientID')
        return pdf
    else:
        print('No batchfile for ', pid)
    return pd.DataFrame(columns=[time_hirid_key, hr_hirid_key, exit_hirid_key, weight_hirid_key, urine_hirid_key, dialysis_hirid_keys[0], dialysis_hirid_keys[1], creatinine_hirid_key, age_hirid_key, gender_hirid_key, admission_hirid_key, 'PatientID'] + [ep_drugs])


def extract_batch_HiRID(batch_number):
    print(glob(base_h5_hirid + '*.h5'))
    batchfile = glob(base_h5_hirid + '/reduced_fmat_' + str(batch_number) + '_*.h5')
    if len(batchfile) > 0:
        pdf = pd.concat([x[[creatinine_hirid_key, 'PatientID', time_hirid_key, urine_hirid_key, hr_hirid_key, ckd_hirid_key, weight_hirid_key] + dialysis_hirid_keys + [ep_drugs]] for x in pd.read_hdf(batchfile[0], '/reduced', mode='r', chunksize=500)], ignore_index=True)
        pdf = pd.merge(pdf, static_hirid, how='left', on='PatientID')
        return pdf
    else:
        print('No batchfile for ', batch_number)
    return pd.DataFrame(columns=[time_hirid_key, exit_hirid_key, weight_hirid_key, urine_hirid_key, hr_hirid_key, creatinine_hirid_key, age_hirid_key, ckd_hirid_key, gender_hirid_key, admission_hirid_key, 'PatientID'] + dialysis_hirid_keys + [ep_drugs])


def process_p(dfp, transition):
    dfp = dfp.sort_values(by=['AbsDatetime'])
    idp = -1
    if len(dfp[dfp[transition] == 0]) == 0:
        return 0
    elif len(dfp[dfp[transition] == 1]) == 0:
        return 1
    else:
        ttrp = []
        dfp[transition] = dfp[transition].fillna(-2)
        dfp['diff'] = dfp[transition].diff()
        dfp_filtered = dfp[dfp['diff'] != 0]
        dfp_filtered['Delta'] = dfp_filtered['AbsDatetime'].diff()
        ttrp = [i / 60 for i in dfp_filtered[dfp_filtered['diff'] == 1]['Delta'].values]
        if len(ttrp) > 0:
            return 3
        else:
            return 2


import multiprocessing
import sys
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser("Event based evaluation")
    parser.add_argument('--p_type', type=int, default=0)
    parser.add_argument('--transition', type=str, default='geq1')
    args = parser.parse_args()

    chunkfiles = glob(base_ep + 'reduced_endpoints_*h5')
    for outfile in chunkfiles:
        dfread = pd.read_hdf(outfile, key='/endpoints')
        dfread = dfread[['PatientID', 'AbsDatetime', args.transition, 'u' + args.transition, 'c' + args.transition, '1.u', '1.i', '1.b', '2.u', '2.b', '3.au', '3.u', '3.b', '3.i', '3.d']]

        grouped = dfread.groupby('PatientID')
        dflist = []
        pdlist = []
        for name, group in grouped:
            dflist.append(group)
            pdlist.append(name)

        for pid, dfp in zip(pdlist, dflist):
            idp = process_p(dfp, args.transition)
            if idp == args.p_type:
                p_df = extract_p_HiRID(pid)
                p_df['AbsDatetime'] = p_df[time_hirid_key]
                p_df = p_df.dropna(how='all', subset=[urine_hirid_key, creatinine_hirid_key])
                ndf = p_df.merge(dfp)
                ndf = ndf.sort_values(by=['AbsDatetime'])
                print(ndf[['PatientID', 'AbsDatetime', weight_hirid_key, urine_hirid_key, creatinine_hirid_key, args.transition, 'u' + args.transition, 'c' + args.transition]])
                if args.p_type == 3 and len(ndf[ndf[args.transition] == 0]) == len(ndf):
                    print(dfp[['PatientID', 'AbsDatetime', args.transition, '1.u', '1.i', '1.b', '2.u', '2.b', '3.au', '3.u', '3.b', '3.i', '3.d']])
