#!/usr/bin/env python
import os
import gc
import pickle
import numpy as np
import pandas as pd

import sys
sys.path.append('../utils')
import preproc_utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-id_path')
parser.add_argument('-data_path')
parser.add_argument('-consent_pids_file')
parser.add_argument('-num_chunk', type=int)
parser.add_argument('-output_file')
parser.add_argument('--filter_ecmo', action='store_true')
parser.add_argument('--ecmo_path', default=None)
args = parser.parse_args()
id_path = args.id_path
data_path = args.data_path
ecmo_path = args.ecmo_path
consent_pids_file = args.consent_pids_file
num_chunk = args.num_chunk
output_file = args.output_file
filter_ecmo = args.filter_ecmo

# patient with consents
pID_consent = set(pd.read_csv(consent_pids_file).PatientID)
print('# consenting patients:', len(pID_consent))

# patient who are admitted after 2008
if '1_hdf5_consent' in data_path:
    gd = pd.read_hdf(os.path.join(data_path, 'generaldata.h5'))
    gd.rename(columns={col: col.lower() for col in gd.columns}, inplace=True)
else:
    tbl_path = os.path.join(data_path, 'p_generaldata')
    gd_file = [f for f in os.listdir(tbl_path) if 'crc' not in f and 'parquet' in f][0]
    gd = pd.read_parquet(os.path.join(tbl_path, gd_file), engine='pyarrow')
    gd.loc[:,'admissiontime'] = pd.to_datetime(gd.admissiontime).dt.floor(freq='s')
    
gd['admissionyear'] = gd.admissiontime.dt.year
gd['age'] = gd.admissionyear - gd.birthyear
gd.drop(gd.index[gd.admissionyear<2008], inplace=True)
gc.collect()
pID_pos_08 = set(gd.patientid)
pID_filtered= pID_consent & pID_pos_08
print('# Consenting patients admitted after 08:', len(pID_filtered))

# patient whose age is between 16 and 100
gd.drop(gd.index[(gd.age<16)|(gd.age>100)], inplace=True)
gc.collect()
pID_right_age = set(gd.patientid)
pID_filtered = pID_filtered & pID_right_age
print('# Consenting patients admitted after 08, aged within [16,100]:', len(pID_filtered))

# patient who are not on ECMO
if filter_ecmo:
    if ecmo_path is None:
        print('Please provide path to the file with ECMO information.')
    else:
        patcare = []
        for f in os.listdir(ecmo_path):
            if '.h5' not in f:
                continue
            patcare.append(pd.read_hdf(os.path.join(ecmo_path, f)))
        patcare = pd.concat(patcare).reset_index(drop=True)
        for col in ['name', 'placename']:
            patcare.loc[:,col] = patcare[col].astype(np.str_)
        n_ecmo = patcare.name.apply(lambda x: 'assist' in x.lower() or 'ecmo' in x.lower())
        p_ecmo = patcare.placename.apply(lambda x: 'assist' in x.lower() or 'ecmo' in x.lower())
        patcare = patcare[n_ecmo|p_ecmo]
        gc.collect()
        pID_ecmo = set(patcare.patientid)
        pID_filtered -= pID_ecmo
        print('# Consenting patients admitted after 08, aged within [16,100], not on ECMO:', len(pID_filtered))

pID_filtered = np.sort(list(pID_filtered))

batch_size = int(np.ceil( len(pID_filtered)/num_chunk ))
pids_bids = []
for i in range(num_chunk):
    pids_batch = pID_filtered[i*batch_size:min((i+1)*batch_size,len(pID_filtered))]
    pids_bids.append(np.stack((pids_batch, [i]*len(pids_batch)), axis=1))
pids_bids = np.vstack(tuple(pids_bids))
pids_bids = pd.DataFrame(pids_bids, 
                         columns=['PatientID', 'BatchID'], 
                         dtype=np.int64)

if os.path.exists(output_file):
    print(output_file, 'alreadt exists.')
    print('Please delete to rewrite.')
else:
    pids_bids.to_csv(output_file.replace("PID_all_vars_chunking_100", "PID_all_vars_chunking_100_201103"), index=False)
    pass

pID_filtered = pd.DataFrame(pID_filtered.reshape((-1,1)), columns=['PatientID'])
print('# patients', len(pID_filtered))
pID_filtered.to_csv(os.path.join(id_path, 'PID_all_vars_201103.csv'), index=False)

# # patient with variables of interest
# id_path = os.path.join(id_path, version)
# lst_pID_voi = []
# for tbl in ['monvals', 'dervals', 'observrec', 'pharmarec', 'labres']:
#     tmp = pd.read_csv(os.path.join(id_path, 'PID_w_voi_%s.csv'%tbl))
#     pID_voi = set(tmp.patientid)
#     print('# patient with variables of interest in %s: %d'%(tbl, len(pID_voi)))
#     lst_pID_voi.append(tmp)
    
# pID_voi = lst_pID_voi[0]
# for i in np.arange(1, len(lst_pID_voi)):
#     pID_voi |= lst_pID_voi[i]

# pID_consent_voi = pID_consent & pID_voi
# print('# Consenting patients with variables of interest:', len(pID_consent_voi))
