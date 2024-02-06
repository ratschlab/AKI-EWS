#!/usr/bin/env python
import os
import gc
import numpy as np
import pandas as pd

import sys
sys.path.append('../utils')
import preproc_utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-version')
parser.add_argument('-tbl_name')
parser.add_argument('-input_path')
parser.add_argument('-output_path')
args = parser.parse_args()
version = args.version
tbl_name = args.tbl_name
input_path = args.input_path
output_path = args.output_path

input_path = os.path.join(input_path, version)
voi = preproc_utils.voi_id_name_mapping(tbl_name, version=version[:2])
vID_set = list(set(voi.index))
if tbl_name == 'pharmarec':
    df = pd.read_hdf(os.path.join(input_path, '%s.h5'%tbl_name), 
                     where='PharmaID in (%s)'%(','.join([str(i) for i in vID_set])), 
                     columns=['PatientID', 'PharmaID', 'GivenDose'])
    df.drop(df.index[df.GivenDose.isnull()], inplace=True)
    gc.collect()
    pID_set = df.PatientID.unique()

elif tbl_name == 'monvals':
    chunksize = 10**7
    hdf_iter = pd.read_hhdf_iter(os.path.join(input_path, '%s.h5'%tbl_name), 
                     where='VariableID in (%s)'%(','.join([str(i) for i in vID_set])), 
                     columns=['PatientID', 'VariableID', 'Value'], 
                     chunksize=chunksize)
    pID_set = []
    for df in hdf_iter:
        df.drop(df.index[df.Value.isnull()], inplace=True)
        gc.collect()
        pID_set.extend(df.PatientID.unique())
    pID_set = np.unique(lst_pID)
else:
    df = pd.read_hdf(os.path.join(input_path, '%s.h5'%tbl_name), 
                     where='VariableID in (%s)'%(','.join([str(i) for i in vID_set])), 
                     columns=['PatientID', 'VariableID', 'Value'])
    df.drop(df.index[df.Value.isnull()], inplace=True)
    gc.collect()
    pID_set = df.PatientID.unique()

pID_set = pd.DataFrame(np.reshape(pID_set, (-1,1)), columns=['PatientID'])
output_path = os.path.join(output_path, version)
if not os.path.exists(output_path):
    os.mkdir(output_path)
pID_set.to_csv(os.path.join(output_path, 'PID_w_voi_%s.csv'%tbl_name), 
               index=False)
