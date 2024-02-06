
#!/usr/bin/env python
import numpy as np
import pandas as pd

import os
import gc

import sys
sys.path.append('../utils')
import preproc_utils

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-tbl')
parser.add_argument('-input_path')
parser.add_argument('-output_path')
parser.add_argument('-voi_path')
parser.add_argument('-version')
parser.add_argument('--idx_vid', type=int, default=None)
args = parser.parse_args()
tbl = args.tbl
input_path = args.input_path
output_path = args.output_path
voi_path = args.voi_path
version = args.version
idx_vid = args.idx_vid


dict_type = dict(monvals='Monitored', dervals='Derived', observrec='Observed',
                 pharmarec='Pharma')
voi = pd.read_csv(os.path.join(voi_path, 'labref_excel_%s.tsv'%version if tbl=='labres' else 'varref_excel_%s.tsv'%version), 
                  sep='\t', encoding='cp1252')
if tbl!='labres':
    voi = voi[voi.Type==dict_type[tbl]]
voi.drop(voi.index[(voi.VariableID.isnull())|(voi.VariableID=='Vm23')], inplace=True)
voi.drop_duplicates('VariableID', inplace=True)
voi.loc[:,'VariableID'] = voi.VariableID.astype(np.int64)
voi.set_index('VariableID', inplace=True)

input_path = os.path.join(input_path, tbl)
# output_path = os.path.join(output_path, version)
if not os.path.exists(output_path):
    os.makedirs(output_path)

if tbl == 'monvals':
    vid = voi.index[idx_vid]
    voi = voi.iloc[idx_vid]
    values = []
    for f in os.listdir(input_path):
        if '.h5' not in f:
            continue
        df = pd.read_hdf(os.path.join(input_path, f), where='VariableID=%d'%vid, columns=['Value'])
        values.extend(df.Value.tolist())
        gc.collect()
    mean_std = [[voi.VariableName, vid, np.mean(values), np.std(values)]]
    mean_std = pd.DataFrame(mean_std, columns=['VariableName', 'VariableID', 'Mean', 'Std'])
    mean_std.to_csv(os.path.join(output_path, tbl+'_%d.tsv'%idx_vid), sep='\t', index=False)
else:
    mean_std = []
    for i, vid in enumerate(voi.index):
        values = []
        for f in os.listdir(input_path):
            if '.h5' not in f:
                continue
            df = pd.read_hdf(os.path.join(input_path, f), where='VariableID=%d'%vid, columns=['Value'])
            values.extend(df.Value.tolist())
            gc.collect()
        mean_std.append([vid, np.mean(values), np.std(values)])
        print('%d / %d'%(i+1, len(voi)))
    mean_std = pd.DataFrame(mean_std, columns=['VariableID', 'Mean', 'Std'])
    mean_std = mean_std.merge(voi[['VariableName']], how='left', left_on='VariableID', right_index=True )
    mean_std.to_csv(os.path.join(output_path, tbl+'.tsv'), sep='\t', index=False)
