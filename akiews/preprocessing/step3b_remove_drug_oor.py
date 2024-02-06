import pandas  as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-merged_path')
parser.add_argument('-batch_id', type=int)
parser.add_argument('--dataset', default="hirid", choices=["hirid", "umc"])
args = parser.parse_args()
merged_path = args.merged_path
batch_id = args.batch_id
dataset = args.dataset

# if dataset.lower() == "hirid":
#     merged_path = "/cluster/work/grlab/clinical/hirid2/research/3_merged/v8/reduced"
# else:
#     merged_path = "/cluster/work/grlab/clinical/umcdb/preprocessed/merged/2021-11-10"

varref = pd.read_csv("/cluster/work/grlab/clinical/hirid2/research/misc_derived/ref_excel/varref_excel_v8_drug.tsv", sep="\t")
drug_ref = varref[(varref.Type=="Pharma")&(varref.MetaVariableID.isin([39, 41, 69, 77, 80, 82, 95]))][["MetaVariableID", "LowerBound", "UpperBound"]].drop_duplicates(["MetaVariableID", "LowerBound", "UpperBound"])
drug_ref.loc[:,"MetaVariableID"] = drug_ref.MetaVariableID.apply(lambda x: "pm%d"%x)
drug_ref = drug_ref.set_index("MetaVariableID")
drug_ref = drug_ref.dropna()
drug_ref.loc["pm83","LowerBound"] = 0
drug_ref.loc["pm83","UpperBound"] = 1
drug_ref = drug_ref.sort_index()

if dataset.lower() == "hirid":
    for f in os.listdir(merged_path):
        if "_%d_"%batch_id in f:
            break
else:
    lst_f = np.sort([f for f in os.listdir(merged_path)])
    f = lst_f[batch_id]

if dataset.lower() == "hirid":
    df_iter = pd.read_hdf(os.path.join(merged_path,f), chunksize=10**5)
    for df in df_iter:
        for pm in drug_ref.index:
            oor_index = df.index[(df[pm]<drug_ref.loc[pm,"LowerBound"])|(df[pm]>drug_ref.loc[pm,"UpperBound"])]
            df.loc[oor_index,pm] = np.nan
        df.to_hdf(os.path.join(merged_path.replace("reduced","reduced_rm_drugoor"),f), 'fmat', append=True, complevel=5, 
                              complib='blosc:lz4', data_columns=['PatientID'], format='table')
else:
    df = pd.read_parquet(os.path.join(merged_path,f))
    for pm in drug_ref.index:
        if pm not in df.columns:
            print(pm)
            continue
        oor_index = df.index[(df[pm]<drug_ref.loc[pm,"LowerBound"])|(df[pm]>drug_ref.loc[pm,"UpperBound"])]
        df.loc[oor_index,pm] = np.nan
    if not os.path.exists(merged_path+"_rm_drugoor"):
        os.mkdir(merged_path+"_rm_drugoor")
    df.to_parquet(os.path.join(merged_path+"_rm_drugoor", f))

    