'''
Imputes static data of all patients and saves back to file-system
'''

import os
import os.path
import sys
import argparse
import ipdb

import numpy as np
import pandas as pd
import gin

from resp_ews.utils.io import load_pickle


def unbin_age(age_lst):
    out_lst=list(map(lambda age_bin: int(age_bin.split("+")[0]) if "+" in age_bin else (int(age_bin.split("-")[0]) + int(age_bin.split("-")[1]))/2, age_lst))
    return out_lst


def unbin_height(height_lst):
    out_lst=[]
    for h_bin in height_lst:
        if h_bin is None:
            convert=np.nan
        elif "+" in h_bin:
            convert=int(h_bin.split("+")[0])
        else:
            comps=h_bin.split("-")
            if not comps[0]=="" and not comps[1]=="":
                convert=(int(h_bin.split("-")[0]) + int(h_bin.split("-")[1]))/2
            else:
                convert=int(h_bin.split("-")[0])
        out_lst.append(convert)
    return out_lst


    
def impute_static_df(df_static,df_train=None, configs=None):
    age_median=int(df_train["Age"].median())
    emergency_mode=int(df_train["Emergency"].mode())
    height_median=float(df_train["Height"].median())
    sex_mode=str(df_train["Sex"].mode().values[0])
    surgical_mode=int(df_train["Surgical"].mode())    

    if configs["database"]=="hirid":
        apache_mode=int(df_train["APACHECode"].mode())
        discharge_mode=int(df_train["Discharge"].mode())

        try:
            euroscores_mode=int(df_train["Euroscores"].mode())
        except:
            euroscores_mode=17 # Encoding of a Euroscores mode

        pat_group_mode=int(df_train["PatGroup"].mode())

        try:
            apache_pat_group_mode=int(df_train["APACHEPatGroup"].mode())
        except:
            apache_pat_group_mode=18 # Encoding of an unknown APACHE Pat Group

        fill_dict={"APACHECode": apache_mode, "Discharge": discharge_mode, 
                   "Emergency": emergency_mode, "Euroscores": euroscores_mode, 
                   "Surgical": surgical_mode, "Height": height_median,
                   "Age": age_median, "PatGroup": pat_group_mode, "Sex": sex_mode,
                   "APACHEPatGroup": apache_pat_group_mode}
    else:
        fill_dict={"Emergency": emergency_mode,"Height": height_median, "Surgical": surgical_mode,
                   "Age": age_median, "Sex": sex_mode}

    if "APACHEPatGroup" not in df_static:
        nan_arr=np.zeros(df_static.shape[0])
        nan_arr[:]=np.nan
        df_static["APACHEPatGroup"]=nan_arr

    if configs["no_impute"]:
        df_static_imputed=df_static
    else:
        df_static_imputed=df_static.fillna(value=fill_dict)

    if configs["database"]=="hirid":
        return df_static_imputed[["PatientID", "Age", "APACHECode", "Discharge", "Emergency", "Euroscores", "Surgical", "Height","PatGroup","APACHEPatGroup","Sex"]]
    elif configs["database"]=="mimic":
        return df_static_imputed[["PatientID", "Age","Emergency","Height","Sex","Surgical"]]
    

def execute(split_key,configs=None):
    data_split=load_pickle(configs["temporal_data_split_binary_path"])[split_key]
    train=data_split["train"]
    val=data_split["val"]
    test=data_split["test"]
    all_pids=train+val+test
    reduced_output_base_dir=configs["imputed_reduced_path"]

    df_static_db=pd.read_hdf(configs["static_info_path"], mode='r')

    if configs["database"]=="mimic":
        df_static_db.reset_index(inplace=True)
        df_static_db.drop(columns=["subject_id"],inplace=True)
        df_static_db.rename(columns={"stay_id": "PatientID",
                                     "gender": "Sex",
                                     "anchor_age": "Age"},inplace=True)

    df_train=df_static_db[df_static_db["PatientID"].isin(train)]
    df_all=df_static_db[df_static_db["PatientID"].isin(all_pids)]
    df_static_imputed=impute_static_df(df_all, df_train=df_train, configs=configs)

    if not configs["no_impute"]:
        assert(df_static_imputed.isnull().sum().sum()==0)
        
    base_dir=os.path.join(reduced_output_base_dir,split_key)

    if not configs["debug_mode"]:
        df_static_imputed.to_hdf(os.path.join(base_dir, "static.h5"),'data',complevel=configs["hdf_comp_level"], complib=configs["hdf_comp_alg"])

@gin.configurable
def parse_gin_args(old_configs,gin_configs=None):
    gin_configs=gin.query_parameter("parse_gin_args.gin_configs")
    for k in old_configs.keys():
        if old_configs[k] is not None:
            gin_configs[k]=old_configs[k]
    gin.bind_parameter("parse_gin_args.gin_configs",gin_configs)
    return gin_configs

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("--debug_mode", default=None, action="store_true", help="Debug mode stores nothing to disk")
    
    parser.add_argument("--gin_config", default="./configs/impute_static.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/impute_static_mimic.gin", help="GIN config to use")    

    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    for split_key in configs["SPLIT_MODES"]:
        print("Imputing static data on split: {}".format(split_key))
        execute(split_key, configs=configs)
