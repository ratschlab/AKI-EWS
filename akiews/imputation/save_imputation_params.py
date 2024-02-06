'''
Saves the statistics of the sampling frequencies which is later used 
during adaptive imputation
'''

import ipdb
import os
import os.path
import sys
import pickle
import glob
import timeit
import random
import argparse
import gin

import pandas as pd
import numpy as np
import scipy.stats as sp_stats

import mlhc_data_manager.util.io as mlhc_io
import mlhc_data_manager.util.math as mlhc_math


def execute(configs):
    ''' Saves global imputation parameters conditional on the split key'''
    split_key=configs["split_key"]
    print("Saving imputation parameters for split: {}".format(split_key),flush=True)
    pid_batch_dict=mlhc_io.load_pickle(configs["pid_batch_map_binary"])["pid_to_chunk"]
    split_desc=mlhc_io.load_pickle(configs["temporal_data_split_binary"])
    relevant_pids=split_desc[split_key]["train"]
    random.seed(configs["random_seed"])
    random.shuffle(relevant_pids)
    batch_schedule={}
    median_interval_pp={}
    iqr_interval_pp={}

    # Build a temporary reverse dictionary mapping the batch directory to the PIDs that have to be loaded in that batch
    for pid in relevant_pids:
        batch_pid=pid_batch_dict[pid]
        if batch_pid not in batch_schedule:
            batch_schedule[batch_pid]=[]
        batch_schedule[batch_pid].append(pid)

    print("Computed batch schedule on-the-fly...",flush=True)
    t_begin=timeit.default_timer()
    n_skipped_patients=0

    if configs["small_sample"]:
        random.shuffle(relevant_pids)
        relevant_pids=relevant_pids[:100]

    for pidx,pid in enumerate(relevant_pids):
        if (pidx+1)%100==0:
            print("#patients: {}/{}".format(pidx+1, len(relevant_pids)),flush=True)
            print("Skipped patients: {}".format(n_skipped_patients),flush=True)
            t_end=timeit.default_timer()
            time_pp=(t_end-t_begin)/pidx
            eta_time=time_pp*(len(relevant_pids)-pidx)/3600.0
            print("ETA in hours: {:.3f}".format(eta_time),flush=True)

        batch_pid=pid_batch_dict[pid]
        cand_files=glob.glob(os.path.join(configs["reduced_merged_path"], "reduced_fmat_{}_*.h5".format(batch_pid)))

        assert(len(cand_files)==1)
        df_patient=pd.read_hdf(cand_files[0], mode='r', where="PatientID={}".format(pid))

        if df_patient.shape[0]==0:
            n_skipped_patients+=1

        df_patient.sort_values(by="Datetime", kind="mergesort", inplace=True)

        if pidx==0:
            all_vars=df_patient.columns.values.tolist()
            relevant_vars=set(all_vars).difference(set(["Datetime", "PatientID"]))

        # Analyze each variable in turn
        for var in relevant_vars:

            # No imputation parameters are saved for pharma variables
            if var[0]=="p":
                continue

            value_vect=df_patient[var].dropna()
            
            # No observations for this variable
            if value_vect.size==0:
                continue

            ts_value_vect=df_patient[["Datetime",var]].dropna()
            
            # At least 3 observations for 2 diffs to make MEDIAN/IQR meaningful
            if ts_value_vect.shape[0]>=3:
                diff_series=np.array(ts_value_vect["Datetime"].diff().dropna(),dtype=np.float64)/1000000000.0
                median_interval=np.median(diff_series)
                iqr_interval=sp_stats.iqr(diff_series)
                
                if var not in median_interval_pp:
                    median_interval_pp[var]=[]
                    iqr_interval_pp[var]=[]

                median_interval_pp[var].append(median_interval)
                iqr_interval_pp[var].append(iqr_interval)

    interval_median_dict={}
    interval_iqr_dict={}

    # Collect the statistics from the patients
    for var in relevant_vars:

        # Interval statistics not required for pharma variables
        if var[0]=="p":
            continue

        if var not in median_interval_pp:
            interval_median_dict[var]=np.nan
            interval_iqr_dict[var]=np.nan
            continue

        interval_median_dict[var]=np.median(np.array(median_interval_pp[var]))
        interval_iqr_dict[var]=np.median(np.array(iqr_interval_pp[var]))

    output_impute_dict=configs["imputation_param_dict_reduced"]

    if not configs["debug_mode"]:
        mlhc_io.save_pickle(interval_median_dict,os.path.join(output_impute_dict,"interval_median_{}.pickle".format(split_key)))
        mlhc_io.save_pickle(interval_iqr_dict,os.path.join(output_impute_dict, "interval_iqr_{}.pickle".format(split_key)))

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
    parser.add_argument("--split_key", default=None, help="For which split should imputation parameters be computed?")
    parser.add_argument("--run_mode", default=None, help="Execution mode")
    parser.add_argument("--gin_config", default="./gin_configs/impute_save_parameters_kidney.gin")
    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    run_mode=configs["run_mode"]
    assert(run_mode in ["CLUSTER","INTERACTIVE"])
    split_key=configs["split_key"]

    if run_mode=="CLUSTER":
        sys.stdout=open(os.path.join(configs["log_dir"],"IMPUTEPARAMS_{}_{}.stdout".format(configs["endpoint"],split_key)),'w')
        sys.stderr=open(os.path.join(configs["log_dir"],"IMPUTEPARAMS_{}_{}.stderr".format(configs["endpoint"],split_key)),'w')

    execute(configs)
    
