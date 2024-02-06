''' Saves a list of PIDs to be included as a basis for split generation'''

import argparse
import glob
import os
import os.path
import gin
import pickle
import ipdb
import tqdm
import sys

import pandas as pd

from akiews.utils.io import write_list_to_file

def execute(configs):

    # Renal endpoint in internal evaluation mode
    if configs["endpoint"]=="renal":
        valid_ep_pids=list(map(int,pd.read_csv(configs["base_list_ep_config_pids"],sep='\t',
                                header=None)[0].values.tolist()))
        sel_pids=valid_ep_pids
        print("Initial set of valid endpoint PIDs: {}".format(len(sel_pids)))

        if configs["restrict_kanonym"]:
            anon_desc=pd.read_csv(configs["kanonym_pid_list"],sep=',')
            anon_desc=anon_desc[anon_desc[configs["kanonym_col"]].notnull()]
            k_pids=list(anon_desc["patientid"].unique())
            sel_pids=list(set(k_pids).intersection(set(sel_pids)))
            print("Filter with k-anonymity, remaining PIDs: {}".format(len(sel_pids)))

        # Final step, intersect with the PIDs actually available in Xinrui's file
        merged_fs=sorted(glob.glob(os.path.join(configs["merged_path"], "reduced_fmat_*.h5")))
        print("Number of reduced merged files: {}".format(len(merged_fs)))
        merged_pids=[]
        
        for fidx,fpath in enumerate(merged_fs):
            print("Merged file: {}/{}".format(fidx+1,len(merged_fs)))
            df_batch=pd.read_hdf(fpath,mode='r')
            merged_pids.extend(list(df_batch.PatientID.unique()))
            if configs["debug_mode"]:
                break

        sel_pids=list(set(merged_pids).intersection(set(sel_pids)))
        print("Final set of PIDs: {}".format(len(sel_pids)))
        write_list_to_file(configs["inc_pid_list"], sel_pids)

    # Renal endpoint for external validation, for the moment we implement
    # Thomas condition on the endpoints
    if configs["endpoint"]=="renal_extval":

        # Scan the base list of valid PIDs
        tg_inc_pids=list(map(int,pd.read_csv(configs["base_list_ep_config_pids"],sep='\t',
                                header=None)[0].values.tolist()))
        print("Number of PIDs with valid endpoints: {}".format(len(tg_inc_pids)))

        df_merged=pd.read_parquet(configs["merged_path"])
        merged_pids=list(set(map(lambda tp: tp[0], df_merged.index.tolist())))
        print("Number of unique stay IDs in merged: {}".format(len(merged_pids)))
        
        inc_pids=[]
        all_eps=sorted(glob.glob(os.path.join(configs["endpoint_path"],"batch_*.h5")))
        all_eps=list(filter(lambda elem: "-1.h5" not in elem, all_eps))
        assert len(all_eps)==50
        all_pids=[]
        for eidx,epf in enumerate(all_eps):
            print("Endpoint file {}/{}".format(eidx+1,len(all_eps)))
            df_ep=pd.read_hdf(epf,mode='r')
            local_pids=list(df_ep["stay_id"].unique())
            all_pids.extend(local_pids)
            if configs["debug_mode"]:
                break

        print("Number of endpoint PIDs: {}".format(len(all_pids)))
        all_pids=list(set(all_pids).intersection(set(merged_pids)))
        print("Number of PIDs in intersection between merged/endpoints: {}".format(len(all_pids)))

        # Get the final set of included PIDs
        inc_pids=[]            
        for pix,pid in enumerate(all_pids):
            if (pix+1)%100==0:
                print("PID: {}/{}".format(pix+1,len(all_pids)))
                print("Included PIDs: {}/{}".format(len(inc_pids),pix+1))
            if pid in tg_inc_pids:
                inc_pids.append(pid)

        print("Final set of PIDs: {}".format(len(inc_pids)))
                
        write_list_to_file(configs["inc_pid_list"], inc_pids)                
    
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
    parser.add_argument("--debug_mode", default=None, action="store_true", help="Only 1 batch")
    
    parser.add_argument("--gin_config", default="./configs/patient_include_mimic.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/patient_include.gin", help="GIN config to use")
    
    configs=vars(parser.parse_args())
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    execute(configs)
