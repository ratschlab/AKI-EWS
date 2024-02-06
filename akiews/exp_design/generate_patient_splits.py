'''
Generates data splits, for the HIRID-II data-set we use
the split posted by Alizee, and follow it for the train/test split. A random 
split will be drawn to find train/validation set. There are further 5 random splits, 
which are used to get uncertainty estimates/error bars. Further there are 5 exploration
splits also added, which are currently un-used. For the MIMIC-IV data-set 5 random splits 
will be drawn because no timing information is available.
'''

import random
import csv
import argparse
import ipdb
import os
import pickle

import numpy as np
import pandas as pd
import gin

from akiews.utils.io import load_pickle, read_list_from_file, save_pickle
from akiews.utils.admissions import lookup_admission_category, lookup_admission_time

def execute(configs):
    all_pids=list(map(lambda s: int(float(s)), read_list_from_file(configs["pid_included_list"])))
    print("Included number of PIDs: {}".format(len(all_pids)))

    first_date_map=[]
    if configs["endpoint"]=="renal_extval":
        pid_key="stay_id"
    else:
        pid_key="PatientID"
    
    df_patient_full=pd.read_hdf(configs["general_data_table_path"], mode='r')

    if configs["endpoint"]=="renal_extval":
        df_patient_full.reset_index(inplace=True)
        
    static_pids=list(df_patient_full[pid_key].unique())
    all_pids=list(set(all_pids).intersection(set(static_pids)))
    print("Number of PIDs after excluding PIDs without static information: {}".format(len(all_pids)))

    # Restrict the base cohort to patients between 2010 and 2018
    if configs["restrict_2010_2018"]:
        print("Patients before filtering: {}".format(len(first_date_map)))
        first_date_map=list(filter(lambda item: item[1]>=np.datetime64("2010-01-01T00:00:00.000000000") and item[1]>=np.datetime64("2010-01-01T00:00:00.000000000"), first_date_map))
        print("Patients after filtering: {}".format(len(first_date_map)))

    # Restrict to patients with LOS > 1 day
    elif configs["restrict_los_gt_1_day"]:
        los_pids=list(map(lambda s: int(float(s)), read_list_from_file(configs["los_1_day_list"])))
        print("Number of PIDs before exclusion: {}".format(len(first_date_map)))
        first_date_map=list(filter(lambda item: item[0] in los_pids, first_date_map))
        print("Number of PIDs after exclusion: {}".format(len(first_date_map)))

    # Find matching random sample to LOS 1 > day
    elif configs["match_los_gt_1_day"]:
        los_pids=list(map(lambda s: int(float(s)), read_list_from_file(configs["los_1_day_list"])))
        print("Number of PIDs before exclusion: {}".format(len(first_date_map)))
        first_date_map_tent=list(filter(lambda item: item[0] in los_pids, first_date_map))
        num_pid_exclusions=len(first_date_map_tent)
        first_date_map=random.sample(first_date_map,num_pid_exclusions)
        print("Number of PIDs after exclusion: {}".format(len(first_date_map)))    

    print("Generating temporal splits...")

    out_dict={}

    # HiRID-II data-set, generate temporal splits
    if not configs["endpoint"]=="renal_extval":
        all_frame=pd.read_csv(configs["kanonym_pid_list"],sep=',')

        # Get the main split from Alizee's descriptor for the temporal split strategy, draw random otherwise
        if configs["train_val_split_strategy"]=="temporal":
            test_fm=all_frame[all_frame[configs["test_set_col"]]==True]
            train_val_fm=all_frame[(all_frame[configs["test_set_col"]]==False)]
            test_pids=list(map(int, test_fm["patientid"].unique()))
            train_val_pids=list(map(int, train_val_fm["patientid"].unique()))
        elif configs["train_val_split_strategy"]=="random":
            test_pids=list(kanonym_desc[kanonym_desc[configs["kanonym_col"]]==True].patientid.unique())
            train_val_pids=list(kanonym_desc[kanonym_desc[configs["kanonym_col"]]==False].patientid.unique())
            random.shuffle(train_val_pids)
            train_pids=train_pids[:int(configs["temporal_train_ratio"]*len(train_pids))]
            val_pids=train_pids[int(configs["temporal_train_ratio"]*len(train_pids)):]
        else:
            assert False
            
        test_pids=list(set(all_pids).intersection(set(test_pids)))
        train_val_pids=list(set(all_pids).intersection(set(train_val_pids)))

        # Temporal 1-5 splits
        for ridx in range(5):
            random.shuffle(train_val_pids)
            local_dict={}
            local_dict["train"]=train_val_pids[:int(configs["temporal_train_ratio"]*len(train_val_pids))]
            local_dict["val"]=train_val_pids[int(configs["temporal_train_ratio"]*len(train_val_pids)):]
            local_dict["test"]=test_pids
            print("Split temporal {}, train set: {}, val set: {}, test set: {}".format(ridx+1,len(local_dict["train"]), len(local_dict["val"]), len(local_dict["test"])))
            out_dict["temporal_{}".format(ridx+1)]=local_dict

    else:
        random.shuffle(all_pids)
        test_pids=all_pids[:int(configs["random_test_ratio"]*len(all_pids))]
        train_val_pids=all_pids[int(configs["random_test_ratio"]*len(all_pids)):]

        for ridx in range(5):
            random.shuffle(train_val_pids)
            local_dict={}
            local_dict["train"]=train_val_pids[:int(configs["random_train_ratio"]*len(train_val_pids))]
            local_dict["val"]=train_val_pids[int(configs["random_train_ratio"]*len(train_val_pids)):]
            local_dict["test"]=test_pids
            print("Split random {}, train set: {}, val set: {}, test set: {}".format(ridx+1,len(local_dict["train"]), len(local_dict["val"]), len(local_dict["test"])))
            out_dict["random_{}".format(ridx+1)]=local_dict 

    if configs["debug_mode"]:
        return

    save_pickle(out_dict, configs["temporal_data_split_binary_path"])
    

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

    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debugging mode, no output to file-system")
    
    parser.add_argument("--gin_config", default="./configs/gen_splits_mimic.gin", help="Location of GIN config to load, and overwrite the arguments")
    #parser.add_argument("--gin_config", default="./configs/gen_splits.gin", help="Location of GIN config to load, and overwrite the arguments")    

    args=parser.parse_args()
    configs=vars(args)

    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    execute(configs)
