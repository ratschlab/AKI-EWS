''' Script to prepare data-set and store back into HDF5 format, only
    for the training/validation sets of a split'''

import os
import os.path
import sys
import random
import timeit
import csv
import warnings
import argparse
import ipdb
import gc
import glob
import itertools
import json
import pickle

import psutil
import time
import gc
import gin

import numpy as np
import pandas as pd
import numpy.random as nprand
import sklearn.dummy as skdummy
import h5py

import akiews.utils.memory as mlhc_memory
import akiews.utils.io as mlhc_io

def extract_var(col):
    ''' Extracts the variable ID from a feature column'''
    if "static_" in col or col in ["RelDatetime"]:
        var=col
    elif "plain_" in col:
        var=col.split("_")[1].strip()
    else:
        var=col.split("_")[0].strip()
    return var


def execute(configs):
    random.seed(2018)
    nprand.seed(2018)
    time_after_begin=timeit.default_timer()

    if configs["profile_report"]:
        cum_time_load_df=0.0
        cum_time_load_shapelet=0.0
        cum_time_load_labels=0.0
        cum_time_merge=0.0
        cum_time_static=0.0
        cum_time_add_data=0.0

    df_batch_buffer={}
    df_label_buffer={}

    split_key=configs["split_key"]
    bern_imputed_base_dir=configs["imputed_dir"]
    bern_ml_input_base_dir=configs["ml_input_dir"]
    problem_dir=os.path.join(bern_ml_input_base_dir,"reduced",configs["split_key"])
    impute_dir=os.path.join(bern_imputed_base_dir, "reduced",configs["split_key"])
    output_dir=os.path.join(configs["output_dir"],"reduced",split_key)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bern_batch_map=mlhc_io.load_pickle(configs["pid_batch_map_path"])["pid_to_chunk"]
    data_split=mlhc_io.load_pickle(configs["temporal_data_split_path"])[split_key]
    train=data_split["train"]
    val=data_split["val"]

    # Subsampling based on patients
    if configs["10percent_sample"]:
        train=train[:int(0.1*len(train))]
        val=val[:int(0.1*len(val))]
    elif configs["10percent_sample_train"]:
        train=train[:int(0.1*len(train))]
    elif configs["5percent_sample_train"]:
        train=train[:int(0.05*len(train))]
    elif configs["1percent_sample_train"]:
        train=train[:int(0.01*len(train))]
    elif configs["2percent_sample_train"]:
        train=train[:int(0.02*len(train))]        
    elif configs["0.1percent_sample_train"]:
        train=train[:int(0.001*len(train))]
    elif configs["25percent_sample_train"]:
        train=train[:int(0.25*len(train))]
    elif configs["50percent_sample_train"]:
        train=train[:int(0.5*len(train))]
    elif configs["1percent_sample_val"]:
        val=val[:int(0.01*len(val))]        
    elif configs["1percent_sample"]:
        train=train[:int(0.01*len(train))]
        val=val[:int(0.01*len(val))]
    elif configs["verysmall_sample"]:
        train=train[:int(0.001*len(train))]
        val=val[:int(0.01*len(val))]

    if configs["restrict_variables"]:
        rest_var_list=mlhc_io.read_list_from_file(configs["var_restrict_path"])
    else:
        rest_var_list=None

    process_info = psutil.Process(os.getpid())
    full_static_df=pd.read_hdf(os.path.join(impute_dir, "static.h5"), mode='r')    

    if configs["restrict_gender"] is not None:
        static_male=list(full_static_df[full_static_df.Sex=="M"].PatientID.unique())
        static_female=list(full_static_df[full_static_df.Sex=="F"].PatientID.unique())
        if configs["restrict_gender"]=="male":
            train=list(set(train).intersection(set(static_male)))
            val=list(set(val).intersection(set(static_male)))
        elif configs["restrict_gender"]=="female":
            train=list(set(train).intersection(set(static_female)))
            val=list(set(val).intersection(set(static_female)))
    
    train=list(sorted(train, key=lambda elem: bern_batch_map[elem]))
    val=list(sorted(val, key=lambda elem: bern_batch_map[elem]))

    print("Number of included training patients: {}".format(len(train)))
    print("Number of included validation patients: {}".format(len(val)))
    
    skip_no_batch_file=0
    skip_no_patdf=0
    skip_no_goodsegment=0
    skip_no_labdf=0
    skip_no_colissue=0
    skip_no_staticdf=0
    n_skipped_patients=0

    if not configs["gender_numeric_encode"]:
        full_static_df.rename(columns={"gender": "Sex"},inplace=True)

    full_static_df["PatientID"]=full_static_df["PatientID"].astype("category")
    final_cols=None

    train_row_idx=0

    metadata_dict={}
    
    X_write_buffer=[]
    Y_write_buffer=[]

    # PID meta-information
    train_pid_long=[]
    val_pid_long=[]

    # Timestamp meta-information
    train_ts_long=[]
    val_ts_long=[]

    # Load the training set patients
    for idx,train_patient in enumerate(train):

        if (idx+1)%100==0:
            print("Train Patient {}/{}: {}, SKIPPED: {}".format(idx+1,len(train),train_patient,n_skipped_patients),flush=True)
            print("SKIP, nbf:{}, npd: {}, ngs: {}, nld: {}, nci: {}".format(skip_no_batch_file, skip_no_patdf, skip_no_goodsegment, skip_no_labdf, skip_no_colissue),
                  flush=True)
            mlhc_memory.print_memory_diags()

        batch_pat=bern_batch_map[train_patient]
        df_ml_path=os.path.join(problem_dir,"batch_{}.h5".format(batch_pat))

        # Skip patients that have no valid file
        if not os.path.exists(df_ml_path):
            n_skipped_patients+=1
            skip_no_batch_file+=1
            continue        

        if df_ml_path not in df_batch_buffer:
            print("Pre-loading data for batch {}".format(batch_pat),flush=True)
            df_batch_buffer={}
            df_label_buffer={}
            df_batch_buffer[df_ml_path]=pd.read_hdf(df_ml_path,"/X",mode='r')
            df_label_buffer[df_ml_path]=pd.read_hdf(df_ml_path,"/y",mode='r')
            df_batch_buffer[df_ml_path].PatientID=df_batch_buffer[df_ml_path].PatientID.astype("category")
            df_label_buffer[df_ml_path].PatientID=df_label_buffer[df_ml_path].PatientID.astype("category")
            gc.collect()

        if configs["profile_report"]:
            t_begin=timeit.default_timer()

        full_df=df_batch_buffer[df_ml_path]
        pat_df=full_df[full_df.PatientID==train_patient]

        if configs["profile_report"]:
            t_end=timeit.default_timer()
            cum_time_load_df+=t_end-t_begin

        if configs["profile_report"]:
            t_begin=timeit.default_timer()

        full_label_df=df_label_buffer[df_ml_path]
        pat_label_df=full_label_df[full_label_df.PatientID==train_patient]

        if pat_label_df.shape[0]==0 or pat_df.shape[0]==0:
            n_skipped_patients+=1
            skip_no_labdf+=1
            continue

        if configs["profile_report"]:
            t_end=timeit.default_timer()
            cum_time_load_labels+=t_end-t_begin
            t_begin=timeit.default_timer()

        static_df=full_static_df[full_static_df["PatientID"]==train_patient]

        if configs["profile_report"]:
            t_end=timeit.default_timer()
            cum_time_static+=t_end-t_begin
        
        if static_df.shape[0]==0:
            n_skipped_patients+=1
            skip_no_staticdf+=1
            continue

        if static_df.shape[0]>1:
            static_df=static_df.head(1)

        cols_X=sorted(pat_df.columns.values.tolist())
        cols_y=sorted(pat_label_df.columns.tolist())
        sel_cols_X=list(filter(lambda col: "Patient" not in col, cols_X))
        sel_cols_y=list(filter(lambda col: "Patient" not in col and "TimeTo" not in col,cols_y))
        X_df=pat_df[sel_cols_X]
        y_df=pat_label_df[sel_cols_y]
        temp_abs_ts=list(X_df["AbsDatetime"])

        if final_cols is None:
            
            if configs["database"]=="mimic":
                final_cols=sel_cols_X+["static_Age","static_Sex","static_Emergency","static_Height","static_Surgical"]
            elif configs["database"]=="hirid":
                final_cols=sel_cols_X+["static_Age","static_APACHEPatGroup","static_Sex","static_APACHECode","static_Emergency",
                                       "static_Surgical","static_Height","static_PatGroup"]
                
            final_static_cols=list(filter(lambda col: "static_" in col, final_cols))

            if configs["restrict_variables"]:
                final_static_cols=list(filter(lambda col: col in rest_var_list, final_static_cols))

            final_static_cols=list(map(lambda col: "_".join(col.split("_")[1:]), final_static_cols))
            final_cols=list(filter(lambda col: "static_" not in col and "SampleStatus" not in col \
                                   and "AbsDatetime" not in col, final_cols))

            # Take out bowen HMM prob columns (only relevant for Kidney)
            final_cols=list(filter(lambda col: "statesProbs" not in col, final_cols))

            if configs["restrict_variables"]:
                final_cols=list(filter(lambda col: extract_var(col) in rest_var_list, final_cols))

            final_cols_y=list(filter(lambda col: col not in ["AbsDatetime","RelDatetime"], sel_cols_y))

        if configs["unify_X_y_grid"]:
            x_grid=list(X_df.AbsDatetime.unique())             
            y_df=y_df[y_df.AbsDatetime.isin(x_grid)]
            
        X_df=X_df[final_cols]
        y_df=y_df[final_cols_y]
        
        assert(y_df.shape[0]==X_df.shape[0])

        static_df=static_df[final_static_cols]

        if configs["gender_numeric_encode"]:
            static_df.replace({"Sex": {"F": 1.0, "M": 0.0, "U": np.nan}},inplace=True)

        if configs["profile_report"]:
            t_begin=timeit.default_timer()

        X_dynamic_arr=np.array(X_df,dtype=np.float32)
        y_arr=np.array(y_df,dtype=np.float32)
        X_static_arr=np.array(static_df,dtype=np.float32)
        X_static_arr=np.repeat(X_static_arr,X_dynamic_arr.shape[0],axis=0)

        X_arr=np.hstack([X_static_arr,X_dynamic_arr])

        # Initialize data-set with maximum size
        if idx==0:
            num_X_cols=X_arr.shape[1]
            num_Y_cols=y_arr.shape[1]
            with h5py.File(os.path.join(output_dir,"ml_input.h5"),'w') as hf:
                print("Preallocating dataset...",flush=True)
                g1=hf.create_group("data")
                g1.attrs["X_columns"]=final_static_cols+final_cols
                #g1.attrs["y_columns"]=final_cols_y (Not possible for large attribute lists)
                dt = h5py.special_dtype(vlen=str)
                dset=g1.create_dataset('y_columns', shape=(len(final_cols_y),), dtype=dt)
                for yidx,y_col in enumerate(final_cols_y):
                    dset[yidx]=y_col
                g1.create_dataset("X_train", dtype=np.float32, fillvalue=0.0,shape=(50000000,num_X_cols),fletcher32=True,compression="gzip")
                g1.create_dataset("y_train", dtype=np.float32, fillvalue=0.0,shape=(50000000,num_Y_cols),fletcher32=True,compression="gzip")
                g1.create_dataset("X_val", dtype=np.float32, fillvalue=0.0,shape=(50000000,num_X_cols),fletcher32=True,compression="gzip")
                g1.create_dataset("y_val", dtype=np.float32, fillvalue=0.0,shape=(50000000,num_Y_cols),fletcher32=True,compression="gzip")

        assert(X_arr.shape[0]==y_arr.shape[0])
        X_write_buffer.append(X_arr)
        Y_write_buffer.append(y_arr)
        train_pid_long.extend([train_patient]*X_arr.shape[0])
        assert(X_arr.shape[0]==len(temp_abs_ts))
        train_ts_long.extend(temp_abs_ts)

        if len(X_write_buffer)>=configs["write_buffer_size"]:
            with h5py.File(os.path.join(output_dir,"ml_input.h5"),'a') as hf:
                X_unit=np.concatenate(X_write_buffer,axis=0)
                Y_unit=np.concatenate(Y_write_buffer,axis=0)
                assert(X_unit.shape[0]==Y_unit.shape[0])
                hf["data"]["X_train"][train_row_idx:train_row_idx+X_unit.shape[0],:]=X_unit
                hf["data"]["y_train"][train_row_idx:train_row_idx+Y_unit.shape[0],:]=Y_unit
            train_row_idx+=X_unit.shape[0]
            X_write_buffer=[]
            Y_write_buffer=[]

        if configs["profile_report"]:
            t_end=timeit.default_timer()
            cum_time_add_data+=t_end-t_begin

        gc.collect()

    if len(X_write_buffer)>=1:
        with h5py.File(os.path.join(output_dir,"ml_input.h5"),'a') as hf:
            X_unit=np.concatenate(X_write_buffer,axis=0)
            Y_unit=np.concatenate(Y_write_buffer,axis=0)
            assert(X_unit.shape[0]==Y_unit.shape[0])
            hf["data"]["X_train"][train_row_idx:train_row_idx+X_unit.shape[0],:]=X_unit
            hf["data"]["y_train"][train_row_idx:train_row_idx+Y_unit.shape[0],:]=Y_unit
        train_row_idx+=X_unit.shape[0]
        X_write_buffer=[]
        Y_write_buffer=[]

    val_row_idx=0

    # Load validation set patients
    skip_no_batch_file=0
    skip_no_patdf=0
    skip_no_goodsegment=0
    skip_no_labdf=0
    skip_no_colissue=0
    skip_no_staticdf=0
    n_skipped_patients=0
    time_after_train_load=timeit.default_timer()
    print("Seconds after training set patients load: {:.3f}".format(time_after_train_load-time_after_begin),flush=True)

    # Load patients from the validation set
    for idx,val_patient in enumerate(val):

        if (idx+1)%100==0:
            print("Val Patient {}/{}: {}, SKIPPED: {}".format(idx+1,len(val),val_patient,n_skipped_patients),flush=True)
            print("SKIP, nbf:{}, npd: {}, ngs: {}, nld: {}, nci: {}".format(skip_no_batch_file, skip_no_patdf, skip_no_goodsegment, skip_no_labdf, skip_no_colissue),
                  flush=True)
            mlhc_memory.print_memory_diags()

        batch_pat=bern_batch_map[val_patient]
        df_ml_path=os.path.join(problem_dir,"batch_{}.h5".format(batch_pat))

        if not os.path.exists(df_ml_path):
            n_skipped_patients+=1
            skip_no_batch_file+=1
            continue        

        if df_ml_path not in df_batch_buffer:
            df_batch_buffer={}
            df_label_buffer={}
            df_batch_buffer[df_ml_path]=pd.read_hdf(df_ml_path,"/X",mode='r')
            df_label_buffer[df_ml_path]=pd.read_hdf(df_ml_path,"/y",mode='r')
            gc.collect()

        full_df=df_batch_buffer[df_ml_path]
        pat_df=full_df[full_df.PatientID==val_patient]
        full_label_df=df_label_buffer[df_ml_path]
        pat_label_df=full_label_df[full_label_df.PatientID==val_patient]
        
        if pat_label_df.shape[0]==0 or pat_df.shape[0]==0:
            n_skipped_patients+=1
            skip_no_labdf+=1
            continue
        
        static_df=pd.read_hdf(os.path.join(impute_dir, "static.h5"), mode='r')

        if not configs["gender_numeric_encode"]:
            static_df.rename(columns={"gender": "Sex"},inplace=True)
        
        static_df=static_df[static_df["PatientID"]==val_patient]
        
        if static_df.shape[0]==0:
            n_skipped_patients+=1
            skip_no_staticdf+=1
            continue

        cols_X=sorted(pat_df.columns.tolist())
        cols_y=sorted(pat_label_df.columns.tolist())
        sel_cols_X=list(filter(lambda col: "Patient" not in col, cols_X))
        sel_cols_y=list(filter(lambda col: "Patient" not in col and "TimeTo" not in col,cols_y))

        X_df=pat_df[sel_cols_X]
        y_df=pat_label_df[sel_cols_y]
        temp_abs_ts=list(X_df["AbsDatetime"])

        if configs["unify_X_y_grid"]:
            x_grid=list(X_df.AbsDatetime.unique())
            y_df=y_df[y_df.AbsDatetime.isin(x_grid)]
        
        X_df=X_df[final_cols]
        y_df=y_df[final_cols_y]
        static_df=static_df[final_static_cols]

        if configs["gender_numeric_encode"]:
            static_df.replace({"Sex": {"F": 1.0, "M": 0.0}},inplace=True)
            
        assert(y_df.shape[0]==X_df.shape[0])

        X_dynamic_arr=np.array(X_df,dtype=np.float32)
        y_arr=np.array(y_df,dtype=np.float32)
        X_static_orig=np.array(static_df,dtype=np.float32)

        # Duplicated info pick the first line
        if X_static_orig.shape[0]>1:
            X_static_orig=X_static_orig[[0],:]
        
        X_static_arr=np.repeat(X_static_orig,X_dynamic_arr.shape[0],axis=0)
        X_arr=np.hstack([X_static_arr,X_dynamic_arr])
        
        assert(X_arr.shape[0]==y_arr.shape[0])
        X_write_buffer.append(X_arr)
        Y_write_buffer.append(y_arr)
        val_pid_long.extend([val_patient]*X_arr.shape[0])
        assert(X_arr.shape[0]==len(temp_abs_ts))
        val_ts_long.extend(temp_abs_ts)

        if len(X_write_buffer)>=configs["write_buffer_size"]:
            with h5py.File(os.path.join(output_dir,"ml_input.h5"),'a') as hf:
                X_unit=np.concatenate(X_write_buffer,axis=0)
                Y_unit=np.concatenate(Y_write_buffer,axis=0)
                assert(X_unit.shape[0]==Y_unit.shape[0])
                hf["data"]["X_val"][val_row_idx:val_row_idx+X_unit.shape[0],:]=X_unit
                hf["data"]["y_val"][val_row_idx:val_row_idx+Y_unit.shape[0],:]=Y_unit
            val_row_idx+=X_unit.shape[0]
            X_write_buffer=[]
            Y_write_buffer=[]        

        gc.collect()

    if len(X_write_buffer)>=1:
        with h5py.File(os.path.join(output_dir,"ml_input.h5"),'a') as hf:
            X_unit=np.concatenate(X_write_buffer,axis=0)
            Y_unit=np.concatenate(Y_write_buffer,axis=0)
            assert(X_unit.shape[0]==Y_unit.shape[0])
            hf["data"]["X_val"][val_row_idx:val_row_idx+X_unit.shape[0],:]=X_unit
            hf["data"]["y_val"][val_row_idx:val_row_idx+Y_unit.shape[0],:]=Y_unit
        val_row_idx+=X_unit.shape[0]
        X_write_buffer=[]
        Y_write_buffer=[]

    # Reshape data-sets to correct dimensions
    with h5py.File(os.path.join(output_dir,"ml_input.h5"),'a') as hf:
        hf["data"]["X_train"].resize((train_row_idx,hf["data"]["X_train"].shape[1]))
        hf["data"]["y_train"].resize((train_row_idx,hf["data"]["y_train"].shape[1]))
        hf["data"]["X_val"].resize((val_row_idx,hf["data"]["X_val"].shape[1]))
        hf["data"]["y_val"].resize((val_row_idx,hf["data"]["y_val"].shape[1]))

    # Save some meta-data like lists of patient-IDs and timestamps
    metadata_dict["train_pids"]=train_pid_long
    metadata_dict["val_pids"]=val_pid_long
    metadata_dict["train_abs_ts"]=train_ts_long
    metadata_dict["val_abs_ts"]=val_ts_long
    assert(len(train_pid_long)==len(train_ts_long))
    assert(len(val_pid_long)==len(val_ts_long))
    
    with open(os.path.join(output_dir,"aux_metadata.pickle"),'wb') as fp:
        pickle.dump(metadata_dict, fp)
        
    time_after_val_load=timeit.default_timer()
    print("Seconds after validation set patients load: {:.3f}".format(time_after_val_load-time_after_begin),flush=True)



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
    parser.add_argument("--debug_mode", default=None, action="store_true", help="Debugging mode that does not write to file-system")
    parser.add_argument("--run_mode", default=None, help="Is the job executed interactive or as batch job?")
    
    #parser.add_argument("--gin_config", default="./configs/save_dset.gin", help="GIN config to use")
    parser.add_argument("--gin_config", default="./configs/save_dset_MIMIC.gin", help="GIN config to use")
    
    parser.add_argument("--split_key", default=None, help="Split to dispatch")

    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)
    run_mode=configs["run_mode"]
    split_key=configs["split_key"]

    execute(configs)

    print("JOB STATUS: EXITED SUCCESSFULLY",flush=True)
