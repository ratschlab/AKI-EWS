
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
import math
import warnings
import copy
import tqdm

import scipy as sp
import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
import sklearn.feature_selection as skfselect
import sklearn.impute as skimpute
import sklearn.preprocessing as skpproc
import sklearn.impute as skimpute
import sklearn.dummy as skdummy
import sklearn.compose as skcompose
import sklearn.linear_model as sklm
import sklearn.neural_network as sknn
import shap
import numpy.random as nprand
import lightgbm
import xgboost
import catboost
import h5py
from flaml import AutoML

import akiews.utils.memory as mlhc_memory
import akiews.utils.io as mlhc_io

import ml_utils

def extract_var(col):
    ''' Extracts the variable ID from a feature column'''
    if "plain_" in col:
        var=col.split("_")[1].strip()
    elif col[:2] in ["pm","vm"] or "fio2estimated" in col or "ventstate" in col:
        var=col.split("_")[0].strip()
    else:
        var=col  
    return var

def select_feat_cols(X_col_names,restrict_vars=False, restrict_var_list=None, only_base_features=False,
                     only_plain_vars=False, restrict_feats=False, restrict_feat_list=None,
                     configs=None):
    ''' Return a set of feature columns and feature indices in the correct order based
        on combination of selection options'''

    new_X_col_idxs=np.arange(len(X_col_names))
    
    # Restrict variables to a list filter
    if restrict_vars:
        
        if configs["add_bowen"]:
            new_X_col_idxs=list(filter(lambda col_idx: extract_var(X_col_names[col_idx]) in restrict_var_list or "bowen" in X_col_names[col_idx],new_X_col_idxs))
        elif configs["only_bowen"]:
            new_X_col_idxs=list(filter(lambda col_idx: "bowen" in X_col_names[col_idx],new_X_col_idxs))
        else:
            new_X_col_idxs=list(filter(lambda col_idx: extract_var(X_col_names[col_idx]) in restrict_var_list,new_X_col_idxs))

    # Restrict features to a list filter
    if restrict_feats:

        if configs["add_bowen"]:
            new_X_col_idxs=list(filter(lambda col_idx: X_col_names[col_idx] in restrict_feat_list or "bowen" in X_col_names[col_idx],new_X_col_idxs))
        elif configs["only_bowen"]:
            new_X_col_idxs=list(filter(lambda col_idx: "bowen" in X_col_names[col_idx],new_X_col_idxs)) 
        else:
            new_X_col_idxs=list(filter(lambda col_idx: X_col_names[col_idx] in restrict_feat_list,new_X_col_idxs)) 

    # Use only plain variable filter
    if only_plain_vars:
        new_X_col_idxs=list(filter(lambda col_idx: "plain_" in X_col_names[col_idx] or extract_var(X_col_names[col_idx])[:2] not in ["vm","pm","fi","ve"],new_X_col_idxs))

    # Ablate static filter
    if configs["ablate_static"]:
        new_X_col_idxs=list(filter(lambda col_idx: X_col_names[col_idx] not in ["Age","Sex","PatGroup"],new_X_col_idxs))

    # Ablate measurement filter
    if configs["ablate_measurement"]:
        new_X_col_idxs=list(filter(lambda col_idx: "meas_density" not in X_col_names[col_idx] and "time_to_last_ms" not in X_col_names[col_idx],new_X_col_idxs))

    # Ablate multi-resolution history filter
    if configs["ablate_multiresolution"]:
        new_X_col_idxs=list(filter(lambda col_idx: "_mean_" not in X_col_names[col_idx] and "_median_" not in X_col_names[col_idx] \
                                   and "_mode_" not in X_col_names[col_idx] and "_iqr_" not in X_col_names[col_idx] \
                                   and "_trend_" not in X_col_names[col_idx],new_X_col_idxs))

    # Ablate instability filter
    if configs["ablate_instability"]:
        new_X_col_idxs=list(filter(lambda col_idx: "_instable_" not in X_col_names[col_idx],new_X_col_idxs))

    # Multires only short filter
    if configs["multires_only_short"]:
        new_X_col_idxs=list(filter(lambda col_idx: "plain_" in X_col_names[col_idx] or "H10" in X_col_names[col_idx],new_X_col_idxs))

    # Multires plus medium horizon filter
    if configs["multires_plus_med"]:
        new_X_col_idxs=list(filter(lambda col_idx: "plain_" in X_col_names[col_idx] or "H10" in X_col_names[col_idx] \
                                   or "H25" in X_col_names[col_idx],new_X_col_idxs))

    if configs["remove_reltime"]:
        new_X_col_idxs=list(filter(lambda col_idx: "RelDatetime" not in X_col_names[col_idx],new_X_col_idxs))

    # Multires plus long horizon filter
    if configs["multires_plus_long"]:
        new_X_col_idxs=list(filter(lambda col_idx: "plain_" in X_col_names[col_idx] or "H10" in X_col_names[col_idx] \
                                   or "H25" in X_col_names[col_idx] or "H62" in X_col_names[col_idx],new_X_col_idxs))

    # Multires plus longest filter
    if configs["multires_plus_longest"]:
        new_X_col_idxs=list(filter(lambda col_idx: "plain_" in X_col_names[col_idx] or "H10" in X_col_names[col_idx] \
                                   or "H25" in X_col_names[col_idx] or "H62" in X_col_names[col_idx] or "H156" in X_col_names[col_idx],new_X_col_idxs))

    # Multi-res only longest filter
    if configs["multires_only_longest"]:
        new_X_col_idxs=list(filter(lambda col_idx: "H156" in X_col_names[col_idx],new_X_col_idxs))

    # Summary location filter
    if configs["summary_loc"]:
        new_X_col_idxs=list(filter(lambda col_idx: "plain_" in X_col_names[col_idx] or "_median_" in X_col_names[col_idx] \
                                   or "_mode_" in X_col_names[col_idx] or "_mean_" in X_col_names[col_idx],new_X_col_idxs))

    # Summary location+trend filter
    if configs["summary_loc_trend"]:
        new_X_col_idxs=list(filter(lambda col_idx: "plain_" in X_col_names[col_idx] or "_median_" in X_col_names[col_idx] \
                                   or "_mode_" in X_col_names[col_idx] or "_mean_" in X_col_names[col_idx] \
                                   or "_trend_" in X_col_names[col_idx],new_X_col_idxs))

    # Summary all filter
    if configs["summary_all"]:
        new_X_col_idxs=list(filter(lambda col_idx: "plain_" in X_col_names[col_idx] or "_median_" in X_col_names[col_idx] \
                                   or "_mode_" in X_col_names[col_idx] or "_mean_" in X_col_names[col_idx] \
                                   or "_trend_" in X_col_names[col_idx] or "_iqr_" in X_col_names[col_idx],new_X_col_idxs))

    # Use only base features that are available for all variables
    if only_base_features:
        new_X_col_idxs=list(filter(lambda col_idx: ("_instable_" not in X_col_names[col_idx] \
                                   and "_H25" not in X_col_names[col_idx] \
                                   and "_H62" not in X_col_names[col_idx] \
                                   and "_H156" not in X_col_names[col_idx]) or extract_var(X_col_names[col_idx])[:2] not in ["vm","pm"],new_X_col_idxs))

    X_col_names_orig=[X_col_names[idx] for idx in new_X_col_idxs]            
    X_col_names=[X_col_names[idx] for idx in new_X_col_idxs]
    return (new_X_col_idxs,X_col_names,X_col_names_orig)

def execute(configs):
    ''' Online fitting, hyperparameter optimization against the validation data, and predicting on testing data,
        saving the results back to disk'''
    random.seed(configs["random_state"])
    nprand.seed(configs["random_state"])
    time_after_begin=timeit.default_timer()

    if configs["profile_report"]:
        cum_time_load_df=0.0
        cum_time_load_shapelet=0.0
        cum_time_load_labels=0.0
        cum_time_merge=0.0
        cum_time_static=0.0
        cum_time_add_data=0.0

    univariate_test=configs["univariate_test"]
    hirid_schema_dict=mlhc_io.load_pickle(configs["hirid_v8_dict"]) 

    check_order=None
    first_fit_status=False
    
    static_cols_without_encode=configs["static_cols_without_encode"]
    static_cols_one_hot_encode=configs["static_cols_one_hot_encode"]
    static_cols_one_hot_encode_str=configs["static_cols_one_hot_encode_str"]
    static_cols_without_encode_final=None
    static_cols_one_hot_encode_final=None
    static_cols_one_hot_encode_str_final=None

    unique_values=configs["unique_values_cat"]
    str_to_int_dict={"M": 0, "F": 1, "U": np.nan}
    configs["str_to_int_sex"]=str_to_int_dict

    pos_to_negative_upsample_factor=3
    neg_only_sample_factor=0.1

    model_type=configs["ml_model"]

    # Assign the correct split keys depending on the training setting we would
    # like to explore
    if configs["ext_val_mode"]=="internal":
        train_split_key=configs["bern_split_key"]
        test_split_key=configs["bern_split_key"]
        train_imputed_base_dir=configs["bern_imputed_dir"]
        train_ml_input_base_dir=configs["bern_ml_input_dir"]
        train_dset_dir=configs["bern_ml_dset_dir"]
        test_imputed_base_dir=configs["bern_imputed_dir"]
        test_ml_input_base_dir=configs["bern_ml_input_dir"]
        test_dset_dir=configs["bern_ml_dset_dir"]
        train_batch_map=mlhc_io.load_pickle(configs["bern_pid_batch_map_path"])["pid_to_chunk"]
        test_batch_map=mlhc_io.load_pickle(configs["bern_pid_batch_map_path"])["pid_to_chunk"]
        train_data_split=mlhc_io.load_pickle(configs["bern_temporal_data_split_path"])[train_split_key]
        test_data_split=mlhc_io.load_pickle(configs["bern_temporal_data_split_path"])[test_split_key]                
    elif configs["ext_val_mode"]=="validation":
        train_split_key=configs["bern_split_key"]
        test_split_key=configs["mimic_split_key"]
        train_imputed_base_dir=configs["bern_imputed_dir"]
        train_ml_input_base_dir=configs["bern_ml_input_dir"]
        train_dset_dir=configs["bern_ml_dset_dir"]
        test_imputed_base_dir=configs["mimic_imputed_dir"]
        test_ml_input_base_dir=configs["mimic_ml_input_dir"]
        test_dset_dir=configs["mimic_ml_dset_dir"]
        train_batch_map=mlhc_io.load_pickle(configs["bern_pid_batch_map_path"])["pid_to_chunk"]
        test_batch_map=mlhc_io.load_pickle(configs["mimic_pid_batch_map_path"])["pid_to_chunk"]
        train_data_split=mlhc_io.load_pickle(configs["bern_temporal_data_split_path"])[train_split_key]
        test_data_split=mlhc_io.load_pickle(configs["mimic_temporal_data_split_path"])[test_split_key]                        
    elif configs["ext_val_mode"]=="retrain":
        train_split_key=configs["mimic_split_key"]
        test_split_key=configs["mimic_split_key"]
        train_imputed_base_dir=configs["mimic_imputed_dir"]
        train_ml_input_base_dir=configs["mimic_ml_input_dir"]
        train_dset_dir=configs["mimic_ml_dset_dir"] 
        test_imputed_base_dir=configs["mimic_imputed_dir"]
        test_ml_input_base_dir=configs["mimic_ml_input_dir"]
        test_dset_dir=configs["mimic_ml_dset_dir"]
        train_batch_map=mlhc_io.load_pickle(configs["mimic_pid_batch_map_path"])["pid_to_chunk"]
        test_batch_map=mlhc_io.load_pickle(configs["mimic_pid_batch_map_path"])["pid_to_chunk"]
        train_data_split=mlhc_io.load_pickle(configs["mimic_temporal_data_split_path"])[train_split_key]
        test_data_split=mlhc_io.load_pickle(configs["mimic_temporal_data_split_path"])[test_split_key]
    
    label_key=configs["label_key"]

    column_desc=configs["column_set"]
    print("Fitting model with label: {}".format(label_key),flush=True)
    
    train_problem_dir=os.path.join(train_ml_input_base_dir,"reduced",train_split_key)
    train_impute_dir=os.path.join(train_imputed_base_dir, "reduced",train_split_key)
    train_dset_dir=os.path.join(train_dset_dir,"reduced",train_split_key)
    test_problem_dir=os.path.join(test_ml_input_base_dir,"reduced",test_split_key)
    test_impute_dir=os.path.join(test_imputed_base_dir, "reduced",test_split_key)
    test_dset_dir=os.path.join(test_dset_dir,"reduced",test_split_key)

    if not configs["special_test_set"]=="NONE":
        special_test_problem_dir=os.path.join(bern_ml_input_base_dir,"reduced",configs["special_test_set"], "{}_{}_{}".format(label_key))
        special_test_impute_dir=os.path.join(bern_imputed_base_dir,"reduced", configs["special_test_set"])

    output_dir=os.path.join(configs["output_dir"],"reduced",train_split_key,"{}_{}_{}".format(label_key, column_desc, model_type))
    train_subsample = configs["negative_subsampling"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not configs["special_test_set"]=="NONE":
        assert(configs["dataset"]=="bern")
        special_test_split=mlhc_io.load_pickle(configs["bern_temporal_data_split_path"])[configs["special_test_set"]]

    eval_set=test_data_split["val"]+test_data_split["test"]
    
    process_info = psutil.Process(os.getpid())

    print("Loading data from H5PY...")

    if configs["save_ml_inputs"]:
        cum_pid_train=[]
        cum_abs_ts_train=[]
        cum_pid_val=[]
        cum_abs_ts_val=[]
        train_val_metadata=pickle.load(open(os.path.join(train_dset_dir,"aux_metadata.pickle"),'rb'))
        train_meta_pid=train_val_metadata["train_pids"]
        val_meta_pid=train_val_metadata["val_pids"]
        train_meta_abs_ts=train_val_metadata["train_abs_ts"]
        val_meta_abs_ts=train_val_metadata["val_abs_ts"]
    
    # Load training/validation data from H5PY
    with h5py.File(os.path.join(train_dset_dir,"ml_input.h5"),'r') as h5f:
        dsets=h5f["data"]
        full_X=dsets["X_train"]
        full_X_val=dsets["X_val"]
        full_y=dsets["y_train"]
        full_y_val=dsets["y_val"]

        X_col_names=list(dsets.attrs["X_columns"])
        X_col_names_orig=list(dsets.attrs["X_columns"])
        y_col_names=list(dsets["y_columns"])

        if configs["restrict_variables"]:
            restrict_var_list=mlhc_io.read_list_from_file(configs["var_restrict_path"])
        else:
            restrict_var_list=None

        if configs["restrict_feats"]:
            restrict_feat_list=mlhc_io.read_list_from_file(configs["feat_restrict_path"])
        else:
            restrict_feat_list=None
        
        new_X_col_idxs,X_col_names,X_col_names_orig=select_feat_cols(X_col_names,restrict_vars=configs["restrict_variables"],
                                                                     restrict_feats=configs["restrict_feats"],
                                                                     restrict_var_list=restrict_var_list,
                                                                     restrict_feat_list=restrict_feat_list,
                                                                     only_base_features=configs["only_base_feats"],
                                                                     only_plain_vars=configs["only_plain_vars"], configs=configs)

        print("Training matrix shape before filter: {}".format(full_X.shape))
        print("Val matrix shape before filter: {}".format(full_X_val.shape))

        y_col_idx=y_col_names.index(configs["label_key"].encode("UTF-8"))
        full_y=np.array(full_y[:,y_col_idx])
        full_y_val=np.array(full_y_val[:,y_col_idx])

        # Fetch training sets
        print("Fetching training set...",flush=True)
        row_fetch_idx=0
        chunk_idx=0
        chunk_collect_X=[]
        chunk_collect_y=[]
        while row_fetch_idx<full_X.shape[0]:
            print("Loading chunk: {}".format(chunk_idx),flush=True)
            X_chunk=np.array(full_X[row_fetch_idx:min(row_fetch_idx+1000000,full_X.shape[0]),:])
            X_chunk=X_chunk[:,new_X_col_idxs]
            y_chunk=full_y[row_fetch_idx:min(row_fetch_idx+1000000,full_X.shape[0])]

            if configs["save_ml_inputs"]:
                pid_chunk=np.array(train_meta_pid[row_fetch_idx:min(row_fetch_idx+1000000,full_X.shape[0])])
                abs_ts_chunk=np.array(train_meta_abs_ts[row_fetch_idx:min(row_fetch_idx+1000000,full_X.shape[0])])
                pid_chunk=pid_chunk[np.isfinite(y_chunk)]
                abs_ts_chunk=abs_ts_chunk[np.isfinite(y_chunk)]
                cum_pid_train.extend(list(pid_chunk))
                cum_abs_ts_train.extend(list(abs_ts_chunk)) 
            
            chunk_idx+=1
            row_fetch_idx+=1000000
            X_chunk=X_chunk[np.isfinite(y_chunk),:]
            y_chunk=y_chunk[np.isfinite(y_chunk)]
            
            chunk_collect_X.append(X_chunk)
            chunk_collect_y.append(y_chunk)
            
        full_X=np.concatenate(chunk_collect_X,axis=0)
        full_y=np.concatenate(chunk_collect_y)

        # Fetch validation sets
        print("Fetching validation set...")
        row_fetch_idx=0
        chunk_idx=0
        chunk_collect_X=[]
        chunk_collect_y=[]
        while row_fetch_idx<full_X_val.shape[0]:
            print("Loading chunk: {}".format(chunk_idx),flush=True)
            X_chunk=np.array(full_X_val[row_fetch_idx:min(row_fetch_idx+1000000,full_X_val.shape[0]),:])
            X_chunk=X_chunk[:,new_X_col_idxs]
            y_chunk=full_y_val[row_fetch_idx:min(row_fetch_idx+1000000,full_X_val.shape[0])]

            if configs["save_ml_inputs"]:
                pid_chunk=np.array(val_meta_pid[row_fetch_idx:min(row_fetch_idx+1000000,full_X_val.shape[0])])
                abs_ts_chunk=np.array(val_meta_abs_ts[row_fetch_idx:min(row_fetch_idx+1000000,full_X_val.shape[0])])
                pid_chunk=pid_chunk[np.isfinite(y_chunk)]
                abs_ts_chunk=abs_ts_chunk[np.isfinite(y_chunk)]
                cum_pid_val.extend(list(pid_chunk))
                cum_abs_ts_val.extend(list(abs_ts_chunk)) 
            
            chunk_idx+=1
            row_fetch_idx+=1000000
            X_chunk=X_chunk[np.isfinite(y_chunk),:]
            y_chunk=y_chunk[np.isfinite(y_chunk)]
            chunk_collect_X.append(X_chunk)
            chunk_collect_y.append(y_chunk)
            
        full_X_val=np.concatenate(chunk_collect_X,axis=0)
        full_y_val=np.concatenate(chunk_collect_y) 

    full_X[:,np.all(np.isnan(full_X), axis=0)]=-1

    if configs["10percent_sample"]:
        full_X=full_X[:int(0.1*len(full_X))]
        full_X_val=full_X_val[:int(0.1*len(full_X_val))]
        full_y=full_y[:int(0.1*len(full_y))]
        full_y_val=full_y_val[:int(0.1*len(full_y_val))]
    elif configs["50percent_sample"]:
        full_X=full_X[:int(0.5*len(full_X))]
        full_X_val=full_X_val[:int(0.5*len(full_X_val))]
        full_y=full_y[:int(0.5*len(full_y))]
        full_y_val=full_y_val[:int(0.5*len(full_y_val))]
    elif configs["50percent_sample_train"]:
        full_X=full_X[:int(0.5*len(full_X))]
        full_y=full_y[:int(0.5*len(full_y))]
        gc.collect()
    elif configs["25percent_sample_train"]:
        full_X=full_X[:int(0.25*len(full_X))]
        full_y=full_y[:int(0.25*len(full_y))]
        gc.collect()
    elif configs["10percent_sample_train"]:
        full_X=full_X[:int(0.10*len(full_X))]
        full_y=full_y[:int(0.10*len(full_y))]
        gc.collect()
    elif configs["5percent_sample_train"]:
        full_X=full_X[:int(0.05*len(full_X))]
        full_y=full_y[:int(0.05*len(full_y))]
        gc.collect()
    elif configs["2percent_sample_train"]:
        full_X=full_X[:int(0.02*len(full_X))]
        full_y=full_y[:int(0.02*len(full_y))]
        gc.collect()
    elif configs["1percent_sample_train"]:
        full_X=full_X[:int(0.01*len(full_X))]
        full_y=full_y[:int(0.01*len(full_y))]
        gc.collect()
    elif configs["0p1percent_sample_train"]:
        full_X=full_X[:int(0.001*len(full_X))]
        full_y=full_y[:int(0.001*len(full_y))]
        gc.collect()                                        
    elif configs["25percent_sample"]:
        full_X=full_X[:int(0.25*len(full_X))]
        full_X_val=full_X_val[:int(0.25*len(full_X_val))]
        full_y=full_y[:int(0.25*len(full_y))]
        full_y_val=full_y_val[:int(0.25*len(full_y_val))]
    elif configs["20percent_sample"]:
        full_X=full_X[:int(0.2*len(full_X))]
        full_X_val=full_X_val[:int(0.2*len(full_X_val))]
        full_y=full_y[:int(0.2*len(full_y))]
        full_y_val=full_y_val[:int(0.2*len(full_y_val))]
    elif configs["5percent_sample"]:
        full_X=full_X[:int(0.05*len(full_X))]
        full_X_val=full_X_val[:int(0.05*len(full_X_val))]
        full_y=full_y[:int(0.05*len(full_y))]
        full_y_val=full_y_val[:int(0.05*len(full_y_val))]
    elif configs["1percent_sample"]:
        full_X=full_X[:int(0.01*len(full_X))]
        full_X_val=full_X_val[:int(0.01*len(full_X_val))]
        full_y=full_y[:int(0.01*len(full_y))]
        full_y_val=full_y_val[:int(0.01*len(full_y_val))]
    elif configs["verysmall_sample"]:
        full_X=full_X[:int(0.001*len(full_X))]
        full_X_val=full_X_val[:int(0.001*len(full_X_val))]
        full_y=full_y[:int(0.001*len(full_y))]
        full_y_val=full_y_val[:int(0.001*len(full_y_val))]    

    impute_model=None
    encode_scale_model=None
    var_remove_model=None
    
    if configs["filter_low_variance"]:
        var_remove_model=skfselect.VarianceThreshold()
        full_X=var_remove_model.fit_transform(full_X)
        full_X_val=var_remove_model.transform(full_X_val)
        keep_feat_support=var_remove_model.get_support(indices=True)
        X_col_names=[X_col_names[idx] for idx in keep_feat_support]

    special_cat_cols=["APACHEPatGroup","Sex","APACHECode","Emergency","Surgical","PatGroup"]
    X_cat_cols=list(filter(lambda col: (extract_var(col),"Datatype") in hirid_schema_dict and
                           hirid_schema_dict[(extract_var(col),'Datatype')]=="Categorical" and
                           ("plain" in col or "mode" in col) \
                           or extract_var(col) in special_cat_cols, X_col_names))
    cat_col_idxs=list(filter(lambda col_idx: X_col_names[col_idx] in X_cat_cols, np.arange(len(X_col_names))))
    cont_col_idxs=list(filter(lambda col_idx: X_col_names[col_idx] not in X_cat_cols, np.arange(len(X_col_names))))    

    if configs["mean_impute_nans"]:
        impute_model=skimpute.SimpleImputer(strategy="mean")
        full_X=impute_model.fit_transform(full_X)
        full_X_val=impute_model.transform(full_X_val)
    if configs["scale_encode_data"]:
        encode_scale_model=skcompose.ColumnTransformer([("cat_pt",skpproc.OneHotEncoder(handle_unknown="ignore"),cat_col_idxs), ("cont_scale",skpproc.StandardScaler(),cont_col_idxs)])
        full_X=encode_scale_model.fit_transform(full_X)
        full_X_val=encode_scale_model.transform(full_X_val)
    elif configs["encode_data"]:
        encode_scale_model=skcompose.ColumnTransformer([("cat_pt",skpproc.OneHotEncoder(handle_unknown="ignore"),cat_col_idxs), ("cont_scale","passthrough",cont_col_idxs)])
        full_X=encode_scale_model.fit_transform(full_X)
        full_X_val=encode_scale_model.transform(full_X_val)

    print("Training matrix shape after filter: {}".format(full_X.shape),flush=True)
    print("Val matrix shape after filter: {}".format(full_X_val.shape),flush=True)

    mlhc_memory.print_memory_diags()

    print("Training {}...".format(model_type),flush=True)

    grid_point_cnt=0
    hp_metric_dict={}

    if configs["custom_eval_metric"]=="auroc":
        custom_eval_metric=ml_utils.custom_auroc_metric
    elif configs["custom_eval_metric"]=="auprc":
        custom_eval_metric=ml_utils.custom_auprc_metric
    elif configs["custom_eval_metric"]=="multiclass":
        custom_eval_metric=ml_utils.custom_mc_auroc_metric
    
    if configs["select_variables_forward"]:
        print("Number of total variables: {}".format(len(restrict_var_list)),flush=True)
        rem_list=copy.copy(restrict_var_list)
        sel_list=[]
        print("Number of remaining variables to be selected: {}".format(len(rem_list)),flush=True)

        while len(rem_list)>0:
            best_auprc_round=-np.inf                
            best_var_round=None
            for tidx,test_var in tqdm.tqdm(enumerate(rem_list),file=sys.stdout):
                final_list=sel_list+[test_var]
                all_idxs=np.arange(len(X_col_names))
                rev_X_col_idxs=list(filter(lambda col_idx: extract_var(X_col_names[col_idx]) in final_list,all_idxs))
                rev_X_col_names=[X_col_names[jdx] for jdx in rev_X_col_idxs]
                rev_full_X=full_X[:,rev_X_col_idxs]
                rev_full_X_val=full_X_val[:,rev_X_col_idxs]
                rev_X_cat_cols=list(filter(lambda col_name: col_name in rev_X_col_names, X_cat_cols))
                ml_model.fit(rev_full_X,full_y,feature_name=rev_X_col_names, categorical_feature=rev_X_cat_cols,
                             eval_set=[(rev_full_X_val, full_y_val)],eval_metric=custom_eval_metric,
                             early_stopping_rounds=10,verbose=False)
                val_score_dict=ml_utils.get_validation_scores(rev_full_X_val,full_y_val,X_col_names=rev_X_col_names,
                                                              X_cat_cols=rev_X_cat_cols,ml_model=ml_model,configs=configs)

                if configs["clf_objective"]=="multiclass":
                    auprc_round=val_score_dict["auroc"]
                else:
                    auprc_round=val_score_dict["auprc"]

                if auprc_round>best_auprc_round:
                    best_var_round=test_var
                    best_auprc_round=auprc_round

            sel_list.append(best_var_round)
            rem_list.remove(best_var_round)
            print("Adding best variable {}: {}, Metric: {:.3f}".format(len(sel_list),best_var_round,best_auprc_round),flush=True)

        sys.exit(0)

    if configs["ml_model"]=="lightgbm":
        hp_grid=configs["GBM_HP_GRID"]
        full_grid=list(itertools.product(hp_grid["n_estimators"], hp_grid["num_leaves"], hp_grid["learning_rate"],
                                         hp_grid["colsample_bytree"], hp_grid["rowsample_bytree"]))

    # Loop over all hyperparameter settings and try them
    for grid_point in full_grid:

        if configs["ml_model"]=="lightgbm":
            lgbm_nestimators, lgbm_numleaves, lgbm_learningrate, lgbm_colsamplebytree, lgbm_rowsamplebytree=grid_point
            hp_dict={}
            hp_dict["n_est"]=lgbm_nestimators
            hp_dict["num_leaves"]=lgbm_numleaves
            hp_dict["learning_rate"]=lgbm_learningrate
            hp_dict["colsample_bytree"]=lgbm_colsamplebytree
            hp_dict["rowsample_bytree"]=lgbm_rowsamplebytree
            print("Exploring GRID point ({},{},{},{},{})".format(lgbm_nestimators, lgbm_numleaves, lgbm_learningrate, 
                                                                 lgbm_colsamplebytree, lgbm_rowsamplebytree),flush=True) 


        print("HP setting {}/{}".format(grid_point_cnt,len(full_grid)))

        grid_point_cnt+=1

        if configs["ml_model"]=="lightgbm":
            ml_model = ml_utils.construct_ml_model("lightgbm",n_estimators=lgbm_nestimators, num_leaves=lgbm_numleaves, learning_rate=lgbm_learningrate,
                                                   colsample_bytree=lgbm_colsamplebytree, rowsample_bytree=lgbm_rowsamplebytree,
                                                   configs=configs)

        t_begin_fit=timeit.default_timer()

        if configs["ml_model"]=="lightgbm":
            ml_model.fit(full_X,full_y,feature_name=X_col_names, categorical_feature=X_cat_cols,
                         eval_set=[(full_X_val, full_y_val)],eval_metric=custom_eval_metric,
                         callbacks=[lightgbm.early_stopping(20,verbose=False)])

        t_end_fit=timeit.default_timer()
        print("Fitting time: {:.3f} seconds".format(t_end_fit-t_begin_fit))
        gc.collect()

        train_score_dict=ml_utils.get_train_scores(full_X,full_y,X_col_names=X_col_names,X_cat_cols=X_cat_cols,
                                                   ml_model=ml_model,configs=configs)
        val_score_dict=ml_utils.get_validation_scores(full_X_val,full_y_val,X_col_names=X_col_names,
                                                      X_cat_cols=X_cat_cols,ml_model=ml_model,configs=configs)

        if configs["ml_model"]=="lightgbm":
            eval_score_dict=ml_utils.get_evaluation_trace(ml_model=ml_model, configs=configs)

        # Hyperparameter optimization mode
        if configs["custom_eval_metric"]=="auprc":
            hp_metric_dict[json.dumps(hp_dict)]=float(val_score_dict["auprc"])
        elif configs["custom_eval_metric"]=="auroc":
            hp_metric_dict[json.dumps(hp_dict)]=float(val_score_dict["auroc"])
        elif configs["custom_eval_metric"]=="multiclass":
            hp_metric_dict[json.dumps(hp_dict)]=float(val_score_dict["auroc"])

        if configs["ml_model"]=="lightgbm":
            config_string="n-est_{}_num-leaves_{}_learning-rate_{}_colsample-bytree_{}_rowsample_{}".format(lgbm_nestimators,
                                                                                                            lgbm_numleaves,
                                                                                                            lgbm_learningrate,
                                                                                                            lgbm_colsamplebytree,
                                                                                                            lgbm_rowsamplebytree)
        train_fname_tsv="trainscore_{}.tsv".format(config_string)
        val_fname_tsv="valscore_{}.tsv".format(config_string)
        eval_fname_tsv="trace_{}.tsv".format(config_string)
        dump_fname="model_{}.pickle".format(config_string)

        # Save diagnostic information about the model
        if not configs["debug_mode"]:
            with open(os.path.join(output_dir,train_fname_tsv),'w') as fp:
                csv_fp=csv.writer(fp,delimiter='\t')

                if configs["clf_objective"]=="binary":
                    csv_fp.writerow(["auroc", "auprc"])
                    csv_fp.writerow([train_score_dict["auroc"], train_score_dict["auprc"]])
                elif configs["clf_objective"]=="multiclass":
                    csv_fp.writerow(["acc", "auroc"])
                    csv_fp.writerow([train_score_dict["acc"], train_score_dict["auroc"]])

            with open(os.path.join(output_dir,val_fname_tsv),'w') as fp:
                csv_fp=csv.writer(fp,delimiter='\t')

                if configs["clf_objective"]=="binary":
                    csv_fp.writerow(["auroc", "auprc"])
                    csv_fp.writerow([val_score_dict["auroc"], val_score_dict["auprc"]])
                elif configs["clf_objective"]=="multiclass":
                    csv_fp.writerow(["acc", "auroc"])
                    csv_fp.writerow([val_score_dict["acc"], val_score_dict["auroc"]])

            with open(os.path.join(output_dir,dump_fname), 'wb') as fp:
                pickle.dump(ml_model, fp)
            if configs["ml_model"]=="lightgbm":
                with open(os.path.join(output_dir,eval_fname_tsv),'w') as fp:
                    csv_fp=csv.writer(fp,delimiter='\t')
                    csv_fp.writerow(["epoch", "val_metric"])
                    for idx,val_auprc in enumerate(eval_score_dict):
                        csv_fp.writerow([str(idx+1), str(val_auprc)])

        if configs["debug_mode"]:
            print("Only explore one hyperparameter")
            break

    best_hps=json.loads(max(hp_metric_dict, key=hp_metric_dict.get))

    if configs["ml_model"]=="lightgbm":
        nestimators_star=best_hps["n_est"]
        numleaves_star=best_hps["num_leaves"]
        learningrate_star=best_hps["learning_rate"]
        colsample_star=best_hps["colsample_bytree"]
        rowsample_star=best_hps["rowsample_bytree"]
        
    if configs["ml_model"]=="lightgbm":
        print("Best HPs: ({},{},{},{},{})".format(nestimators_star, numleaves_star, learningrate_star, colsample_star, rowsample_star))        
        ml_model = ml_utils.construct_ml_model("lightgbm",n_estimators=nestimators_star, num_leaves=numleaves_star, learning_rate=learningrate_star,
                                               colsample_bytree=colsample_star, rowsample_bytree=rowsample_star,configs=configs)
        
    if configs["save_ml_inputs"]:
        ml_input_logging={}
        ml_input_logging["X_train"]=full_X
        ml_input_logging["X_val"]=full_X_val
        ml_input_logging["y_train"]=full_y
        ml_input_logging["y_val"]=full_y_val
        ml_input_logging["fnames"]=np.array(X_col_names)
        cum_X_test=[]
        cum_y_test=[]
        cum_pid_test=[]
        cum_abs_ts_test=[]        

    # Refit the model with the best hyperparameters
    if configs["ml_model"] in ["lightgbm"]:
        ml_model.fit(full_X,full_y,feature_name=X_col_names, categorical_feature=X_cat_cols,
                     eval_set=[(full_X_val, full_y_val)],eval_metric=custom_eval_metric,
                     callbacks=[lightgbm.early_stopping(20,verbose=False)])

    train_score_dict=ml_utils.get_train_scores(full_X,full_y,X_col_names=X_col_names,X_cat_cols=X_cat_cols,
                                               ml_model=ml_model,configs=configs)
    val_score_dict=ml_utils.get_validation_scores(full_X_val,full_y_val,X_col_names=X_col_names,
                                                  X_cat_cols=X_cat_cols,ml_model=ml_model,configs=configs)

    if configs["ml_model"] in ["lightgbm"]:
        eval_score_dict=ml_utils.get_evaluation_trace(ml_model=ml_model, configs=configs)

    train_fname_tsv="best_model_trainscore.tsv"
    val_fname_tsv="best_model_valscore.tsv"
    eval_fname_tsv="best_model_trace.tsv"

    # Save diagnostic information about the model
    if not configs["debug_mode"]:
        with open(os.path.join(output_dir,train_fname_tsv),'w') as fp:
            csv_fp=csv.writer(fp,delimiter='\t')

            if configs["clf_objective"]=="binary":
                csv_fp.writerow(["auroc", "auprc"])
                csv_fp.writerow([train_score_dict["auroc"], train_score_dict["auprc"]])
            elif configs["clf_objective"]=="multiclass":
                csv_fp.writerow(["acc", "auroc"])
                csv_fp.writerow([train_score_dict["acc"], train_score_dict["auroc"]])

        with open(os.path.join(output_dir,val_fname_tsv),'w') as fp:
            csv_fp=csv.writer(fp,delimiter='\t')

            if configs["clf_objective"]=="binary":
                csv_fp.writerow(["auroc", "auprc"])
                csv_fp.writerow([val_score_dict["auroc"], val_score_dict["auprc"]])
            elif configs["clf_objective"]=="multiclass":
                csv_fp.writerow(["acc", "auroc"])
                csv_fp.writerow([val_score_dict["acc"], val_score_dict["auroc"]])

        if configs["ml_model"]=="lightgbm":
            with open(os.path.join(output_dir,eval_fname_tsv),'w') as fp:
                csv_fp=csv.writer(fp,delimiter='\t')
                csv_fp.writerow(["epoch", "val_metric"])
                for idx,val_auprc in enumerate(eval_score_dict):
                    csv_fp.writerow([str(idx+1), str(val_auprc)])    
        
    if not configs["debug_mode"] and configs["plot_tree"] and configs["ml_model"] in ["lightgbm","lgbm_flaml"]:
        ml_utils.plot_tree(ml_model)
        plt.savefig(os.path.join(output_dir, "tree_diagram.pdf"),dpi=2000)

    # Save variable importance / best model to disk
    if configs["ml_model"] in ["lightgbm","lgbm_flaml"] and not configs["debug_mode"]:

        if configs["ml_model"]=="lgbm_flaml":
            underlying_model=ml_model.model._model
        else:
            underlying_model=ml_model
        
        fimps_fname_tsv="best_model_shapley_values.tsv"
        feat_imp_vect=ml_utils.feature_importances(full_X_val,full_y_val,underlying_model)
        assert(len(X_col_names)==feat_imp_vect.size)

        with open(os.path.join(output_dir,fimps_fname_tsv),'w') as fp:
            csv_fp=csv.writer(fp,delimiter='\t')
            csv_fp.writerow(["feature", "importance_score"])
            for idx in range(len(X_col_names)):
                csv_fp.writerow([X_col_names[idx], "{}".format(feat_imp_vect[idx])])

        dump_fname="best_model.pickle"

        if configs["ml_model"]=="lightgbm":
            with open(os.path.join(output_dir, dump_fname),'wb') as fp:
                pickle.dump(ml_model, fp)

    print("Testing...",flush=True)
    eval_set=list(sorted(eval_set, key=lambda elem: test_batch_map[elem]))
    
    n_skipped_patients=0

    test_skip_status_no_feat_labels=0
    test_skip_status_no_shapelets=0
    test_skip_status_no_static=0
    test_skip_status_pred_error=0

    time_after_full_fitting=timeit.default_timer()
    print("Seconds after full fitting: {:.3f}".format(time_after_full_fitting-time_after_begin))

    # Delete the output files so old patients are removed
    output_fs=glob.glob(os.path.join(output_dir,"batch_*.h5"))
    if not configs["debug_mode"]:
        for output_fpath in output_fs:
            print("Deleting existing pred file: {}".format(output_fpath))
            os.remove(output_fpath)

    df_X_buffer={}
    df_y_buffer={}
    all_static_df=pd.read_hdf(os.path.join(test_impute_dir, "static.h5"), mode='r')
    all_static_df["PatientID"]=all_static_df["PatientID"].astype("category")

    if configs["compat_thomas_relabel_static"]:
        all_static_df.rename(columns={"gender": "Sex"}, inplace=True)
            
    for idx,val_patient in enumerate(eval_set):

        if (idx+1)%100==0:
            print("Patient {}/{}: {}, SKIPPED: {}".format(idx+1,len(eval_set),val_patient,n_skipped_patients),flush=True)
            print("Skip reasons: FEAT/LABELS: {}, SHAPELETS: {}, STATIC: {}, PREDS: {}".format(test_skip_status_no_feat_labels, test_skip_status_no_shapelets,
                                                                                               test_skip_status_no_static, test_skip_status_pred_error))
            mlhc_memory.print_memory_diags()

        batch_pat=test_batch_map[val_patient]
        df_path=os.path.join(test_problem_dir,"batch_{}.h5".format(batch_pat))
        
        if not os.path.exists(df_path):
            n_skipped_patients+=1
            test_skip_status_no_feat_labels+=1
            continue

        if df_path not in df_X_buffer.keys():
            df_X_buffer={}
            df_y_buffer={}
            df_X_buffer[df_path]=pd.read_hdf(df_path,"/X",mode='r')
            df_y_buffer[df_path]=pd.read_hdf(df_path,"/y",mode='r')
            df_X_buffer[df_path]["PatientID"]=df_X_buffer[df_path]["PatientID"].astype("category")
            df_y_buffer[df_path]["PatientID"]=df_y_buffer[df_path]["PatientID"].astype("category")
            gc.collect()

        all_pat_df=df_X_buffer[df_path]
        pat_df=all_pat_df[all_pat_df.PatientID==val_patient]

        input_ts_diff=pat_df.AbsDatetime.diff().unique()
        input_ts_diff=input_ts_diff[np.isfinite(input_ts_diff)]

        all_label_df=df_y_buffer[df_path]
        pat_label_df=all_label_df[all_label_df.PatientID==val_patient]
        static_df=all_static_df[all_static_df["PatientID"]==val_patient]
        
        if static_df.shape[0]==0:
            n_skipped_patients+=1
            test_skip_status_no_static+=1
            continue

        if static_df.shape[0]>1:
            static_df=static_df.head(1)

        # Empty patient
        if pat_label_df.shape[0]==0 or pat_df.shape[0]==0:
            n_skipped_patients+=1
            test_skip_status_no_feat_labels+=1
            continue

        if configs["ext_val_mode"]=="internal":
            static_df=static_df[configs["static_cols_raw"]]
        else:
            static_df=static_df[configs["static_cols_raw_mimic"]]

        if not configs["compat_thomas_relabel_static"]:
            static_df.replace({"Sex": {"F": 1.0, "M": 0.0}},inplace=True)

        if column_desc=="mews_score":
            pred_vect=X_mat.flatten()
        else:

            if configs["compat_thomas_relabel_static"]:
                x_ref_grid=list(pat_df.AbsDatetime.unique())
                pat_label_df=pat_label_df[pat_label_df.AbsDatetime.isin(x_ref_grid)]
            
            assert(pat_df.shape[0]==pat_label_df.shape[0])

            if configs["ml_model"] in ["lightgbm","lgbm_flaml"]:
                df_pred,X_input,y_input,abs_ts_input=ml_utils.predict(pat_df, pat_label_df ,pid=val_patient, df_static=static_df, ml_model=ml_model, configs=configs, feat_order=X_col_names,
                                                                      low_var_model=var_remove_model,feat_order_orig=X_col_names_orig, return_X_y_inputs=True)
                
                # Only save data from the test set patients
                if configs["save_ml_inputs"] and val_patient in test_data_split["test"] and X_input is not None and y_input is not None and abs_ts_input is not None:
                    cum_X_test.append(X_input)
                    cum_y_test.append(y_input)
                    cum_pid_test.extend([val_patient]*X_input.shape[0])
                    cum_abs_ts_test.extend(abs_ts_input)

                if df_pred is not None:
                    debug_input_ts=list(sorted(pat_df.AbsDatetime.unique()))
                    debug_output_ts=list(sorted(df_pred.AbsDatetime.unique()))
                    diff_output_ts=df_pred.AbsDatetime.diff().unique()
                    diff_output_ts=diff_output_ts[np.isfinite(diff_output_ts)]
                    assert(debug_input_ts==debug_output_ts)
                
            else:
                df_pred=ml_utils.predict(pat_df, pat_label_df ,pid=val_patient, df_static=static_df, ml_model=ml_model, configs=configs, feat_order=X_col_names,
                                         impute_model=impute_model, low_var_model=var_remove_model,feat_order_orig=X_col_names_orig,
                                         encode_scale_model=encode_scale_model)
                
            if df_pred is None:
                n_skipped_patients+=1
                test_skip_status_pred_error+=1
                continue

            assert(df_pred.shape[0]==pat_df.shape[0])

            if not configs["debug_mode"]:
                df_pred.to_hdf(os.path.join(output_dir,"batch_{}.h5".format(batch_pat)), "/p{}".format(val_patient),
                               complevel=configs["hdf_comp_level"], complib=configs["hdf_comp_alg"])

        gc.collect()

    print("Number of skipped test patients: {}".format(n_skipped_patients),flush=True)

    if configs["save_ml_inputs"]:
        ml_input_logging["pid_train"]=cum_pid_train
        ml_input_logging["pid_val"]=cum_pid_val
        ml_input_logging["abs_ts_train"]=cum_abs_ts_train
        ml_input_logging["abs_ts_val"]=cum_abs_ts_val
        ml_input_logging["X_test"]=np.concatenate(cum_X_test,axis=0)
        ml_input_logging["y_test"]=np.concatenate(cum_y_test,axis=0)
        ml_input_logging["pid_test"]=cum_pid_test
        ml_input_logging["abs_ts_test"]=cum_abs_ts_test

        if configs["ext_val_mode"] in ["internal","validation"]:
            ml_logging_path=configs["bern_ml_input_logging"]
        elif configs["ext_val_mode"]=="retrain":
            ml_logging_path=configs["mimic_ml_input_logging"]
        
        with open(ml_logging_path,'wb') as fp:
            pickle.dump(ml_input_logging,fp)

    time_after_all=timeit.default_timer()
    print("Seconds after entire execution: {:.3f}".format(time_after_all-time_after_begin))


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

    parser.add_argument("--ml_model", default=None, help="Which model should produce the predictions?")
    
    parser.add_argument("--bern_split_key", default=None, help="Which data split to be used for HIRID-II data-set?")
    parser.add_argument("--mimic_split_key", default=None, help="Which data split to be used for MIMIC data-set?")
    
    parser.add_argument("--label_key", default=None, help="Which label function should be used for training and/or evaluation?")
    parser.add_argument("--eval_label_key", default=None, help="Evaluation label key")
    
    parser.add_argument("--column_set", default=None, help="Which feature columns should be selected from the model?")
    parser.add_argument("--debug_mode", default=None, action="store_true", help="Debugging mode that does not write to file-system")
    
    #parser.add_argument("--gin_config", default="./configs/learning.gin", help="GIN config to use")
    parser.add_argument("--gin_config", default="./configs/learning_separate.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/learning_separate_MIMIC.gin", help="GIN config to use")    

    #parser.add_argument("--gin_config", default="./configs/internal.gin", help="GIN config to use")     
    #parser.add_argument("--gin_config", default="./configs/validation.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/retrain.gin", help="GIN config to use") 
    
    #parser.add_argument("--gin_config", default="./configs/variable_selection.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/shap_pre_selection.gin", help="GIN config to use") 
    
    parser.add_argument("--run_mode", default=None, help="Cluster execution mode")

    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)
    
    run_mode=configs["run_mode"]
    
    split_key=configs["bern_split_key"]
    
    task_key=configs["label_key"]
    model_type=configs["ml_model"]

    execute(configs)

