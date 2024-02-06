
import gc
import timeit
import os.path
import sys
import os
import os.path
import datetime
import random
import gc
import psutil
import multiprocessing as mp
import concurrent.futures as conc_futures
import time
import csv
import pickle
import glob
import argparse
import ipdb
import gin

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

from akiews.utils.forward_filling import impute_forward_fill_simple

from akiews.utils.io import load_pickle
from akiews.utils.array import value_empty, empty_nan

def change_type_height_weight_col(height_col, weight_col):
    ''' Change weight/height group to a numerical encoding'''
    height_col[height_col=='160-169']=165                          
    height_col[height_col=='170-179']=175                                                                                                                                       
    height_col[height_col=='180-189']=185                                                                                                                                                                                                          
    height_col[height_col=='159-']=159                                                                                                                                                                                   
    height_col[height_col=='190+']=190                                                                                                                                                       
    height_col[height_col is None]=np.nan                                                                                                                                                                                                   
    height_new=height_col.astype(np.float)
    weight_col[weight_col=='60-69']=65
    weight_col[weight_col=='70-79']=75
    weight_col[weight_col=='80-89']=85
    weight_col[weight_col=='90-99']=95
    weight_col[weight_col=='100-109']=105
    weight_col[weight_col=='110+']=110
    weight_col[weight_col=='59-']=59
    weight_col[weight_col is None]=np.nan
    weight_new=weight_col.astype(np.float)
    return height_new,weight_new
    

def impute_dynamic_df(patient_df,pid=None,df_static=None, typical_weight_dict=None,
                      median_bmi_dict=None, configs=None, schema_dict=None,
                      interval_median_dict=None, interval_iqr_dict=None):
    ''' Transformer method, taking as input a data-frame with irregularly sampled input data. The method 
        assumes that the data-frame contains a time-stamp column, and the data-frame is sorted along the first 
        axis in non-decreasing order with respect to the timestamp column. Pass the <pid> of the patient stay
        as additional information'''
    pid_key="stay_id" if configs["endpoint"]=="kidney_extval" else "PatientID"
    gender_key="gender" if configs["endpoint"]=="kidney_extval" else "Sex"
    abs_ts_key="Datetime"
    
    static_table=df_static[df_static[pid_key]==pid]
    max_grid_length_secs=configs["max_grid_length_days"]*24*3600

    # No static data, patient is not valid, exclude on-the-fly
    if static_table.shape[0]==0:
        print("WARNING: No static data in patient table...")
        return None

    # More than one row, select one of the rows arbitrarily
    if static_table.shape[0]>1:
        print("WARNING: More than one row in static table...")
        static_table=static_table.take([0],axis=0)

    static_height=float(static_table["Height"])
    static_gender=str(static_table[gender_key].values[0]).strip()
    assert(static_gender in ["F","M","U"])

    if static_gender in ["F", "M"]:
        typical_weight=typical_weight_dict[static_gender]
    else:
        typical_weight=(typical_weight_dict["M"]+typical_weight_dict["F"])/2.0

    personal_bmi=median_bmi_dict[configs["static_key_dict"][static_gender]]

    ## If either the endpoints or the features don't exist, log the failure but do nothing, the missing patients can be
    #  latter added as a new group to the output H5
    if patient_df.shape[0]==0:
        print("WARNING: p{} has missing features, skipping output generation...".format(pid))
        return None

    all_keys=list(set(patient_df.columns.values.tolist()).difference(set(["Datetime","PatientID","stay_id", "a_temp","m_pm_1", "m_pm_2",
                                                                          "Height","Emergency","Surgical","RelDatetime"])))

    ts=patient_df[abs_ts_key]
    ts_arr=np.array(ts)
    n_ts=ts_arr.size
    hr=np.array(patient_df["vm1"])
    finite_hr=ts_arr[np.isfinite(hr) & np.isfinite(ts_arr)]

    if finite_hr.size<2:
        print("WARNING: Patient {} less than 2 HR, ignoring patient...".format(pid))
        return None

    # Define a grid anchored at the first and last HR measurement
    if not configs["extended_grid"]:
        ts_min=ts_arr[np.isfinite(hr) & np.isfinite(ts_arr)][0]
        ts_max=ts_arr[np.isfinite(hr) & np.isfinite(ts_arr)][-1]

    # Extended grid, given by the first and last urine/creatinine measurement
    else:
        creat=np.array(patient_df["vm156"])
        urine=np.array(patient_df["vm24"])
        finite_creat=ts_arr[np.isfinite(creat)]
        finite_urine=ts_arr[np.isfinite(urine)]        
        ts_min=ts_arr[np.isfinite(hr)][0]
        ts_max=ts_arr[np.isfinite(hr)][-1]
        if finite_creat.size>0:
            ts_min=min(ts_min,ts_arr[np.isfinite(creat)][0])
            ts_max=max(ts_max,ts_arr[np.isfinite(creat)][-1])
        if finite_urine.size>0:
            ts_min=min(ts_min,ts_arr[np.isfinite(urine)][0])
            ts_max=max(ts_max,ts_arr[np.isfinite(urine)][-1])

    max_ts_diff=(ts_max-ts_min)/np.timedelta64(1,'s')

    try:
        time_grid=np.arange(0.0,min(max_ts_diff+1.0,max_grid_length_secs),configs["grid_period"])
    except ValueError:
        ipdb.set_trace()
        print("FOO")
        
    time_grid_abs=[ts_min+pd.Timedelta(seconds=time_grid[idx]) for idx in range(time_grid.size)]
        
    imputed_df_dict={}
    imputed_df_dict[configs["patient_id_key"]]=[int(pid)]*time_grid.size
    imputed_df_dict[configs["rel_datetime_key"]]=time_grid
    imputed_df_dict[configs["abs_datetime_key"]]=time_grid_abs

    ## There is nothing to do if the patient has no records, just return...
    if n_ts==0:
        print("WARNING: p{} has an empty record, skipping output generation...".format(patient))
        return None

    ## Initialize the storage for the imputed time grid, NANs for the non-pharma, 0 for pharma.
    for col in all_keys:
        if col[:2]=="pm":
            imputed_df_dict[col]=np.zeros(time_grid.size)
        elif col[:2]=="vm":
            imputed_df_dict[col]=empty_nan(time_grid.size)
        else:
            print("ERROR: Invalid variable type")
            assert(False)

    imputed_df=pd.DataFrame(imputed_df_dict)
    norm_ts=np.array(ts-ts_min)/np.timedelta64(1,'s')

    all_keys.remove("vm131")
    all_keys=["vm131"]+all_keys

    ## Impute all variables independently, with the two relevant cases pharma variable and other variable,
    #  distinguishable from the variable prefix. We enforce that weight is the first variable to be imputed, so that 
    #  its time-gridded information can later be used by other custom formulae imputations that depend on it.
    for var_idx,variable in enumerate(all_keys):
        df_var=patient_df[variable]
        assert(n_ts==df_var.shape[0]==norm_ts.size)
        valid_normal=False
        var_encoding=schema_dict[(variable,"Datatype")]

        if not var_encoding in ["Binary","Ordinal","Categorical"]:
            assert(False)

        # Saved a value in the dict of normal values
        if (variable,"Normal value") in schema_dict:
            saved_normal_var=schema_dict[(variable,"Normal value")]
            try:
                saved_normal_var=float(schema_dict[(variable,"Normal value")])
                if np.isfinite(saved_normal_var):
                    global_impute_val=saved_normal_var
                    valid_normal=True
                else:
                    valid_normal=False
            except:
                valid_normal=False

        if not valid_normal:

            # Fill in the weight using BMI calculations
            if variable in ["vm131"]:

                # If we have an observed height can use BMI
                if np.isfinite(static_height):
                    global_impute_val=personal_bmi*(static_height/100)**2
                else:
                    global_impute_val=typical_weight

            elif variable in ["vm13","vm24","vm31","vm32"]:
                global_impute_val=np.nan
            else:
                assert(False)

        # Set non-observed value to NAN (special mode for the Kidney project)
        if configs["impute_normal_value_as_nan"]:
            global_impute_val=0 if variable[:2]=="pm" and configs["zero_impute_pharma"] else np.nan

        raw_col=np.array(df_var)
        assert(raw_col.size==norm_ts.size)

        # For binary pharma variables we remove 0 observations, because they are irrelevant
        if var_encoding=="Binary" and "pm" in variable and configs["remove_redundant_zeros"]:
            observ_idx=np.isfinite(raw_col) & (raw_col!=0) & np.isfinite(norm_ts)
            observ_ts=norm_ts[observ_idx]
            observ_val=raw_col[observ_idx]

        # Only use finite observations otherwise
        else:
            observ_idx=np.isfinite(raw_col) & np.isfinite(norm_ts)
            observ_ts=norm_ts[observ_idx]
            observ_val=raw_col[observ_idx]

        ## No values have been observed for this variable, it has to be imputed using the normal value. 
        if observ_val.size==0 and variable not in ["vm13","vm24","vm31","vm32"]:
            est_vals=value_empty(time_grid.size,global_impute_val)
            imputed_df[variable]=est_vals
            imputed_df["{}_IMPUTED_STATUS_CUM_COUNT".format(variable)]=np.zeros(time_grid.size)
            imputed_df["{}_IMPUTED_STATUS_TIME_TO".format(variable)]=value_empty(time_grid.size,-1.0)
            continue

        assert(np.isfinite(observ_val).all())
        assert(np.isfinite(observ_ts).all())

        # Get the correct imputation mode for this variable
        imp_mode=schema_dict[(variable,"Impute semantics")]
        assert(imp_mode in ["Forward fill indefinite","Forward fill limited","Attribute one grid point","Forward fill manual"])

        # Determine the forward fill length
        if imp_mode=="Forward fill limited":
            imp_limited_interval=schema_dict[(variable,"Max FFILL [hours]")]
            if imp_limited_interval=="Data adaptive":
                med_interval=interval_median_dict[variable]
                iqr_interval=interval_iqr_dict[variable]
                base_val=2*med_interval+iqr_interval
                fill_interval_secs=max(configs["grid_period"], base_val)
            else:
                try:
                    fill_interval_secs=float(imp_limited_interval)*3600
                except:
                    assert(False)

        elif imp_mode=="Forward fill manual":
            try:
                imp_limited_interval=schema_dict[(variable,"Max FFILL [hours]")]                
                fill_interval_secs=float(imp_limited_interval)*3600
            except:
                assert(False)
                    
        elif imp_mode=="Attribute one grid point":
            fill_interval_secs=300.0

        elif imp_mode=="Forward fill indefinite":
            fill_interval_secs=np.inf

        if configs["force_infinite_filling"]:
            fill_interval_secs=np.inf

        if "pm" in variable:
            assert(global_impute_val==0.0 or configs["impute_normal_value_as_nan"])

        # Custom formulae imputation
        if variable in ["vm13","vm24","vm31","vm32"] and configs["custom_formula_imputation"]:
            existing_weight_col=np.array(imputed_df["vm131"])
            est_vals,cum_count_ts,time_to_last_ms=impute_forward_fill_simple(observ_ts,observ_val,time_grid, global_impute_val, configs["grid_period"],
                                                                                               fill_interval_secs= fill_interval_secs, variable_id=variable, var_type=var_encoding,
                                                                                               weight_imputed_col=existing_weight_col, static_height=static_height,personal_bmi=personal_bmi)

        # Generic forward filling imputation with a suitable horizon
        else:
            est_vals,cum_count_ts,time_to_last_ms=impute_forward_fill_simple(observ_ts,observ_val,time_grid, global_impute_val, configs["grid_period"], var_type=var_encoding,
                                                                                               fill_interval_secs= fill_interval_secs,variable_id=variable, custom_formula=configs["custom_formula_imputation"])

        if not configs["impute_normal_value_as_nan"]:
            assert(np.isfinite(est_vals).all())
            
        imputed_df[variable]=est_vals
        imputed_df["{}_IMPUTED_STATUS_CUM_COUNT".format(variable)]=cum_count_ts
        imputed_df["{}_IMPUTED_STATUS_TIME_TO".format(variable)]=time_to_last_ms

    return (imputed_df,patient_df)

def is_df_sorted(df, colname):
    return (np.array(df[colname].diff().dropna(),dtype=np.float64) >=0).all()

def execute(configs):
    ''' Batch wrapper that loops through the patients of one the 50 batches'''
    batch_idx=configs["batch_idx"]
    split_key=configs["split_key"]
    data_split=load_pickle(configs["temporal_data_split_binary"])[split_key]
    all_pids=data_split["train"]+data_split["val"]+data_split["test"]
    batch_map=load_pickle(configs["pid_batch_map_binary"])["chunk_to_pids"]
    merged_reduced_base_path=configs["reduced_merged_path"]
    output_reduced_base_path=configs["imputed_reduced_dir"]
    pid_list=batch_map[batch_idx]
    selected_pids=list(set(pid_list).intersection(set(all_pids)))
    n_skipped_patients=0
    no_patient_output=0
    first_write=True
    hirid_schema_dict=load_pickle(configs["hirid_v8_dict"])

    df_static=pd.read_hdf(configs["static_info_path"],mode='r')
    if configs["endpoint"]=="kidney_extval":
        df_static.reset_index(inplace=True)
        
    typical_weight_dict=np.load(configs["typical_weight_dict_path"],allow_pickle=True).item()
    median_bmi_dict=np.load(configs["median_bmi_dict_path"],allow_pickle=True).item()
    impute_params_dir=configs["imputation_param_dict_reduced"]
    interval_median_dict=load_pickle(os.path.join(impute_params_dir,"interval_median_point_est.pickle"))
    interval_iqr_dict=load_pickle(os.path.join(impute_params_dir,"interval_iqr_point_est.pickle"))
    first_write=True

    if configs["endpoint"]=="kidney_extval":
        cand_files=glob.glob(os.path.join(merged_reduced_base_path,"batch_{}.h5".format(batch_idx)))        
    else:
        cand_files=glob.glob(os.path.join(merged_reduced_base_path,"reduced_fmat_{}_*.h5".format(batch_idx)))
        
    assert(len(cand_files)==1)
    source_fpath=cand_files[0]

    print("Number of patient IDs: {}".format(len(selected_pids),flush=True))
    datetime_key="Datetime"
    
    for pidx,pid in enumerate(selected_pids):

        if configs["endpoint"]=="kidney_extval":
            patient_df=pd.read_hdf(source_fpath,where="stay_id={}".format(pid))
            patient_df["Datetime"]=patient_df["intime"]+pd.to_timedelta(patient_df['RelDatetime'],unit='s')
            patient_df.drop(columns=["intime","Surgical","Emergency","hadm_id"],inplace=True)
        else:
            patient_df=pd.read_hdf(source_fpath,where="PatientID={}".format(pid))
        
        if patient_df.shape[0]==0:
            n_skipped_patients+=1
            
        if not is_df_sorted(patient_df,datetime_key):
            patient_df=patient_df.sort_values(by=datetime_key, kind="mergesort")

        ret_impute_dynamic_df=impute_dynamic_df(patient_df,pid=pid,df_static=df_static,typical_weight_dict=typical_weight_dict,
                                                median_bmi_dict=median_bmi_dict,configs=configs,schema_dict=hirid_schema_dict,
                                                interval_median_dict=interval_median_dict,interval_iqr_dict=interval_iqr_dict)

        # No data could be output for this patient...
        if ret_impute_dynamic_df is None:
            no_patient_output+=1
            continue
        
        imputed_df,tf_merged_df=ret_impute_dynamic_df

        if first_write:
            append_mode=False
            open_mode='w'
        else:
            append_mode=True
            open_mode='a'

        output_dir=os.path.join(output_reduced_base_path,split_key)

        if not configs["debug_mode"]:
            imputed_df.to_hdf(os.path.join(output_dir,"batch_{}.h5".format(batch_idx)),
                              configs["imputed_dset_id"],complevel=configs["hdf_comp_level"],complib=configs["hdf_comp_alg"],
                              format="table", append=append_mode, mode=open_mode,data_columns=["PatientID"])

        gc.collect()        
        first_write=False

        if (pidx+1)%10==0:
            print("Thread {}: {:.2f} %".format(batch_idx,(pidx+1)/len(selected_pids)*100),flush=True)
            print("Number of skipped patients: {}".format(n_skipped_patients))
            print("Number of no patients output: {}".format(no_patient_output))

    return 0

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
    
    parser.add_argument("--batch_idx", type=int, default=None, help="On which batch should imputation be run?")
    parser.add_argument("--split_key", default=None, help="On which split should imputation be run?")
    parser.add_argument("--run_mode", default=None, help="Execution mode")
    
    #parser.add_argument("--gin_config", default="./configs/impute_dynamic.gin", help="GIN config file to use")
    parser.add_argument("--gin_config", default="./configs/impute_dynamic_mimic.gin", help="GIN config file to use")   
    
    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    assert(configs["run_mode"] in ["CLUSTER", "INTERACTIVE"])
    split_key=configs["split_key"]
    batch_idx=configs["batch_idx"]

    execute(configs)

    print("COMPLETED SUCCESSFULLY...")




