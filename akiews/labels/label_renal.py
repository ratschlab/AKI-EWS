
import sys
import os
import os.path
import datetime
import timeit
import random
import gc
import psutil
import csv
import timeit
import time
import argparse
import glob
import ipdb

import pandas as pd
import numpy as np
import scipy as sp

import mlhc_data_manager.util.filesystem as mlhc_fs
import mlhc_data_manager.util.io as mlhc_io

import label_renal_util as bern_labels


def label_df(df_pat,input_endpoints,pid=None, configs=None):
    abs_time_col=df_pat[configs["abs_datetime_key"]]
    rel_time_col=df_pat[configs["rel_datetime_key"]]
    patient_col=df_pat[configs["patient_id_key"]]

    if df_pat.shape[0]==0:
        print("WARNING: Patient {} has no impute data, skipping...".format(pid), flush=True)
        return None

    output_df_dict={}
    output_df_dict[configs["abs_datetime_key"]]=abs_time_col
    output_df_dict[configs["rel_datetime_key"]]=rel_time_col
    output_df_dict[configs["patient_id_key"]]=patient_col

    # Now loop over pairs of endpoints and descriptors
    for endpoint_desc, df_endpoint in input_endpoints:

        df_endpoint.set_index(keys="AbsDatetime", inplace=True, verify_integrity=True)
        try:
            df_endpoint=df_endpoint.reindex(index=df_pat.AbsDatetime,method="nearest")
        except:
            print("WARNING: Issue when re-indexing frame of patient: {}".format(pid), flush=True)
            return None

        if "endpoint_status" in df_endpoint.columns.values.tolist():
            endpoint_status_arr=np.array(df_endpoint.endpoint_status)
            endpoint_status_arr[endpoint_status_arr=="unknown"]=np.nan
            endpoint_status_arr=endpoint_status_arr.astype(np.float)
            unique_status=np.unique(endpoint_status_arr)
            for status in unique_status:
                assert(np.isnan(status) or status in [0,1,2,3])

        geq1_arr=np.array(df_endpoint.geq1,dtype=np.float)
        geq2_arr=np.array(df_endpoint.geq2,dtype=np.float)
        geq3_arr=np.array(df_endpoint.geq3,dtype=np.float)

        # Urine state base arrays
        urine_1_arr=np.array(df_endpoint["1.u"],dtype=np.float)
        urine_2_arr=np.array(df_endpoint["2.u"],dtype=np.float)
        urine_3_arr=np.array(df_endpoint["3.u"],dtype=np.float)
        urine_3a_arr=np.array(df_endpoint["3.au"],dtype=np.float)

        # Creatinine state base arrays
        creat_1_arr=np.array(df_endpoint["1.i"],dtype=np.float)
        creat_1b_arr=np.array(df_endpoint["1.b"],dtype=np.float)
        creat_2_arr=np.array(df_endpoint["2.b"],dtype=np.float)
        creat_3_arr=np.array(df_endpoint["3.i"],dtype=np.float)
        creat_3b_arr=np.array(df_endpoint["3.b"],dtype=np.float)

        # GEQ1 array for urine
        urine_geq1_arr=np.zeros_like(urine_1_arr)
        urine_geq1_arr[np.isnan(urine_1_arr) | np.isnan(urine_2_arr) | np.isnan(urine_3_arr) | np.isnan(urine_3a_arr)]=np.nan
        urine_geq1_arr[(urine_1_arr==1) | (urine_2_arr==1) | (urine_3_arr==1) | (urine_3a_arr==1)]=1.0

        # GEQ1 array for creatinine
        creat_geq1_arr=np.zeros_like(creat_1_arr)
        creat_geq1_arr[np.isnan(creat_1_arr) | np.isnan(creat_1b_arr) | np.isnan(creat_2_arr) | np.isnan(creat_3_arr) | np.isnan(creat_3b_arr)]=np.nan
        creat_geq1_arr[(creat_1_arr==1) | (creat_1b_arr==1) | (creat_2_arr==1) | (creat_3_arr==1) | (creat_3b_arr==1) ]=1.0

        # GEQ2 array for urine
        urine_geq2_arr=np.zeros_like(urine_2_arr)
        urine_geq2_arr[np.isnan(urine_1_arr) | np.isnan(urine_2_arr) | np.isnan(urine_3_arr) | np.isnan(urine_3a_arr)]=np.nan
        urine_geq2_arr[(urine_2_arr==1) | (urine_3_arr==1) | (urine_3a_arr==1)]=1.0

        # GEQ2 array for creatinine
        creat_geq2_arr=np.zeros_like(creat_2_arr)
        creat_geq2_arr[np.isnan(creat_1_arr) | np.isnan(creat_1b_arr) | np.isnan(creat_2_arr) | np.isnan(creat_3_arr) | np.isnan(creat_3b_arr)]=np.nan
        creat_geq2_arr[(creat_2_arr==1) | (creat_3_arr==1) | (creat_3b_arr==1) ]=1.0

        # GEQ3 array for urine
        urine_geq3_arr=np.zeros_like(urine_2_arr)
        urine_geq3_arr[np.isnan(urine_1_arr) | np.isnan(urine_2_arr) | np.isnan(urine_3_arr) | np.isnan(urine_3a_arr)]=np.nan
        urine_geq3_arr[(urine_3_arr==1) | (urine_3a_arr==1)]=1.0

        # GEQ3 aray for creatinine
        creat_geq3_arr=np.zeros_like(creat_2_arr)
        creat_geq3_arr[np.isnan(creat_1_arr) | np.isnan(creat_1b_arr) | np.isnan(creat_2_arr) | np.isnan(creat_3_arr) | np.isnan(creat_3b_arr)]=np.nan
        creat_geq3_arr[(creat_3_arr==1) | (creat_3b_arr==1)]=1.0

        if configs["endpoint"]=="renal" and configs["load_extra_drug_cols"]:
            
            # Drug one array
            kidney_drug1_arr=np.array(df_endpoint["pm74"],dtype=np.float)

            # Drug two array
            kidney_drug2_arr=np.array(df_endpoint["pm75"],dtype=np.float)

            # Drug three array
            kidney_drug3_arr=np.array(df_endpoint["pm88"],dtype=np.float)

            # Drug four array
            kidney_drug4_arr=np.array(df_endpoint["pm91"],dtype=np.float)

            # Any of the kidney-harming drugs
            kidney_alldrugs_arr=np.array(df_endpoint["ep_drug_combined"],dtype=np.float)

        for (lhours,rhours) in configs["pred_horizons"]:

            if configs["endpoint"]=="renal" and configs["load_extra_drug_cols"]:
                output_df_dict["{}_DrugAntimycoticStop{}To{}Hours".format(endpoint_desc,lhours,rhours)]=bern_labels.drug_stop(kidney_drug1_arr, lhours=lhours, rhours=rhours)
                output_df_dict["{}_DrugAntiviralStop{}To{}Hours".format(endpoint_desc,lhours,rhours)]=bern_labels.drug_stop(kidney_drug2_arr, lhours=lhours, rhours=rhours)
                output_df_dict["{}_DrugNSARStop{}To{}Hours".format(endpoint_desc,lhours,rhours)]=bern_labels.drug_stop(kidney_drug3_arr, lhours=lhours, rhours=rhours)
                output_df_dict["{}_DrugSteroidsStop{}To{}Hours".format(endpoint_desc,lhours,rhours)]=bern_labels.drug_stop(kidney_drug4_arr, lhours=lhours, rhours=rhours)
                output_df_dict["{}_DrugKidneyHarmingStop{}To{}Hours".format(endpoint_desc,lhours,rhours)]=bern_labels.drug_stop(kidney_alldrugs_arr, lhours=lhours, rhours=rhours)

            # Tasks fixed at particular time-points (0->1,2,3), we use samples up to 48 hours from admission as separate bin, everything
            # else thereafter goes into 1 bin.
            for at_hour in list(range(0,48,1)):
                output_df_dict["{}_WorseStateFromZero{}To{}Hours_At{}Hours".format(endpoint_desc,lhours,rhours,at_hour)] \
                    =bern_labels.future_worse_state_from_0_FIXED(geq1_arr,lhours,rhours,configs["grid_step_seconds"],at_hour,onwards=False)

            # Use one bin for all hours onwards from 48 hours
            output_df_dict["{}_WorseStateFromZero{}To{}Hours_From48Hours".format(endpoint_desc,lhours,rhours)] \
                =bern_labels.future_worse_state_from_0_FIXED(geq1_arr,lhours,rhours,configs["grid_step_seconds"],48,onwards=True)        

            # Urine tasks fixed at particular time-points (0->1,2,3)
            for at_hour in list(range(4,49,1)):
                output_df_dict["{}_WorseStateUrineFromZero{}To{}Hours_At{}Hours".format(endpoint_desc,lhours,rhours,at_hour)] \
                    =bern_labels.future_worse_state_urine_from_0_FIXED(urine_geq1_arr, lhours, rhours, configs["grid_step_seconds"],at_hour)

            # Creatinine tasks fixed at particular time-points (0->1,2,3)
            for at_hour in list(range(4,49,1)):
                output_df_dict["{}_WorseStateCreatFromZero{}To{}Hours_At{}Hours".format(endpoint_desc,lhours,rhours,at_hour)]\
                    =bern_labels.future_worse_state_creat_from_0_FIXED(creat_geq1_arr, lhours, rhours, configs["grid_step_seconds"],at_hour)

            # Tasks using data before particular time-points (0->1,2,3)
            from_0_arr_before_24=bern_labels.future_worse_state_from_0_BEFORE(geq1_arr, lhours, rhours, configs["grid_step_seconds"],24)
            output_df_dict["{}_WorseStateFromZero{}To{}Hours_Before24Hours".format(endpoint_desc,lhours,rhours)]=from_0_arr_before_24

            # Overall arrays (0->1,2,3), (0,1->2,3), (0,1,2->3) (joint model)

            if configs["endpoint"]=="renal" and configs["load_extra_drug_cols"]:
                from_0_TRAIN_arr=bern_labels.future_worse_state_from_0_TRAIN(geq1_arr,kidney_alldrugs_arr,lhours,rhours,configs["grid_step_seconds"]) # From state 0 (training side label)
                from_1_TRAIN_arr=bern_labels.future_worse_state_from_1_TRAIN(geq1_arr,geq2_arr,kidney_alldrugs_arr, lhours, rhours, configs["grid_step_seconds"]) # From state 1 (training side label)
                from_2_TRAIN_arr=bern_labels.future_worse_state_from_2_TRAIN(geq2_arr,geq3_arr,kidney_alldrugs_arr,lhours, rhours, configs["grid_step_seconds"]) # From state 2 (training side label)                
                
            from_0_EVAL_arr=bern_labels.future_worse_state_from_0_EVAL(geq1_arr,lhours, rhours, configs["grid_step_seconds"]) # From state 0 (eval side label)
            from_1_EVAL_arr=bern_labels.future_worse_state_from_1_EVAL(geq1_arr,geq2_arr, lhours, rhours, configs["grid_step_seconds"]) # From state 1 (eval side label)
            from_2_EVAL_arr=bern_labels.future_worse_state_from_2_EVAL(geq2_arr,geq3_arr, lhours, rhours, configs["grid_step_seconds"]) # From state 2 (eval side label)

            # Separate tasks for urine/creatinine (joint model)
            from_0_urine_arr=bern_labels.future_worse_state_urine_from_0(urine_geq1_arr,lhours, rhours, configs["grid_step_seconds"]) 
            from_0_creat_arr=bern_labels.future_worse_state_creat_from_0(creat_geq1_arr,lhours, rhours, configs["grid_step_seconds"]) 
            from_1_urine_arr=bern_labels.future_worse_state_urine_from_1(urine_geq1_arr,urine_geq2_arr,lhours, rhours, configs["grid_step_seconds"]) 
            from_1_creat_arr=bern_labels.future_worse_state_creat_from_1(creat_geq1_arr,creat_geq2_arr,lhours, rhours, configs["grid_step_seconds"])
            from_2_urine_arr=bern_labels.future_worse_state_urine_from_2(urine_geq2_arr,urine_geq3_arr,lhours, rhours, configs["grid_step_seconds"])
            from_2_creat_arr=bern_labels.future_worse_state_creat_from_2(creat_geq2_arr,creat_geq3_arr,lhours, rhours, configs["grid_step_seconds"])

            if configs["endpoint"]=="renal" and configs["load_extra_drug_cols"]:
                output_df_dict["{}_WorseStateFromZeroTRAIN{}To{}Hours".format(endpoint_desc,lhours, rhours)]=from_0_TRAIN_arr
                output_df_dict["{}_WorseStateFromOneTRAIN{}To{}Hours".format(endpoint_desc,lhours, rhours)]=from_1_TRAIN_arr
                output_df_dict["{}_WorseStateFromTwoTRAIN{}To{}Hours".format(endpoint_desc,lhours,rhours)]=from_2_TRAIN_arr
                
            output_df_dict["{}_WorseStateFromZeroEVAL{}To{}Hours".format(endpoint_desc, lhours, rhours)]=from_0_EVAL_arr
            output_df_dict["{}_WorseStateFromOneEVAL{}To{}Hours".format(endpoint_desc,lhours, rhours)]=from_1_EVAL_arr
            output_df_dict["{}_WorseStateFromTwoEVAL{}To{}Hours".format(endpoint_desc,lhours,rhours)]=from_2_EVAL_arr

            output_df_dict["{}_WorseStateFromZeroUrine{}To{}Hours".format(endpoint_desc,lhours, rhours)]=from_0_urine_arr
            output_df_dict["{}_WorseStateFromOneUrine{}To{}Hours".format(endpoint_desc,lhours, rhours)]=from_1_urine_arr
            output_df_dict["{}_WorseStateFromTwoUrine{}To{}Hours".format(endpoint_desc,lhours,rhours)]=from_2_urine_arr

            output_df_dict["{}_WorseStateFromZeroCreat{}To{}Hours".format(endpoint_desc,lhours, rhours)]=from_0_creat_arr
            output_df_dict["{}_WorseStateFromOneCreat{}To{}Hours".format(endpoint_desc,lhours, rhours)]=from_1_creat_arr
            output_df_dict["{}_WorseStateFromTwoCreat{}To{}Hours".format(endpoint_desc,lhours,rhours)]=from_2_creat_arr
        
    output_df=pd.DataFrame(output_df_dict)
    return output_df


def is_df_sorted(df, colname):
    return (np.array(df[colname].diff().dropna(),dtype=np.float64) >=0).all()


def label_gen_renal(configs):
    '''Creation of base labels directly defined on the imputed data / endpoints'''
    split_key=configs["split_key"]
    label_base_dir=configs["label_dir"]
    endpoint_base_dir=configs["endpoint_dir"]
    imputed_base_dir=configs["imputed_dir"]
    base_dir=os.path.join(label_base_dir,"reduced",split_key)
    
    try:
        if not configs["debug_mode"]:
            mlhc_fs.create_dir_if_not_exist(base_dir,recursive=True)
    except:
        print("WARNING: Race condition when creating directory from different jobs...")

    data_split=mlhc_io.load_pickle(configs["temporal_data_split_binary_renal"])[split_key]
    all_pids=data_split["train"]+data_split["val"]+data_split["test"]

    if configs["verbose"]:
        print("Number of patient IDs: {}".format(len(all_pids),flush=True))

    batch_map=mlhc_io.load_pickle(configs["pid_batch_map_binary"])["chunk_to_pids"]
    batch_idx=configs["batch_idx"]
    
    if not configs["debug_mode"]:
        mlhc_fs.delete_if_exist(os.path.join(base_dir,"batch_{}.h5".format(batch_idx)))

    pids_batch=batch_map[batch_idx]
    selected_pids=list(set(pids_batch).intersection(all_pids))
    n_skipped_patients=0
    first_write=True
    print("Number of selected PIDs: {}".format(len(selected_pids)),flush=True)

    for pidx,pid in enumerate(selected_pids):
        patient_path=os.path.join(imputed_base_dir, "reduced" ,split_key,"batch_{}.h5".format(batch_idx))

        input_endpoints=[]
        
        # Have to loop over the feasible endpoints...
        for endpoint_desc in configs["endpoint_versions"]:
            endpoint_path=os.path.join(endpoint_base_dir,endpoint_desc,"batch_{}.h5".format(batch_idx))

            try:
                df_endpoint=pd.read_hdf(endpoint_path,mode='r', where="PatientID={}".format(pid))
            except:
                print("WARNING: Issue while reading endpoints of patient {} in version {}".format(pid,endpoint_desc),flush=True)
                n_skipped_patients+=1
                continue            

            if configs["endpoint"]=="renal" and "leq0" not in df_endpoint.columns.values.tolist():
                df_endpoint["leq0"]=(df_endpoint["geq1"]==0).astype(np.int)
                df_endpoint["leq0"][df_endpoint["geq1"]==-1]=-1

            if df_endpoint.shape[0]==0:
                print("WARNING: Empty endpoints in patient {} for version {}".format(pid,endpoint_desc), flush=True)
                n_skipped_patients+=1
                continue

            df_endpoint.sort_values(by="AbsDatetime",kind="mergesort")

            input_endpoints.append((endpoint_desc,df_endpoint))
            
        output_dir=os.path.join(label_base_dir, "reduced", split_key)

        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        except:
            pass

        if not os.path.exists(patient_path):
            print("WARNING: Patient {} does not exists, skipping...".format(pid),flush=True)
            n_skipped_patients+=1
            continue

        df_pat=pd.read_hdf(patient_path,mode='r',where="PatientID={}".format(pid))

        if df_pat.shape[0]==0:
            print("WARNING: Empty imputed data in patient {}".format(pid), flush=True)
            n_skipped_patients+=1
            continue

        df_label=label_df(df_pat,input_endpoints,pid=pid,configs=configs)

        if df_label is None:
            print("WARNING: Label could not be created for PID: {}".format(pid),flush=True)
            n_skipped_patients+=1
            continue

        assert(df_label.shape[0]==df_pat.shape[0])
        output_path=os.path.join(output_dir,"batch_{}.h5".format(batch_idx))

        if first_write:
            open_mode='w'
        else:
            open_mode='a'

        if not configs["debug_mode"]:
            df_label.to_hdf(output_path,"/labels_{}".format(pid),complevel=configs["hdf_comp_level"],complib=configs["hdf_comp_alg"], format="fixed",
                            append=False, mode=open_mode)

        gc.collect()
        first_write=False

        if (pidx+1)%10==0 and configs["verbose"]:
            print("Progress for batch {}: {:.2f} %".format(batch_idx, (pidx+1)/len(selected_pids)*100),flush=True)
            print("Number of skipped patients: {}".format(n_skipped_patients))
