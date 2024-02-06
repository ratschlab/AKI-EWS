'''Tests correctness of merged stage, correct number of patients, 
   rough correctness of the value distribution of variables and
   compare with the valeu schema.'''

import os
import os.path
import glob
import ipdb
import random
import pickle
import gc

import pandas as pd
import pytest
import numpy as np

import akiews.utils.memory as mlhc_memory

random.seed(2022)

# ---------- FIXTURES ----------------------------------------------------------------------------------------

@pytest.fixture
def configs():
    config_dict={}
    config_dict["MERGED_PATH"]="/cluster/work/grlab/clinical/hirid2/research/3_merged/v8/reduced_rm_drugoor"
    config_dict["META_TABLE"]="/cluster/work/grlab/clinical/hirid2/research/misc_derived/RESP_project/hirid_v8_schema.pickle"
    config_dict["DEBUG_MODE"]=False
    config_dict["SMALL_SAMPLE"]=True

    config_dict["CAT_SPECIAL_DOMAINS"]={"vm19": [1,5,6,7,8,9,10,11,12,13,14,15,16], # Rhythmus
                                        "vm60": [1,2,3,4,5,6,8,11,12,13,15]} # Ventilator servoi mode
    
    config_dict["SPECIAL_DOMAINS"]={"vm239": [1,2], # AIDS
                                    "vm236": [1,2], # Metastases
                                    "vm237": [1,2], # Hem malignancy
                                    "vm238": [1,2]} # Zytostatika


    config_dict["ORD_SPECIAL_DOMAINS"]={"vm203": [], # EMG index (no values)
                                        "vm221": [], # Intrinsic PEEP, Hamilton (no values)
                                        "vm222": [], # Work of breathing, Hamilton (no values)
                                        "vm226": [1,2,3,4], # Sekretmenge
                                        "vm232": [0,1,2,3,4], # Daily living
                                        "vm282": [], # Urin 1 (no values)
                                        "vm283": [], # Urin 2 (no values)
                                        "vm284": [], # OUTIleumconduit (no values)
                                        "vm247": [], # PupReaktLi (no values)
                                        "vm249": [], # PubReaktRe (no values)
                                        "vm25": [1,2,3,4,5], # GCS Antwort
                                        "vm30": [0,1,1.5,2,2.5,3,4], # TOF
                                        "vm253": [0,1,2,3,4,4.5,5], # M-Kr Arm li
                                        "vm254": [0,1,2,3,4,5], # M-Kr Arm rechts
                                        "vm255": [0,0.1,1,2,3,4,5,33], # M-Kr Bein li
                                        "vm256": [0,1,2,3,4,5], # M-Kr Bein rechts
                                        "vm257": [], # Rass ziel mi (no values)
                                        "vm258": [], # Rass ziel ma (no values)
                                        "vm259": [1,2,3,4,5], # PeriphHandLi
                                        "vm260": [1,2,3,4,5], # PeriphHandRe
                                        "vm261": [1,2,3,4,5], # PeriphFussLi
                                        "vm262": [1,2,3,4,5], # PeriphFussRe
                                        "vm263": [], # Rekap-Zeit (no values)
                                        "vm26": [1,2,3,4,5,6], # GCS Motorik
                                        "vm27": [1,2,3,4], # GCS Augenoeffnen
                                        "vm278": [], # Urin II (no values)
                                        "vm295": [], # SEF1 (no values)
                                        "vm296": [], # MF1 (no values)
                                        "vm297": [], # SEF2 (no values)
                                        "vm298": [], # MF2 (no values)
                                        "vm299": [], # BSR2 (no values)
                                        "vm199": [], # Bak/Mi/Kul-u (no values)
                                        "vm210": [], # Peak(s) (no values)
                                        "vm300": [], # BSR3 (no values)
                                        "vm301": [], # BSR4 (no values)
                                        "vm302": [], # TP1 (no values)
                                        "vm303": [], # TP2 (no values)
                                        "vm304": [], # Cerebrale O2 (no values)
                                        "vm305": [], # Region 2 O2 saturation (no values)
                                        "vm310": [], # High Flow (no values)
                                        "vm311": [], # High flow gas flow (no values)
                                        "pm108": [0,1,2,3,4,5,6,7,8,9,10]} # Antiepileptica, count of drugs
    
    return config_dict 

@pytest.fixture
def merged_files(configs):
    all_merged_fs=sorted(glob.glob(os.path.join(configs["MERGED_PATH"], "reduced_fmat*.h5")))
    assert len(all_merged_fs)==100
    return all_merged_fs

@pytest.fixture
def merged_complete_frame(merged_files, configs):
    random.shuffle(merged_files)
    if configs["SMALL_SAMPLE"]:
        merged_files=merged_files[:40]
    value_dists={}
    
    for fidx,fpath in enumerate(merged_files):
        mlhc_memory.print_memory_diags()
        print("Loading merged file: {}/{}".format(fidx+1,len(merged_files)))
        df=pd.read_hdf(fpath,mode='r')
        cols=df.columns.values.tolist()
        cols_meta=list(set(cols).difference(set(["PatientID","Datetime"])))
        for col in cols_meta:
            vals_df=np.array(df[col],dtype=np.float32)
            vals_fin=vals_df[np.isfinite(vals_df)]
            if col not in value_dists:
                value_dists[col]=[]
            value_dists[col].extend(list(vals_fin))
        if configs["DEBUG_MODE"]:
            break
        gc.collect()

    gc.collect()
    return {"frame": df, "value_dist": value_dists}

# ---------- TEST CASES -------------------------------------------------------------------------------

def test_merged(merged_complete_frame, configs):
    example_frame=merged_complete_frame["frame"]
    pid_type=example_frame.PatientID.dtype
    assert np.issubdtype(pid_type,int)
    example_frame["PatientID"]=example_frame["PatientID"].astype("category")

    with open(configs["META_TABLE"],'rb') as fp:
        hirid_meta=pickle.load(fp)

    ks=hirid_meta.keys()
    expected_vids=list(set(list(map(lambda tp: tp[0], ks))))
    print("Number of VIDs: {}".format(len(expected_vids)))

    cols=example_frame.columns.values.tolist()
    assert "PatientID" in cols
    assert "Datetime" in cols
    cols_meta=list(set(cols).difference(set(["PatientID","Datetime"])))

    # Check that merged data has exactly the expected meta-variables
    assert sorted(expected_vids)==sorted(cols_meta)

    value_dists=merged_complete_frame["value_dist"]
    
    # Check the types of the columns is correct
    for col in sorted(cols_meta):
        fin_vals=np.array(value_dists[col])
        unique_fin_vals=np.unique(fin_vals)
        dtype=hirid_meta[(col,"Datatype")]
        
        if dtype=="Binary":
            if col in configs["SPECIAL_DOMAINS"]:
                exp_domain=set(configs["SPECIAL_DOMAINS"][col])
            else:
                exp_domain=set([0,1])
            assert set(unique_fin_vals)<=exp_domain, "Problem in variable: {}".format(col)                

        elif dtype=="Categorical":
            if col in configs["CAT_SPECIAL_DOMAINS"]:
                exp_domain=set(configs["CAT_SPECIAL_DOMAINS"][col])
                assert set(unique_fin_vals)<=exp_domain, "Problem in variable: {}".format(col)
            else:
                assert len(unique_fin_vals)<=10, "Variable {} likely not categorical".format(col)

        elif dtype=="Ordinal":
            if col in configs["ORD_SPECIAL_DOMAINS"]:
                exp_domain=set(configs["ORD_SPECIAL_DOMAINS"][col])
                unique_fin_vals=set(map(lambda fl: round(fl,3), unique_fin_vals))
                assert set(unique_fin_vals)<=exp_domain, "Problem in variable: {}".format(col)
            else:
                assert len(unique_fin_vals)>10, "Variable {} likely not ordinal".format(col)

        gc.collect()

    all_pids=list(example_frame.PatientID.unique())
    for pix,pid in enumerate(all_pids):
        print("PID: {}/{}".format(pix+1,len(all_pids)))
        df_pid=example_frame[example_frame.PatientID==pid]
        assert pd.Index(df_pid["Datetime"]).is_monotonic
        
    

    

    

