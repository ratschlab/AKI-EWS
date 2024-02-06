''' Post-process separate model predictions and merge them 
    using isotonic regression'''

import os
import os.path
import ipdb
import glob
import argparse
import random
import sys

import numpy as np
import pandas as pd

def execute(configs):
    batch_idx=configs["batch_idx"]
    all_fs=glob.glob(os.path.join(configs["pred_path"],"reduced",configs["main_split"],"*Hours_{}_lightgbm*".format(configs["model_id"])))
    assert len(all_fs)==49
    print("Loading batch {}".format(batch_idx),flush=True)
    batch_dict={}
    for mix,model_f in enumerate(all_fs):
        print("Loading model {}/{}".format(mix+1,len(all_fs)))
        model_suffix=model_f.split("/")[-1].strip()
        model_batch_f=os.path.join(model_f,"batch_{}.h5".format(batch_idx))
        if not os.path.exists(model_batch_f):
            print("WARNING: Model batch does not exist")
            continue
        
        with pd.HDFStore(model_batch_f,'r') as h5_store:
            pat_keys=list(h5_store.keys())

        if mix==0 and configs["small_sample"]:
            random.shuffle(pat_keys)
            batch_pat_keys=pat_keys[:20]
        elif configs["small_sample"]:
            pat_keys=list(filter(lambda pat_key: pat_key in batch_pat_keys, pat_keys))

        for pid in pat_keys:
            if pid not in batch_dict:
                batch_dict[pid]={}
            batch_dict[pid][model_suffix]=pd.read_hdf(os.path.join(model_f,"batch_{}.h5".format(batch_idx)),pid)

    print("Number of patients: {}".format(len(batch_dict.keys())),flush=True)

    batch_out_frames=[]

    out_folder=os.path.join(configs["pred_path"],"reduced",configs["main_split"],
                          "Label_{}_merged_24h_deleted_4h_WorseStateFromZeroEVAL0To48Hours_combined_model_from_separate_{}_lightgbm".format(configs["dataset"],configs["model_id"]))
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    output_f=os.path.join(out_folder,"batch_{}.h5".format(batch_idx))
    if os.path.exists(output_f):
        os.remove(output_f)

    print("Writing patients out...")
    print("Output file: {}".format(output_f))
        
    for pid in batch_dict.keys():
        models=batch_dict[pid]
        all_frames=[]
        for hour_range in range(48):
            model_key="Label_{}_merged_24h_deleted_4h_WorseStateFromZero0To48Hours_At{}Hours_{}_lightgbm".format(configs["dataset"],hour_range, configs["model_id"])
            if model_key not in models.keys():
                continue
            frame=models[model_key]
            rest_frame=frame[(frame["RelDatetime"]>=hour_range*3600.) & (frame["RelDatetime"]<(hour_range+1)*3600.)]
            rest_frame["model_source"]=model_key
            all_frames.append(rest_frame)
        after_model_key="Label_{}_merged_24h_deleted_4h_WorseStateFromZero0To48Hours_From48Hours_{}_lightgbm".format(configs["dataset"],configs["model_id"])
        if after_model_key in models.keys():
            frame=models[after_model_key]
            rest_frame=frame[(frame["RelDatetime"]>=48*3600.)]
            rest_frame["model_source"]=after_model_key
            all_frames.append(rest_frame)            

        # No data exists for the patient
        if len(all_frames)==0:
            continue

        combined_frame=pd.concat(all_frames,axis=0)

        print("WRITING PATIENT...")
        combined_frame.to_hdf(output_f,"{}".format(pid),mode='a',complevel=5,complib="blosc:lz4",
                              format="fixed",index=False)

    print("JOB STATUS: COMPLETED SUCCESSFULLY",flush=True)


if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--pred_path", default="../../data/predictions",
                        help="Prediction base dir")

    parser.add_argument("--batch_map", default="../../data/exp_design/hirid2_chunking_100.pickle",
                        help="Batch map for patients")

    #parser.add_argument("--batch_map", default="../../data/exp_design/mimic_chunking_50.pickle",
    #                    help="Batch map for patients")    

    # Output paths
    parser.add_argument("--log_dir", default="/cluster/home/mhueser/log_files/icu_score_renal",
                        help="Log directory")

    # Arguments
    #parser.add_argument("--main_split", default="temporal_1", help="Main split to use")
    #parser.add_argument("--main_split", default="temporal_2", help="Main split to use")
    #parser.add_argument("--main_split", default="temporal_3", help="Main split to use")
    #parser.add_argument("--main_split", default="temporal_4", help="Main split to use")
    parser.add_argument("--main_split", default="temporal_5", help="Main split to use")

    #parser.add_argument("--main_split", default="random_1", help="Main split to use")
    #parser.add_argument("--main_split", default="random_2", help="Main split to use")
    #parser.add_argument("--main_split", default="random_3", help="Main split to use")
    #parser.add_argument("--main_split", default="random_4", help="Main split to use")
    #parser.add_argument("--main_split", default="random_5", help="Main split to use")    

    parser.add_argument("--small_sample", default=False, action="store_true", help="Small sample?")

    parser.add_argument("--dataset", default="hirid", help="Dataset on which predictions are produced")
    #parser.add_argument("--dataset", default="mimic", help="Dataset on which predictions are produced")    

    #parser.add_argument("--model_id", default="separate_model_simple_features_MALE", help="Model key to load the predictions from")
    parser.add_argument("--model_id", default="separate_model_simple_features_FEMALE", help="Model key to load the predictions from")    

    parser.add_argument("--batch_idx", default=0, type=int, help="Patient batch to transform")

    parser.add_argument("--run_mode", default="INTERACTIVE", help="Run on batch or interactive mode?")

    configs=vars(parser.parse_args())

    run_mode=configs["run_mode"]
    batch_idx=configs["batch_idx"]

    if run_mode=="BATCH":
        sys.stdout=open(os.path.join(configs["log_dir"],"POSTPROC_PRED_{}.stdout".format(batch_idx)),'w')
        sys.stderr=open(os.path.join(configs["log_dir"],"POSTPROC_PRED_{}.stderr".format(batch_idx)),'w')

    execute(configs)

