''' Calibrate the joined model together'''

import os
import os.path
import argparse
import glob
import pickle
import ipdb
import random
import timeit

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression

def execute(configs):
    with open(configs["split_desc"],'rb') as fp:
        split_desc=pickle.load(fp)
        val_pids=split_desc[configs["split_key"]]["val"]
        print("Number of validation PIDs: {}".format(len(val_pids)))

    if configs["restrict_gender"] is not None:
        df_static=pd.read_hdf(os.path.join(configs["static_df_path"],configs["split_key"],"static.h5"))
        male_pids=list(df_static[df_static.Sex=="M"].PatientID.unique())
        female_pids=list(df_static[df_static.Sex=="F"].PatientID.unique())
        if configs["restrict_gender"]=="male":
            val_pids=list(set(val_pids).intersection(set(male_pids)))
        elif configs["restrict_gender"]=="female":
            val_pids=list(set(val_pids).intersection(set(female_pids)))

    batch_fs=glob.glob(os.path.join(configs["pred_path"],configs["uncalib_key"],"batch_*.h5"))
    pid_counter=0

    all_pred_scores=[]
    all_labels=[]

    for bix,batch_f in enumerate(batch_fs):
        print("Processing batch file {}/{}".format(bix+1,len(batch_fs)))
        with pd.HDFStore(batch_f,'r') as hstore:
            all_pids=list(hstore.keys())
        numeric_pids=list(map(lambda pid_key: int(pid_key[2:]), all_pids))
        isect_pids=list(set(numeric_pids).intersection(set(val_pids)))
        if len(isect_pids)>0:

            if configs["small_sample"]:
                random.shuffle(isect_pids)
                isect_pids=isect_pids[:4]
            
            for pid in isect_pids:
                df_pid=pd.read_hdf(batch_f,"/p{}".format(pid), mode='r')
                df_valid=df_pid[df_pid["PredScore"].notnull()]
                pred_scores=np.array(df_valid["PredScore"])
                true_labels=np.array(df_valid["TrueLabel"])
                all_pred_scores.append(pred_scores)
                all_labels.append(true_labels)
                pid_counter+=1
                if pid_counter%100==0:
                    print("Processing PID: {}".format(pid_counter))

    all_pred_scores=np.concatenate(all_pred_scores)
    all_labels=np.concatenate(all_labels)

    print("Number of scores {}, Number of labels {}".format(len(all_pred_scores), len(all_labels)))
                
    print("Fitting isotonic regression model...")
    t_begin=timeit.default_timer()
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(all_pred_scores,all_labels)
    t_end=timeit.default_timer()
    print("Fitting time: {:.2f} seconds".format(t_end-t_begin))

    # Retraverse the frame to apply the fitted model
    for bix,batch_f in enumerate(batch_fs):
        print("Processing batch file {}/{}".format(bix+1,len(batch_fs)))        
        batch_suffix=batch_f.split("/")[-1].strip()
        output_f=os.path.join(configs["pred_path"],configs["calib_key"],
                              batch_suffix)
        if os.path.exists(output_f):
            os.remove(output_f)        

        with pd.HDFStore(batch_f,'r') as hstore:
            all_pids=list(hstore.keys())

        print("Number of PIDs: {}".format(len(all_pids)))
        for pid in all_pids:
            df_pid=pd.read_hdf(batch_f,pid,mode='r')
            pred_scores=np.array(df_pid["PredScore"])
            finite_scores=pred_scores[np.isfinite(pred_scores)]
            out_scores=np.zeros_like(pred_scores)
            out_scores[np.isnan(pred_scores)]=np.nan
            tf_scores=ir.transform(finite_scores)
            out_scores[np.isfinite(pred_scores)]=tf_scores
            df_pid["PredScoreCalibrated"]=out_scores
            df_pid.to_hdf(output_f,pid,mode='a',complevel=5,
                          complib="blosc:lz4", format="fixed", index=False)

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--pred_path",
                        default="/cluster/home/mhueser/git/projects/2022/kidnews_public/data/predictions/reduced/temporal_5",
                        help="Prediction path")

    parser.add_argument("--split_desc",
                        default="../../data/exp_design/temp_splits_hirid2.pickle",
                        help="Split descriptor")

    #parser.add_argument("--split_desc",
    #                    default="../../data/exp_design/random_splits_mimic.pickle",
    #                    help="Split descriptor")    

    parser.add_argument("--static_df_path", default="/cluster/work/grlab/clinical/hirid2/research/KIDNEY_RELEASE/imputed/noimpute_hirid2/reduced",
                        help="Static data-frame path")

    #parser.add_argument("--static_df_path", default="/cluster/work/grlab/clinical/hirid2/research/KIDNEY_RELEASE/imputed/noimpute_mimic/reduced",
    #                    help="Static data-frame path")    

    # Output paths

    # Arguments
    #parser.add_argument("--split_key", default="temporal_1", help="Split to use")
    #parser.add_argument("--split_key", default="temporal_2", help="Split to use")
    #parser.add_argument("--split_key", default="temporal_3", help="Split to use")
    #parser.add_argument("--split_key", default="temporal_4", help="Split to use")
    parser.add_argument("--split_key", default="temporal_5", help="Split to use")
    
    #parser.add_argument("--split_key", default="random_1", help="Split to use")
    #parser.add_argument("--split_key", default="random_2", help="Split to use")
    #parser.add_argument("--split_key", default="random_3", help="Split to use")
    #parser.add_argument("--split_key", default="random_4", help="Split to use")
    #parser.add_argument("--split_key", default="random_5", help="Split to use")    
    
    parser.add_argument("--small_sample", default=False, action="store_true", help="Small sample?")

    parser.add_argument("--restrict_gender", default=None, help="If not none, restrict calibration set to the gender")

    parser.add_argument("--uncalib_key",
                        default="Label_hirid_merged_24h_deleted_4h_WorseStateFromZeroEVAL0To48Hours_combined_model_from_separate_separate_model_simple_features_FEMALE_lightgbm",
                        help="Key of uncalibrated model")
    parser.add_argument("--calib_key",
                        default="Label_hirid_merged_24h_deleted_4h_WorseStateFromZeroEVAL0To48Hours_combined_model_from_separate_separate_model_simple_features_FEMALE_calibrated_lightgbm",
                        help="Key of calibrated model")

    configs=vars(parser.parse_args())

    execute(configs)
