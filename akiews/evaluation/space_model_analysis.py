''' Space the models according to percentiles of length into
    the stay'''

import os
import os.path
import sys
import glob
import ipdb
import argparse
import pickle
import random

import pandas as pd
import numpy as np

def execute(configs):

    with open(configs["split_desc"],'rb') as fp:
        split_desc=pickle.load(fp)
        val_pids=split_desc[configs["split_key"]]["val"]

    all_reltimes=[]
    
    batch_fs=glob.glob(os.path.join(configs["pred_path"],configs["eval_key"],"batch_*.h5"))

    if configs["small_sample"]:
        random.shuffle(batch_fs)
        batch_fs=batch_fs[:10]
    
    for bix,batch_f in enumerate(batch_fs):

        print("Exploring batch file {}/{}".format(bix+1,len(batch_fs)))

        with pd.HDFStore(batch_f,'r') as hstore:
            pat_keys=list(hstore.keys())
            search_pids=list(map(lambda pid_key: int(pid_key[2:]), pat_keys))
            rel_pids=list(set(search_pids).intersection(set(val_pids)))

        for pid in rel_pids:
            df_pid=pd.read_hdf(batch_f,"/p{}".format(pid),mode='r')
            df_pid=df_pid[df_pid["PredScore"].notnull()]
            unique_reltimes=list(df_pid["RelDatetime"].unique())
            all_reltimes.extend(unique_reltimes)

    for pct in range(0,101,3):
        print("Percentile of rel-times: {}: {:.2f}".format(pct,np.percentile(all_reltimes,pct)/3600.))

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--split_desc",
                        default="../../data/exp_design/temp_splits_hirid2.pickle",
                        help="Split descriptor")
    parser.add_argument("--pred_path",
                        default="/cluster/home/mhueser/git/projects/2022/kidnews_public/data/predictions/reduced/temporal_1",
                        help="Prediction path")    

    # Output paths

    # Arguments
    parser.add_argument("--split_key", default="temporal_1", help="Split to use")

    parser.add_argument("--eval_key", default="Label_hirid_merged_24h_deleted_4h_WorseStateFromZeroEVAL0To48Hours_simple_features_lightgbm",
                        help="Evaluate key to analyze")

    parser.add_argument("--small_sample", default=False, action="store_true", help="Small sample?")
    
    configs=vars(parser.parse_args())

    execute(configs)
