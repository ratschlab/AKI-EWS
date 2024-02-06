'''Label statistics'''

import argparse
import os
import os.path
import ipdb
import glob
import random

import numpy as np
import pandas as pd

def execute(configs):
    random.seed(2022)
    label_fs=sorted(glob.glob(os.path.join(configs["label_path"],"reduced", configs["split_key"], "batch_*.h5")))
    print("Number of label files: {}".format(len(label_fs)))

    pos_count=0
    neg_count=0
    nan_count=0
    all_count=0

    random.shuffle(label_fs)

    for fidx,fpath in enumerate(label_fs):
        print("Label file: {}/{}".format(fidx+1,len(label_fs)))
        with pd.HDFStore(fpath,'r') as store:
            ks=store.keys()
        for pat in ks:
            df=pd.read_hdf(fpath,pat,mode='r')
            label_col=np.array(df[configs["label_key"]])
            nan_count+=np.sum(np.isnan(label_col))
            all_count+=label_col.size
            neg_count+=np.sum(label_col==0)
            pos_count+=np.sum(label_col==1)
        print("NAN ratio [%]: {:.2f}, Prev: [%]: {:.2f}".format(100*nan_count/all_count,100*pos_count/(pos_count+neg_count)))

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--label_path", default="/cluster/work/grlab/clinical/hirid2/research/6b_labels_renal/mimic_iv_extval",
                        help="Label path to analyze")

    # Output paths

    # Arguments
    parser.add_argument("--split_key", default="exploration_1", help="Split key to analyze")

    parser.add_argument("--label_key", default="220821_merged_24h_deleted_4h_WorseStateFromZeroEVAL0To48Hours",
                        help="Label to analyze")

    configs=vars(parser.parse_args())

    execute(configs)
