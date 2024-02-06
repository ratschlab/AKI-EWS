''' Batch the merged data into the 50 batches, to process
    it in parallel'''

import argparse
import os
import os.path
import ipdb
import pickle

import pandas as pd
import numpy as np

def execute(configs):
    df_merged=pd.read_parquet(configs["merged_path"])
    mi=df_merged.index.values
    sid_fields=list(map(lambda mi: int(mi[0]), mi))
    df_merged["stay_id"]=sid_fields

    with open(configs["batch_map"],'rb') as fp:
        bmap=pickle.load(fp)["chunk_to_pids"]

    search_pids=[]
    for k in bmap.keys():
        search_pids.extend(bmap[k])

    merged_pids=list(df_merged["stay_id"].unique())
        
    for chunk in sorted(bmap.keys()):
        print("Processing chunk: {}".format(chunk))
        chunk_pids=list(map(int,bmap[chunk]))
        print("Number of PIDs in chunk {}: {}".format(chunk, len(chunk_pids)))
        merged_subset=df_merged[df_merged["stay_id"].isin(chunk_pids)]
        print("Number of unique PIDs found: {}".format(len(merged_subset["stay_id"].unique())))
        merged_subset.reset_index(drop=True,inplace=True)

        # The variables for Urine can be treated the same
        merged_subset["vm276"]=merged_subset["vm24"]
        
        merged_subset.to_hdf(os.path.join(configs["out_path"],"batch_{}.h5".format(chunk)), key="data",
                             mode='w', complevel=5, complib="blosc:lz4", data_columns=["stay_id"], format="table")
        
if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--batch_map", default="/cluster/work/grlab/clinical/hirid2/research/misc_derived/id_lists/MIMIC_PID_chunking_50.pickle",
                        help="Batch map which to apply to the merged data")

    parser.add_argument("--merged_path", default="/cluster/work/grlab/clinical/hirid2/research/3_merged/MIMIC-IV/MIMICIV_merged.parquet",
                        help="Merged data path")

    # Output paths
    parser.add_argument("--out_path", default="/cluster/work/grlab/clinical/hirid2/research/3_merged/MIMIC-IV",
                        help="Output location to write the batched files to")

    # Arguments

    configs=vars(parser.parse_args())

    execute(configs)
