''' Batch the endpoint data into the 50 batches, to process
    it in parallel'''

import argparse
import os
import os.path
import ipdb
import pickle

import pandas as pd
import numpy as np

def execute(configs):
    df_endpoint=pd.read_parquet(configs["ep_path"])
    mi=df_endpoint.index.values
    sid_fields=list(map(lambda mi: int(mi[0]), mi))
    time_fields=list(map(lambda mi: mi[1], mi))
    df_endpoint["stay_id"]=sid_fields
    df_endpoint["AbsDatetime"]=time_fields

    with open(configs["batch_map"],'rb') as fp:
        bmap=pickle.load(fp)["chunk_to_pids"]

    search_pids=[]
    for k in bmap.keys():
        search_pids.extend(bmap[k])

    endpoint_pids=list(df_endpoint["stay_id"].unique())

    complete_num_ep=0
        
    for chunk in sorted(bmap.keys()):
        print("Processing chunk: {}".format(chunk))
        chunk_pids=list(map(int,bmap[chunk]))
        print("Number of PIDs in chunk {}: {}".format(chunk, len(chunk_pids)))
        ep_subset=df_endpoint[df_endpoint["stay_id"].isin(chunk_pids)]
        print("Number of unique PIDs found: {}".format(len(ep_subset["stay_id"].unique())))
        complete_num_ep+=len(ep_subset["stay_id"].unique())
        ep_subset.reset_index(drop=True,inplace=True)
        ep_subset.to_hdf(os.path.join(configs["out_path"],"batch_{}.h5".format(chunk)), key="data",
                         mode='w', complevel=5, complib="blosc:lz4", data_columns=["stay_id"], format="table")

    print("Number of Endpoints PIDs written: {}".format(complete_num_ep))
        
if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--batch_map", default="../../data/exp_design/mimic_chunking_50.pickle",
                        help="Batch map which to apply to the merged data")

    parser.add_argument("--ep_path", default="/cluster/work/grlab/clinical/hirid2/research/3_merged/MIMIC-IV/MIMICIV_endpoint_reprocessed.parquet",
                        help="Merged data path")

    # Output paths
    parser.add_argument("--out_path", default="../../data/endpoints/mimic_endpoints",
                        help="Output location to write the batched files to")

    # Arguments

    configs=vars(parser.parse_args())

    execute(configs)
