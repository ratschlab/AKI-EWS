''' A script to batch the MIMIC-IV 
    patients into 50 batches, this just stores the identical batching
    of Thomas endpoint files'''

import os
import os.path
import argparse
import ipdb
import pickle
import glob

import pandas as pd
import numpy as np

def execute(configs):
    all_pids=list(pd.read_csv(configs["valid_pid_list"],sep='\t',header=None)[0].values)

    if configs["thomas_compat"]:
        chunks=[]
        for i in range(50):
            batch_df=pd.read_hdf(os.path.join(configs["ep_path"], "batch_{}.h5".format(i)), mode='r')
            unique_pids=list(batch_df.stay_id.unique())
            chunks.append(unique_pids)
    else:
        chunks=np.array_split(all_pids, configs["n_batches"])
        
    chunk_to_pids={}
    pid_to_chunk={}    
    for idx in range(len(chunks)):
        chunk_to_pids[idx]=chunks[idx]
        all_pids=chunks[idx]
        for pid in all_pids:
            pid_to_chunk[pid]=idx
    out_dict={"chunk_to_pids": chunk_to_pids,
              "pid_to_chunk": pid_to_chunk}
    pickle.dump(out_dict,open(configs["batch_desc"],'wb'))
    

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--valid_pid_list", default="/cluster/work/grlab/clinical/hirid2/research/misc_derived/RENAL_project/included_pids_MIMIC.tsv",
                        help="List of valid PIDs of MIMIC-IV to batch")

    parser.add_argument("--ep_path", default="/cluster/work/grlab/clinical/hirid2/research/3b_endpoints_renal/MIMIC-IV/220821",
                        help="Endpoint path")

    # Output paths
    parser.add_argument("--batch_desc",
                        default="/cluster/work/grlab/clinical/hirid2/research/misc_derived/id_lists/MIMIC_PID_chunking_50.pickle",
                        help="Path to hold the batch descriptor")

    # Arguments
    parser.add_argument("--n_batches", type=int, default=50, help="Number of batches to create")

    parser.add_argument("--thomas_compat", default=True, action="store_true", help="Simply use the batches of Thomas")

    configs=vars(parser.parse_args())

    # Arguments

    execute(configs)
