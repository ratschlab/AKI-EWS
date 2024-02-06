''' A script that post-processes the renal endpoint'''

import argparse
import os
import os.path
import ipdb
import glob
import sys

import pandas as pd
import numpy as np

def delete_short_gaps_events(geq_arr, configs=None):

    # Traverse twice to first delete short gaps, in the second run delete short events
    for tidx in range(2):

        events=[]
        gaps=[]

        # Pre-process the array to detect event lengths and gaps
        in_event=False
        in_gap=False
        for jdx in range(geq_arr.size):
            if not in_event and geq_arr[jdx]==1:
                in_event=True
                event_length=configs["grid_size_min"]
                event_start_idx=jdx
                if in_gap:
                    gaps.append((gap_start_idx,jdx-1))
                in_gap=False
            elif in_event and not geq_arr[jdx]==1:
                in_event=False
                in_gap=True
                gap_start_idx=jdx
                gap_length=configs["grid_size_min"]
                events.append((event_start_idx,jdx-1))
            elif in_event and geq_arr[jdx]==1:
                event_length+=configs["grid_size_min"]
                if jdx==geq_arr.size==1:
                    event.append(event_start_idx,jdx)

            elif not in_event and not geq_arr[jdx]==1:
                if in_gap:
                    gap_length+=configs["grid_size_min"]

        # Post-process the array to delete small gaps and events
        if tidx==0:
            short_gaps=list(filter(lambda gap: configs["grid_size_min"]*(gap[1]-gap[0])<=60*configs["gap_length"], gaps))                    
            for gstart,gend in short_gaps:
                geq_arr[gstart:gend+1]=1.0

        if tidx==1:
            short_events=list(filter(lambda event: configs["grid_size_min"]*(event[1]-event[0])<=60*configs["small_event_length"], events))
            for estart,eend in short_events:
                geq_arr[estart:eend+1]=0.0

    return geq_arr
    

def execute(configs):

    pid_key="PatientID"
    ts_key="AbsDatetime"
    
    out_base_path=os.path.join(configs["out_ep_path"],"{}_merged_{}h_deleted_{}h".format(configs["database"], configs["gap_length"],configs["small_event_length"]))
    if not os.path.exists(out_base_path):
        os.mkdir(out_base_path)

    for eidx,ep_file in enumerate(sorted(glob.glob(os.path.join(configs["ep_path"],"batch_*.h5")))):
        # Not a valid batch
        if "batch_100" in ep_file:
            continue
        batch_suffix=ep_file.split("/")[-1].strip()
        batch_idx=int(batch_suffix.split("_")[1][:-3])
        out_ep_file=os.path.join(out_base_path,batch_suffix)
        out_eps=[]
        print("Endpoint file: {}/{}".format(eidx+1,100),flush=True)
        df_ep=pd.read_hdf(ep_file,mode='r')
        df_ep.rename(columns={"stay_id": "PatientID", "charttime": "AbsDatetime"}, inplace=True)
        
        imp_path=os.path.join(configs["imp_path"], "batch_{}.h5".format(batch_idx))
        if not os.path.exists(imp_path):
            print("WARNING: No imputed batch file...")
            continue
        df_imp=pd.read_hdf(imp_path,mode='r')
        pids=df_ep[pid_key].unique()
        for pid in pids:
            df_pid=df_ep[df_ep[pid_key]==pid]
            assert configs["database"]=="mimic" or df_pid["AbsDatetime"].equals(df_pid["Datetime"])
            df_pid.sort_values(by=ts_key,inplace=True)
            unique_td=df_pid[ts_key].diff().iloc[1:].unique()

            # 5-minute endpoint grid for the HiRID-database
            if configs["database"]=="hirid":
                assert df_pid.shape[0]==1 or len(unique_td)==1 and unique_td[0]==np.timedelta64(300000000000,'ns')

            # 1 hour expected endpoint grid for the MIMIC database
            elif configs["database"]=="mimic":
                assert df_pid.shape[0]==1 or len(unique_td)==1 and unique_td[0]==np.timedelta64(3600000000000,'ns')

            if configs["database"]=="hirid":
                df_pid.drop(columns=["Datetime"],inplace=True)
                
            df_pid.set_index(keys="AbsDatetime",inplace=True,verify_integrity=True)
            df_imp_pid=df_imp[df_imp["PatientID"]==pid]
            df_pid=df_pid.reindex(index=df_imp_pid.AbsDatetime,method="nearest")
            df_pid.reset_index(inplace=True)
            if df_pid.shape[0]==0:
                continue
            unique_td=df_pid.AbsDatetime.diff().iloc[1:].unique()
            assert(df_pid.shape[0]==1 or len(unique_td)==1 and unique_td[0]==np.timedelta64(300000000000,'ns'))
            geq1_arr=np.array(df_pid.geq1,dtype=np.float)
            geq2_arr=np.array(df_pid.geq2,dtype=np.float)
            geq3_arr=np.array(df_pid.geq3,dtype=np.float)

            # HIRID-II endpoints have a special encoding with separate unknown arrays
            if configs["database"]=="hirid":
                unknown_geq1_arr=np.array(df_pid.unknown_geq1,dtype=np.float)
                unknown_geq2_arr=np.array(df_pid.unknown_geq2,dtype=np.float)
                unknown_geq3_arr=np.array(df_pid.unknown_geq3,dtype=np.float)

                for jdx in range(geq1_arr.size):
                    if unknown_geq1_arr[jdx]==1:
                        geq1_arr[jdx]=np.nan

                for jdx in range(geq2_arr.size):
                    if unknown_geq2_arr[jdx]==1:
                        geq2_arr[jdx]=np.nan

                for jdx in range(geq3_arr.size):
                    if unknown_geq3_arr[jdx]==1:
                        geq3_arr[jdx]=np.nan

            # MIMIC endpoints have a special encoding with -1 denoting uncertain state
            elif configs["database"]=="mimic":
                for jdx in range(geq1_arr.size):
                    if geq1_arr[jdx]==-1:
                        geq1_arr[jdx]=np.nan

                for jdx in range(geq2_arr.size):
                    if geq2_arr[jdx]==-1:
                        geq2_arr[jdx]=np.nan

                for jdx in range(geq3_arr.size):
                    if geq3_arr[jdx]==-1:
                        geq3_arr[jdx]=np.nan 

            geq1_arr=delete_short_gaps_events(geq1_arr,configs=configs)
            geq2_arr=delete_short_gaps_events(geq2_arr,configs=configs)
            geq3_arr=delete_short_gaps_events(geq3_arr,configs=configs)

            df_pid_out=df_pid.copy()
            df_pid_out["geq1"]=geq1_arr
            df_pid_out["geq2"]=geq2_arr
            df_pid_out["geq3"]=geq3_arr
            
            out_eps.append(df_pid_out)

        all_eps=pd.concat(out_eps,axis=0,ignore_index=True)
        
        if configs["write_out"]:
            all_eps.to_hdf(out_ep_file,'data',mode='w',complevel=5,complib="blosc:lz4",data_columns=["PatientID"],
                           format="table")

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    
    #parser.add_argument("--ep_path", default="../../data/endpoints/hirid2_endpoints")
    parser.add_argument("--ep_path", default="../../data/endpoints/mimic_endpoints")
    
    #parser.add_argument("--imp_path", default="../../data/imputed/noimpute_hirid2/reduced/temporal_1")
    parser.add_argument("--imp_path", default="../../data/imputed/noimpute_mimic/reduced/random_1")
    
    # Output paths
    parser.add_argument("--out_ep_path", default="../../data/endpoints")
    
    parser.add_argument("--log_path", default="/cluster/home/mhueser/log_files/icu_score_renal")

    # Arguments
    parser.add_argument("--small_event_length", type=int, default=4, help="Small events with size less than this shall be deleted")
    parser.add_argument("--gap_length", type=int, default=24, help="Gaps of this length and smaller shall be closed")
    parser.add_argument("--write_out", default=True, action="store_true", help="Write the data out?")
    parser.add_argument("--cluster_mode", default=False, action="store_true", help="Cluster batch mode?")

    #parser.add_argument("--database", default="hirid", help="This is either hirid or mimic")
    parser.add_argument("--database", default="mimic", help="This is either hirid or mimic")    

    parser.add_argument("--grid_size_min", default=5, help="Cluster grid size in minutes")

    configs=vars(parser.parse_args())

    if configs["cluster_mode"]:
        sys.stdout=open(os.path.join(configs["log_path"], "postproc_ep_gl_{}h_se_{}h.stdout".format(configs["gap_length"],
                                                                                                    configs["small_event_length"])),'w')
        sys.stderr=open(os.path.join(configs["log_path"], "postproc_ep_gl_{}h_se_{}h.stderr".format(configs["gap_length"],
                                                                                                    configs["small_event_length"])),'w')

    execute(configs)
