''' Analyze the risk of Furosemide treatment
    with propensity score matched patient that
    do not receive Furosemide'''

import os
import os.path
import sys
import argparse
import glob
import pickle
import ipdb

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import scipy.spatial as sp_spatial
import sksurv.nonparametric as sksurv_np
import sksurv.compare as sksurv_comp
import sksurv.util as sksurv_util

def plot_one_curve(pred_path, var_interest, var_label, plot_color, configs=None):

    with open(configs["split_path"],'rb') as fp:
        split_desc=pickle.load(fp)

    test_pids=split_desc["temporal_1"]["test"]

    print("Number of test PIDs: {}".format(len(test_pids)))

    static_df=pd.read_hdf(os.path.join(configs["imputed_path"],"static.h5"))

    pred_files=list(sorted(glob.glob(os.path.join(pred_path,"batch_*.h5"))))
    print("Number of prediction files: {}".format(len(pred_files)))

    all_pred_scores=[]
    all_furosemide=[]
    all_event_status=[]
    all_event_time=[]

    for pred_f in pred_files:
        suffix=pred_f.split("/")[-1].strip()
        batch_no=int(suffix[:-3].split("_")[1])

        # No test PIDs in these batches
        if batch_no<90:
            continue

        df_imp=pd.read_hdf(os.path.join(configs["imputed_path"],"batch_{}.h5".format(batch_no)))[["PatientID","AbsDatetime","RelDatetime",var_interest]]
        df_ep=pd.read_hdf(os.path.join(configs["endpoint_path"],"batch_{}.h5".format(batch_no)))[["PatientID","AbsDatetime","geq1"]]
        print("Batch number: {}".format(batch_no))

        with pd.HDFStore(pred_f,'r') as hstore:
            pat_keys=list(hstore.keys())
            pat_ids=list(map(lambda pkey: int(pkey[2:]), pat_keys))
            test_batch_pids=list(set(pat_ids).intersection(set(test_pids)))
            if len(test_batch_pids)==0:
                continue

            for pidx,pid in enumerate(test_batch_pids):

                if (pidx+1)%100==0:
                    print("PID: {}/{}".format(pidx+1,len(test_batch_pids)))
                
                df_pid=pd.read_hdf(pred_f,"/p{}".format(pid),mode='r')
                df_static_pid=static_df[static_df.PatientID==pid]
                has_died=df_static_pid["Discharge"].iloc[0]==4
                df_imp_pid=df_imp[df_imp.PatientID==pid]
                df_ep_pid=df_ep[df_ep.PatientID==pid]
                assert (df_ep_pid.AbsDatetime.values==df_pid.AbsDatetime.values).all()
                assert (df_imp_pid.AbsDatetime.values==df_pid.AbsDatetime.values).all()                

                eps=df_ep_pid["geq1"].values
                time_to_failure=np.zeros_like(eps)
                events=np.zeros_like(eps)
                
                for i in range(len(eps)):

                    # Event at this timestamp, event time 0, event indicator 1
                    if eps[i]==1:
                        events[i]=1

                    # Not the last timestamp
                    elif i<len(eps)-1:
                        rem_eps=eps[i+1:]
                        fail_idx=np.where(rem_eps==1)[0]

                        # Kidney failure in the remaining data, event time offset, event indicator 1
                        if len(fail_idx)>0:
                            time_to_failure[i]=5*fail_idx.min()
                            events[i]=1

                        # No kidney failure in remaining data, event time length of follow-up, event indicator 0
                        else:
                            time_to_failure[i]=5*len(rem_eps)

                pred_scores=df_pid["PredScore"].values
                furosemide=df_imp_pid[var_interest].values

                index_arr=np.isfinite(pred_scores)
                red_pred_scores=pred_scores[index_arr]
                red_furosemide=furosemide[index_arr]
                red_time_to_event=time_to_failure[index_arr]
                red_event_status=events[index_arr]

                all_pred_scores.append(red_pred_scores)
                all_furosemide.append(red_furosemide)
                all_event_status.append(red_event_status)
                all_event_time.append(red_time_to_event)

        if configs["debug_mode"]:
            break

    GLOBAL_pred_scores=np.concatenate(all_pred_scores)
    GLOBAL_furo=np.concatenate(all_furosemide)
    GLOBAL_event_status=np.concatenate(all_event_status)
    GLOBAL_event_time=np.concatenate(all_event_time)

    no_furo_index=(GLOBAL_furo==0) | np.isnan(GLOBAL_furo)
    furo_index=GLOBAL_furo>0

    pred_furo=GLOBAL_pred_scores[furo_index]
    pred_no_furo=GLOBAL_pred_scores[no_furo_index]

    event_status_furo=GLOBAL_event_status[furo_index]
    event_status_no_furo=GLOBAL_event_status[no_furo_index]

    event_time_furo=GLOBAL_event_time[furo_index]
    event_time_no_furo=GLOBAL_event_time[no_furo_index]

    print("Number of time-points with furosemide: {}".format(len(event_time_furo)))
    print("Number of time-points w/o furosemide: {}".format(len(event_time_no_furo)))

    kd_data=np.zeros((len(pred_no_furo),1))
    kd_data[:,0]=pred_no_furo
    
    kd_tree=sp_spatial.KDTree(kd_data)

    event_time_prop=np.zeros_like(event_time_furo)
    event_status_prop=np.zeros_like(event_status_furo)

    for i in range(len(pred_furo)):
        query_point=np.array([[pred_furo[i]]])
        dists,indices=kd_tree.query(query_point)
        event_time_prop[i]=event_time_no_furo[indices[0]]
        event_status_prop[i]=event_status_no_furo[indices[0]]

    event_status_furo=event_status_furo.astype(bool)
    event_status_prop=event_status_prop.astype(bool)

    treatment_time,treatment_prop=sksurv_np.kaplan_meier_estimator(event_status_furo,event_time_furo)
    control_time,control_prop=sksurv_np.kaplan_meier_estimator(event_status_prop,event_time_prop)

    treatment_time=treatment_time/60.
    control_time=control_time/60.

    group_treat=np.ones_like(event_time_furo)
    group_control=np.zeros_like(event_time_prop)

    all_groups=np.concatenate([group_treat,group_control])
    all_times=np.concatenate([event_time_furo,event_time_prop])
    all_events=np.concatenate([event_status_furo,event_status_prop])
    print("Running log-rank test")
    struct_arr=sksurv_util.Surv.from_arrays(all_events,all_times)
    chisq,pval=sksurv_comp.compare_survival(struct_arr,all_groups)
    print("Log rank-test, Chisq={}, p-val: {}".format(chisq,pval))
    
    plt.step(treatment_time,treatment_prop,linestyle="-",color=plot_color,
             label="{} given at t=0".format(var_label))
    plt.step(control_time,control_prop,linestyle="--",color=plot_color,
             label="No {} given at t=0 (matched)".format(var_label))


def execute(configs):

    for pred_path, var_interest, var_label, plot_color in [(configs["fluid_pred_path"],"vm31","Fluid","red"),
                                                           (configs["furo_pred_path"],"pm69","Furosemide","blue")]:
        plot_one_curve(pred_path, var_interest, var_label,plot_color, configs=configs)
    
    plt.legend()
    plt.ylabel("Est. probability of freedom of AKI at time t")
    plt.xlabel("Time t [Hours]")
    plt.ylim((0,1))
    plt.xlim((0,60))
    plt.savefig(os.path.join(configs["plot_path"],"furo_fluid_vs_controls.pdf"))
    plt.savefig(os.path.join(configs["plot_path"],"furo_fluid_vs_controls.png"))
    

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--furo_pred_path",
                        default="/cluster/work/grlab/clinical/hirid2/research/KIDNEY_RELEASE/predictions/reduced/temporal_1/Label_hirid_merged_24h_deleted_4h_WorseStateFromZeroEVAL0To48Hours_furo_effect_matching_model_lgbm_flaml",
                        help="Predictions to use for Furo matching model")
    parser.add_argument("--fluid_pred_path",
                        default="/cluster/work/grlab/clinical/hirid2/research/KIDNEY_RELEASE/predictions/reduced/temporal_1/Label_hirid_merged_24h_deleted_4h_WorseStateFromZeroEVAL0To48Hours_fluid_effect_matching_model_lgbm_flaml",
                        help="Predictions to use for Fluid matching model")

    parser.add_argument("--imputed_path",
                        default="/cluster/work/grlab/clinical/hirid2/research/KIDNEY_RELEASE/imputed/noimpute_hirid2/reduced/temporal_1",
                        help="Input data to use")

    parser.add_argument("--endpoint_path",
                        default="/cluster/work/grlab/clinical/hirid2/research/KIDNEY_RELEASE/endpoints/hirid_merged_24h_deleted_4h",
                        help="Endpoints to use")

    parser.add_argument("--split_path",
                        default="/cluster/work/grlab/clinical/hirid2/research/KIDNEY_RELEASE/exp_design/temp_splits_hirid2.pickle",
                        help="Split descriptor to use")

    # Output paths
    parser.add_argument("--plot_path",
                        default="/cluster/work/grlab/clinical/hirid2/research/KIDNEY_RELEASE/plots/effect_furosemide",
                        help="Location where to store the plots")

    # Arguments
    parser.add_argument("--debug_mode", action="store_true", default=False, help="Debug mode?")
    
    configs=vars(parser.parse_args())
    
    execute(configs)
