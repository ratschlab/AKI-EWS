'''
Generate tables of time-slice based results for a set of prediction results
'''
import argparse
import os
import os.path
import math
import itertools
import ipdb
import pickle

import pandas as pd
import numpy as np
import scipy
import gin

import sklearn.metrics as skmetrics

from resp_ews.utils.io import load_pickle


def custom_roc_curve(labels, scores):
    ''' A custom ROC curve with a large number of thresholds'''
    n_thresholds=10000
    perc_range=np.flip(np.linspace(0,100,n_thresholds))
    fpr_out=[]
    tpr_out=[]
    taus=[]
    for ts in scores:
        #ts=np.percentile(scores,perc_ts)
        taus.append(ts)
        pred_labels=(scores>=ts).astype(np.int)
        fpr=1-np.sum((pred_labels==0)&(labels==0))/np.sum(labels==0)
        tpr=np.sum((pred_labels==1)&(labels==1))/np.sum(labels==1)
        fpr_out.append(fpr)
        tpr_out.append(tpr)
    
    return (np.array(fpr_out),np.array(tpr_out),np.array(taus))
    


def custom_pr_curve(labels, scores):
    ''' A custom PR curve with a large number of thresholds'''
    n_thresholds=10000
    perc_range=np.linspace(0,100,n_thresholds)
    precs_out=[]
    recs_out=[]
    taus=[]
    for ts in scores:
        #ts=np.percentile(scores,perc_ts)
        taus.append(ts)
        pred_labels=(scores>=ts).astype(np.int)
        rec=np.sum((pred_labels==1)&(labels==1))/np.sum(labels==1)
        prec=np.sum((pred_labels==1)&(labels==1))/np.sum(pred_labels==1)
        precs_out.append(prec)
        recs_out.append(rec)
    return (np.array(precs_out),np.array(recs_out),np.array(taus))
    


def corrected_pr_curve(labels, scores, correct_factor=None, custom_curve=False):
    ''' Returns a collection of metrics'''
    taus=[]
    tps=[]
    fps=[]
    npos=np.sum(labels==1.0)    

    if custom_curve:
        threshold_set=np.copy(scores)
    else:
        threshold_set=np.arange(0.0,1.001,0.001)
    
    for tau in threshold_set:
        #tau=np.percentile(scores,perc_ts)
        der_labels=(scores>=tau).astype(np.int)
        taus.append(tau)
        tp=np.sum((labels==1.0) & (der_labels==1.0))
        fp=np.sum((labels==0.0) & (der_labels==1.0))
        tps.append(tp)
        fps.append(fp)

    tps=np.array(tps)
    fps=np.array(fps)
    taus=np.array(taus)

    recalls=tps/npos
    precisions=tps/(tps+correct_factor*fps)
    precisions[np.isnan(precisions)]=1.0
        
    return (precisions, recalls, taus)


def execute(configs):

    n_skipped_patients=0
    scores_dict={}
    labels_dict={}

    df_out_dict= { 
        "test_set_pids": [],
        "test_set_samples": [],
        "auroc": [],
        "auprc": [],
        "recall": [],
        "precision": [],
        "fpr": [],
        "label_prevalence": [],
        "task": [],
        "database": [],
        "split": []
    }

    curve_dict= {
        "roc_curves": {},
        "pr_curves": {}
    }

    work_idx=1
    all_work=configs["task_keys"]

    if configs["restrict_tis_hours"]:
        print("Restricting evaluation to [{},{}] [h] into stay".format(configs["restrict_tis_hours_left"],
                                                                      configs["restrict_tis_hours_right"]))

    if configs["autosense_eval_hour"]:
        print("Auto-restricting evaluation hour")

    for label_key,custom_str,database,split_key,correct_curve in all_work:
        print("Processing task/split {}/{}".format(work_idx,len(all_work)))

        if "combined_model_from_separate" in label_key:
            pred_score_key="PredScoreCalibrated"
        else:
            pred_score_key="PredScore"

        if configs["autosense_eval_hour"]:
            eval_point=int(label_key.split("_")[2][2:-5])
        
        work_idx+=1

        if custom_str is not None:
            df_out_dict["task"].append("{}_{}".format(label_key,custom_str))
        else:
            df_out_dict["task"].append(label_key)

        df_out_dict["database"].append(database)
            
        df_out_dict["split"].append(split_key)
        cum_pred_scores=[]
        cum_labels=[]
        
        if database=="hirid":
            data_split=load_pickle(configs["bern_temporal_split_path"])[split_key]
            batch_map=load_pickle(configs["hirid_pid_map_path"])["pid_to_chunk"]            
        elif database=="mimic":
            data_split=load_pickle(configs["mimic_temporal_split_path"])[split_key]
            batch_map=load_pickle(configs["umc_pid_map_path"])["pid_to_chunk"]
            
        pred_pids=data_split["test"]
        n_test_pids=0
        nan_pred_cnt=0
        coh_cnt=0
        incoh_cnt=0

        if database in ["hirid","mimic"]:
            output_dir=os.path.join(configs["predictions_dir"],"reduced","temporal_1",label_key)
        
        ep_dir=os.path.join(configs["endpoint_path"],split_key)

        for pidx,pid in enumerate(pred_pids):
            if (pidx+1)%1000==0 and configs["verbose"]:
                print("{}/{}".format(pidx+1,len(pred_pids)))
            if pidx>=500 and configs["debug_mode"]:
                break
            batch_pat=batch_map[pid]
            try:
                search_f=os.path.join(output_dir,"batch_{}.h5".format(batch_pat))
                df_pred=pd.read_hdf(search_f,"/p{}".format(pid), mode='r')
                
                if custom_str is not None:
                    df_ep=pd.read_hdf(os.path.join(ep_dir,"batch_{}.h5".format(batch_pat)),
                                      mode='r',where="PatientID={}".format(pid))
                    df_ep=df_ep[pd.notna(df_pred["TrueLabel"]) & pd.notna(df_pred[pred_score_key])]

                if "Multiclass" in label_key:
                    df_pred=df_pred[pd.notna(df_pred["TrueLabel"]) & pd.notna(df_pred["PredScore_0"])]
                else:
                    df_pred=df_pred[pd.notna(df_pred["TrueLabel"]) & pd.notna(df_pred[pred_score_key])]         

            except (KeyError, FileNotFoundError) as exc:
                n_skipped_patients+=1
                continue

            n_test_pids+=1

            if configs["restrict_tis_hours"]:
                df_pred=df_pred[(df_pred["RelDatetime"]>=3600*configs["restrict_tis_hours_left"]) & \
                                (df_pred["RelDatetime"]<=3600*configs["restrict_tis_hours_right"])]

            if configs["autosense_eval_hour"]:
                df_pred=df_pred[df_pred["RelDatetime"]==3600*eval_point]
                
            if df_pred.shape[0]==0:
                continue

            if custom_str is None:

                if "Multiclass" in label_key:
                    pred_scores=np.array(df_pred["PredScore_1"])+np.array(df_pred["PredScore_2"])
                else:
                    pred_scores=np.array(df_pred[pred_score_key])
                    
            else:
                if custom_str=="custom_0":
                    pred_scores=(np.array(df_ep["ext_ready_violation_score"])>0).astype(np.float)
                elif custom_str=="custom_threshold":
                    pred_scores=np.array(1-np.array(df_ep["readiness_ext"]))
                    
            true_labels=np.array(df_pred["TrueLabel"])

            if np.sum(np.isnan(pred_scores))==0 and np.sum(np.isnan(true_labels))==0:
                cum_pred_scores.append(pred_scores)
                cum_labels.append(true_labels)
            else:
                nan_pred_cnt+=1

        #print("NAN pred count: {}".format(nan_pred_cnt))
        #print("Label coh, inh count: {}, {}".format(coh_cnt,incoh_cnt))
        
        df_out_dict["test_set_pids"].append(n_test_pids)
        n_test_samples=np.concatenate(cum_pred_scores).size
        df_out_dict["test_set_samples"].append(n_test_samples)
        raw_scores=np.concatenate(cum_pred_scores)
        raw_labels=np.concatenate(cum_labels)

        print("Number of scores: {}".format(len(raw_scores)))

        assert(np.sum(np.isnan(raw_labels))==0)
        prevalence=np.sum(raw_labels==1.0)/raw_labels.size
        df_out_dict["label_prevalence"].append(prevalence)

        if not correct_curve:
            reference_prevalence=prevalence
            print("Reference prevalence: {:.3f}".format(reference_prevalence))

        if configs["invert_scores"]:
            raw_labels=1.0-raw_labels
            raw_scores=1.0-raw_scores

        if "Multiclass" in label_key and configs["conf_matrix_eval"]:
            conf_matrix=skmetrics.confusion_matrix(raw_labels,raw_scores)
        else:

            if configs["custom_roc_pr_curves"]:
                fpr_model,tpr_model,_=custom_roc_curve(raw_labels, raw_scores)
            else:
                fpr_model,tpr_model,_=skmetrics.roc_curve(raw_labels,raw_scores,pos_label=1)

            tpr_model=tpr_model[np.argsort(fpr_model)]
            fpr_model=fpr_model[np.argsort(fpr_model)]                
                
            curve_dict["roc_curves"][(database,label_key,split_key)]={"fpr": fpr_model, "tpr": tpr_model}
            print("Scores Mean: {:.3f}, {:.3f}".format(np.mean(raw_scores), np.std(raw_scores)))
            auroc=skmetrics.auc(fpr_model,tpr_model)
            df_out_dict["auroc"].append(auroc)

        if custom_str is not None:
            conf_matrix=skmetrics.confusion_matrix(raw_labels,raw_scores)
            fpr=1-conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
            df_out_dict["fpr"].append(fpr)
            df_out_dict["precision"].append(skmetrics.precision_score(raw_labels,raw_scores))
            df_out_dict["recall"].append(skmetrics.recall_score(raw_labels,raw_scores))
        else:
            df_out_dict["fpr"].append(np.nan)
            df_out_dict["precision"].append(np.nan)
            df_out_dict["recall"].append(np.nan)

        if correct_curve:
            correction_factor=(1/reference_prevalence-1)/(1/prevalence-1)
            print("Target prevalence: {:.3f}".format(prevalence))
            print("Correction factor: {:.3f}".format(correction_factor))
            precs_model,recalls_model,_=corrected_pr_curve(raw_labels,raw_scores,correct_factor=correction_factor,
                                                           custom_curve=configs["custom_roc_pr_curves"])
        else:

            if configs["custom_roc_pr_curves"]:
                precs_model,recalls_model,_=custom_pr_curve(raw_labels, raw_scores)
            else:
                precs_model,recalls_model,_=skmetrics.precision_recall_curve(raw_labels,raw_scores,pos_label=1)

        precs_model=precs_model[np.argsort(recalls_model)]
        recalls_model=recalls_model[np.argsort(recalls_model)]
                
        auprc=skmetrics.auc(recalls_model,precs_model)
            
        df_out_dict["auprc"].append(auprc)
        curve_dict["pr_curves"][(database,label_key,split_key)]={"recalls": recalls_model, "precs": precs_model}

    result_frame=pd.DataFrame(df_out_dict)

    if configs["restrict_tis_hours"]:
        result_frame.to_csv(os.path.join(configs["eval_table_dir"], "task_results_TIS_{}_{}.tsv".format(configs["restrict_tis_hours_left"],
                                                                                                        configs["restrict_tis_hours_right"])),sep='\t',index=False)
        with open(os.path.join(configs["eval_table_dir"], "raw_results_TIS_{}_{}.pickle".format(configs["restrict_tis_hours_left"],
                                                                                                configs["restrict_tis_hours_right"])),'wb') as fp:
            pickle.dump(curve_dict,fp)
    else:
        result_frame.to_csv(os.path.join(configs["eval_table_dir"], "task_results.tsv"),sep='\t',index=False)
        with open(os.path.join(configs["eval_table_dir"], "raw_results.pickle"),'wb') as fp:
            pickle.dump(curve_dict,fp)

@gin.configurable
def parse_gin_args(old_configs,gin_configs=None):
    gin_configs=gin.query_parameter("parse_gin_args.gin_configs")
    for k in old_configs.keys():
        if old_configs[k] is not None:
            gin_configs[k]=old_configs[k]
    gin.bind_parameter("parse_gin_args.gin_configs",gin_configs)
    return gin_configs

 
if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--verbose", default=None, action="store_true", help="Should verbose messages be printed?")
    parser.add_argument("--debug_mode", default=None, action="store_true", help="Debug mode with fewer patients")
 
    parser.add_argument("--gin_config", default="./configs/eval.gin", help="GIN config to use")   
    
    parser.add_argument("--restrict_tis_hours", default=False, action="store_true", help="Restrict evaluation to a certain hour")
    parser.add_argument("--restrict_tis_hours_left", default=48, type=int, help="Restrict evaluation to a certain hour, left bound")
    parser.add_argument("--restrict_tis_hours_right", default=100, type=int, help="Restrict evaluation to a certain hour, right bound")    

    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    execute(configs)
