''' An analysis of signed SHAP values in the test set of a split
    for individual features, using a scatter plot of feature value 
    vs. SHAP value for the most relevant features'''

import csv
import os.path
import os
import ipdb
import argparse
import glob
import sys
import random
import pickle
import copy

import numpy as np
import numpy.random as nprand
import pandas as pd
import matplotlib as mpl
mpl.use("PDF")
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as skmetrics

def translate_label(feat,name_dict=None,backup_dict=None, current_value=False):
    ''' Returns the label for the variable'''
    if "plain_" in feat:
        vid=feat.split("_")[1].strip()
        vname=name_dict[vid] if vid in name_dict else backup_dict[vid]

        if current_value:
            vlabel="{} Current value".format(vname)
        else:
            vlabel=vname
            
    elif "time_to_last_ms" in feat:
        vid=feat.split("_")[0].strip()
        vname=name_dict[vid] if vid in name_dict else backup_dict[vid]
        vlabel="{} Time to last ms.".format(vname)
    elif "measure_density" in feat:
        vid=feat.split("_")[0].strip()
        vname=name_dict[vid] if vid in name_dict else backup_dict[vid]
        vlabel="{} Measure density".format(vname)
    elif "instable" in feat:
        vid=feat.split("_")[0].strip()
        vname=name_dict[vid] if vid in name_dict else backup_dict[vid]
        vlabel="{} Instability {}".format(vname,feat," ".join(feat.split("_")[2:]))
    elif "median" in feat:
        vid=feat.split("_")[0].strip()
        vname=name_dict[vid] if vid in name_dict else backup_dict[vid]
        vlabel="{} Median {}".format(vname,feat,feat.split("_")[2])
    elif "iqr" in feat:
        vid=feat.split("_")[0].strip()
        vname=name_dict[vid] if vid in name_dict else backup_dict[vid]
        vlabel="{} IQR {}".format(vname,feat,feat.split("_")[2])
    elif "mode" in feat:
        vid=feat.split("_")[0].strip()
        vname=name_dict[vid] if vid in name_dict else backup_dict[vid]
        vlabel="{} Mode {}".format(vname,feat,feat.split("_")[2])
    else:

        if feat in name_dict:
            vlabel=name_dict[feat]
        else:
            vlabel=feat
            
    return vlabel

def execute(configs):
    name_dict=np.load(configs["mid_dict"],allow_pickle=True).item()
    name_dict["vm31"]="Fluid administration"
    #name_dict["pm69"]="Furosemide [mg/h]"
    name_dict["pm69"]="Furosemide"
    name_dict["RelDatetime"]="Time since admission"
    name_dict["Emergency"]="Emergency admission?"
    name_dict["pm290"]="Laxatives"
    name_dict["vm24"]="OUTUrine/c"
    name_dict["pm104"]="Mg (medication)"
    name_dict["pm92"]="Platelet inhibitors"
    name_dict["vm226"]="Amount of secretion"
    
    backup_dict={"vm202": "OUTDialysis/c", "RelDatetime": "Time since admission", "vm201": "Chronic kidney disease?",
                 "static_APACHEPatGroup": "APACHE Patient Group", "static_Age": "Patient age", "static_Sex": "Patient gender", "vm212": "vm212",
                 "vm224": "Spontanatmung", "vm275": "Peritoneal dialysis", "vm253": "M-kr arm li", "vm204": "Hematocrit", "vm200": "vm200", "vm255": "M-Kr Bein li", "vm214": "I:E (s)",
                 "vm215": "MV Exp", "vm309": "Supplemental FiO2 %", "pm290": "Abführend", "vm249": "PubReaktre", "vm240": "Woher?", "vm227": "Sekret Konsistenz",
                 "vm216": "MV Spont. servo","vm318": "Atemfrequenz",
                 "vm276": "OUTurine/h",
                 "vm306": "Ventilator mode group", "vm308": "Ventilator mode mode", "vm307": "Ventilator mode subgroup","vm223": "Extubation time-point",
                 "vm313": "Tracheotomy state", "vm319": "RRsp/m", "vm312": "Intubation state", "vm315": "TVs", "vm293": "PEEP(s)", "vm211": "Pressure support", 
                 "vm226": "Sekret menge", "vm317": "Urine culture", "vm251": "NRS Ans 0-10", "vm256": "M-Kr Bein re","vm242": "Gehen 400m", "vm314": "TV(m)",
                 "fio2estimated": "FiO2 estimate","ventstate": "Ventilation state", "Age": "Patient age", "vm259": "PeriphHandLi", "Sex": "Gender",
                 "PatGroup": "Patient group"}

    static_encode_dict= {"Sex": {"M": 0, "F": 1}}

    # Features that should be plotted
    feats_to_analyze=configs["FEATS_TO_ANALYZE"]

    # Load LSTM importance arrays
    lstm_imp=pd.read_csv(configs["lstm_imp_path"],sep=',',header=0)
    lstm_imp["importance_score"]=-lstm_imp["mean delta AUPRC"]
    lstm_imp_dict=dict(zip(lstm_imp["variable"],lstm_imp["importance_score"]))

    df_static=pd.read_hdf(configs["static_path"], mode='r')        
    static_cols=df_static.columns.values.tolist()
    kept_static_cols=list(filter(lambda col: col in feats_to_analyze, static_cols))
    kept_pred_cols=["AbsDatetime","PredScore","TrueLabel"]+list(map(lambda col: "RawShap_"+col, feats_to_analyze))
    kept_feat_cols=["AbsDatetime"]+list(filter(lambda col: col not in kept_static_cols, feats_to_analyze))


    if configs["create_overview_plot"]:    
        print("Loading LGBM model...")

        modelpath=os.path.join(configs["pred_path"], "temporal_1", configs["model_config"],"best_model.pickle")
        with open(modelpath,'rb') as fp:
            LGBM_model=pickle.load(fp)

        print("Loading full HIRID-II data-set...")

        with open(configs["full_data_path"],'rb') as fp:
            full_HIRID2_data=pickle.load(fp)

        X_test=full_HIRID2_data["X_test"]
        y_test=full_HIRID2_data["y_test"]
        varnames=list(full_HIRID2_data["fnames"])
        LGBM_feat_order=LGBM_model.feature_name_
        colidxs=[varnames.index(var) for var in LGBM_feat_order]
        X_test_red=X_test[:,colidxs]
        y_test_pred=LGBM_model.predict_proba(X_test_red)[:,1]
        orig_AUPRC=skmetrics.average_precision_score(y_test,y_test_pred)

        lgbm_imp_dict=dict()

        # Now corrupt the data for each variable 5 times
        for vidx,var in enumerate(LGBM_feat_order):
            print("Corrupting variable {}/{}: {}".format(vidx+1,len(LGBM_feat_order), var))
            perf_deltas=[]
            for _ in range(configs["n_replicates"]):
                X_corrupted=copy.copy(X_test_red)
                orig_col=X_corrupted[:,vidx]
                nprand.shuffle(orig_col)
                X_corrupted[:,vidx]=orig_col
                y_test_pred_corrupt=LGBM_model.predict_proba(X_corrupted)[:,1]
                corrupt_AUPRC=skmetrics.average_precision_score(y_test,y_test_pred_corrupt)
                perf_deltas.append(corrupt_AUPRC-orig_AUPRC)
            mean_delta=np.mean(perf_deltas)
            print("Mean AUPRC delta {:.5f}".format(mean_delta))
            lgbm_imp_dict[var]=-mean_delta

    with open(configs["split_path"],'rb') as fp:
        splits=pickle.load(fp)

    acc_dict={}

    for split,split_desc in configs["SPLITS"]:
        print("Analyzing split: {}".format(split))
        predfs=glob.glob(os.path.join(configs["pred_path"], split, configs["model_config"], "batch_*.h5"))
        print("Number of batches: {}".format(len(predfs)))
        for fpath in sorted(predfs):
            batch_id=int(fpath.split('/')[-1].split(".")[0][6:])

            if batch_id<90:
                continue

            featpath=os.path.join(configs["feat_path"], split_desc, "batch_{}.h5".format(batch_id))
            with pd.HDFStore(fpath,'r') as hstore:
                all_pids=list(map(lambda item: int(item[2:]), list(hstore.keys())))
                print("Number of PIDs in batch {}: {}".format(batch_id, len(all_pids)))

            test_pids=splits[split_desc]["test"]
            all_test_pids=list(set(all_pids).intersection(set(test_pids)))

            if len(all_test_pids)==0:
                continue

            df_feat_batch=pd.read_hdf(featpath,"/X",mode='r')

            if configs["small_sample"]:
                random.shuffle(all_test_pids)
                all_test_pids=all_test_pids[:100]
                
            for pid in all_test_pids:
                df_static_pid=df_static[df_static["PatientID"]==pid]
                df_pred=pd.read_hdf(fpath,"/p{}".format(pid),mode='r')

                if configs["worst_pred_analysis"]:
                    all_pred_cols=df_pred.columns.values.tolist()
                    shap_cols=list(filter(lambda col: "RawShap_" in col, all_pred_cols))
                    kept_pred_cols=["AbsDatetime","PredScore","TrueLabel"]+shap_cols

                df_pred=df_pred[kept_pred_cols]
                df_feat=df_feat_batch[df_feat_batch.PatientID==pid][kept_feat_cols]
                df_merged=df_pred.merge(df_feat,how="inner",on=["AbsDatetime"])
                df_merged=df_merged[df_merged["PredScore"].notnull()]
                
                for col in kept_static_cols:
                    empty_arr=np.zeros(df_merged.shape[0])
                    fill_val=df_static_pid[col].iloc[0]
                    if col in static_encode_dict:
                        fill_val=static_encode_dict[col][fill_val]
                    empty_arr[:]=fill_val
                    df_merged[col]=empty_arr
                
                for feat in feats_to_analyze:
                    if feat+"_val" not in acc_dict:
                        acc_dict[feat+"_val"]=[]
                        acc_dict[feat+"_SHAP"]=[]
                    acc_dict[feat+"_val"].extend(list(df_merged[feat]))
                    acc_dict[feat+"_SHAP"].extend(list(df_merged["RawShap_"+feat]))
                    
            if configs["debug_mode"]:
                break

    if configs["create_overview_plot"]:
        LGBM_shap_vals=[]
        LSTM_shap_vals=[]
        for feat in feats_to_analyze:
            y_arr=np.array(acc_dict[feat+"_SHAP"])
            mean_SHAP_val=np.mean(np.abs(y_arr))

            if configs["lgbm_strategy"]=="SHAP":
                LGBM_shap_vals.append(mean_SHAP_val)
                print("Feat: {}, SHAP val: {}".format(feat,mean_SHAP_val))
            elif configs["lgbm_strategy"]=="corrupt":
                LGBM_shap_vals.append(lgbm_imp_dict[feat])

            LSTM_shap_vals.append(lstm_imp_dict[feat])

        LGBM_shap_vals=np.reshape(np.array(LGBM_shap_vals), (len(LGBM_shap_vals),1))
        LSTM_shap_vals=np.reshape(np.array(LSTM_shap_vals), (len(LSTM_shap_vals),1))

        # [0,1] normalization
        LGBM_shap_vals=LGBM_shap_vals/LGBM_shap_vals.max()
        LSTM_shap_vals=LSTM_shap_vals/LSTM_shap_vals.max()

        both_arr=np.hstack([LGBM_shap_vals,LSTM_shap_vals])
        mean_imp=np.mean(both_arr,axis=1)
        sort_idxs=np.argsort(mean_imp)[::-1]

        LGBM_shap_vals=LGBM_shap_vals[sort_idxs].flatten()
        LSTM_shap_vals=LSTM_shap_vals[sort_idxs].flatten()

        font_color = '#525252'
        facecolor = '#eaeaf2'
        color_red = '#fd625e'
        color_blue = '#01b8aa'
        title1="Feat. importance GBDT"
        title2="Feat. importance LSTM"

        feat_labels=list(map(lambda fname: translate_label(fname,name_dict=name_dict,backup_dict=backup_dict),
                             feats_to_analyze))
        feat_labels=list(np.array(feat_labels)[sort_idxs])

        fig,axes=plt.subplots(figsize=(10,5), ncols=2,
                              sharey=True)
        fig.tight_layout()

        axes[0].barh(feat_labels,LGBM_shap_vals,align="center",color=color_red,zorder=10)
        axes[0].set_title(title1,fontsize=18,pad=15,color=color_red)
        axes[1].barh(feat_labels,LSTM_shap_vals,align="center",color=color_blue,zorder=10)
        axes[1].set_title(title2,fontsize=18,pad=15,color=color_blue)
        axes[0].invert_xaxis()
        plt.gca().invert_yaxis()
        plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.35, right=0.95)

        plt.tight_layout()

        plt.savefig(os.path.join(configs["plot_path"],"feat_importance_both_models_{}.png".format(configs["lgbm_strategy"])),dpi=300)
        plt.savefig(os.path.join(configs["plot_path"],"feat_importance_both_models_{}.pdf".format(configs["lgbm_strategy"])))

    # Create individual plots
    specific_feats=["plain_vm31","plain_pm69"]
    
    for feat in specific_feats:
        x_arr=np.array(acc_dict[feat+"_val"])
        y_arr=np.array(acc_dict[feat+"_SHAP"])

        if feat=="plain_vm31":
            fin_x_arr=np.isfinite(x_arr)
            x_arr=x_arr[fin_x_arr]
            y_arr=y_arr[fin_x_arr]
        elif feat=="plain_pm69":
            x_arr[np.isnan(x_arr)]=0
            
        assert np.sum(np.isnan(x_arr))==0
        assert np.sum(np.isnan(y_arr))==0
        
        idx=nprand.choice(np.arange(len(x_arr)), configs["rsample"], replace=False)
        x_arr=x_arr[idx]
        y_arr=y_arr[idx]

        # Do not display extreme outliers
        upper_pct=np.percentile(x_arr,99)
        filter_arr=x_arr<=upper_pct
        x_arr=x_arr[filter_arr]
        y_arr=y_arr[filter_arr]

        h=sns.jointplot(x=x_arr,y=y_arr,kind="reg",height=8,joint_kws={"lowess": True, "marker": "x","scatter_kws": {"s": 1}, "line_kws": {"color": "gold"}})
        

        # Set x-axis to log-scale to avoid outliers
        #h.ax_joint.set_xscale('log')
        
        #h=sns.jointplot(x=x_arr,y=y_arr,kind="reg",joint_kws={"order": 3, "marker": "x","scatter_kws": {"s": 1}, "line_kws": {"color": "gold"}})        

        vlabel=translate_label(feat,name_dict=name_dict,backup_dict=backup_dict,current_value=True)
        
        h.set_axis_labels(vlabel, 'SHAP value', fontsize=12)
        for ax in (h.ax_joint, h.ax_marg_y):
            ax.axhline(0, color='crimson', ls='--', lw=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(configs["plot_path"], "{}.pdf".format(feat)))
        plt.savefig(os.path.join(configs["plot_path"], "{}.png".format(feat)),dpi=300)
        plt.clf()

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--pred_path", default="../../data/predictions/reduced",
                        help="Path from where to load the predictions")
    parser.add_argument("--feat_path", default="../../data/ml_input/hirid2_features/reduced",
                        help="Path from where to load feature values")
    
    parser.add_argument("--mid_dict", default="../../data/misc/mid2string_v6.npy")
    
    parser.add_argument("--static_path", default="../../data/imputed/noimpute_hirid2/reduced/temporal_1/static.h5")
    
    parser.add_argument("--split_path", default="../../data/exp_design/temp_splits_hirid2.pickle", help="Split descriptor")

    parser.add_argument("--lstm_imp_path", default="../../data/evaluation/lstm_feat_importance.csv", help="Feature importance of LSTM to load")

    parser.add_argument("--full_data_path", default="../../data/tests/hirid2_all_variables.pkl",
                        help="Data path where we can perform the corruption experiment")

    # Output paths
    parser.add_argument("--plot_path", default="../../data/plots/introspection",
                         help="Plotting folder")    

    # Arguments
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Process one batch")
    parser.add_argument("--rsample", type=int, default=50000, help="Random samples to draw before plotting")
    parser.add_argument("--small_sample", default=False, action="store_true", help="Small sample?")

    parser.add_argument("--n_replicates", type=int, default=5, help="Number of replicates in corruption experiment")

    parser.add_argument("--create_overview_plot", default=True, action="store_true", help="Should overview feature plot be created?")
    parser.add_argument("--lgbm_strategy", default="corrupt", help="Which strategy should be used for LGBM feature importance?")

    parser.add_argument("--worst_pred_analysis", default=False, action="store_true", help="Perform a worst prediction analysis")
    
    parser.add_argument("--model_config", default="Label_hirid_merged_24h_deleted_4h_WorseStateFromZeroEVAL0To48Hours_simple_features_lightgbm", help="Model to analyze")    

    configs=vars(parser.parse_args())

    configs["SPLITS"]=[("temporal_1","temporal_1")]

    # Kidney failure
    configs["FEATS_TO_ANALYZE"]=["RelDatetime","Emergency",
                                 "plain_vm21","plain_vm24",
                                 "plain_vm31","plain_pm35",
                                 "plain_pm43","plain_vm65",
                                 "plain_pm69","plain_pm73",
                                 "plain_pm86","plain_pm92",
                                 "plain_pm93","plain_pm94",
                                 "plain_pm95","plain_pm101",
                                 "plain_pm104","plain_pm109",
                                 "plain_vm131","plain_vm154",
                                 "plain_vm156","plain_vm162",
                                 "plain_vm176","plain_vm226",
                                 "plain_vm275","plain_vm276",
                                 "plain_pm290","plain_vm313"]

    execute(configs)
