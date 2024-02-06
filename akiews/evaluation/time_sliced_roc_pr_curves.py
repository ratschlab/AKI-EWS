''' Plot a time-slice based ROC/PR plot using 
    a set of saved curves to the file-system'''

import argparse
import os
import os.path
import ipdb
import pickle
import sys

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use("PDF")
mpl.rcParams["pdf.fonttype"]=42
mpl.rcParams["agg.path.chunksize"]=10000
import matplotlib.pyplot as plt
import scipy
import sklearn.metrics as skmetrics

def execute(configs):

    with open(configs["result_path"],'rb') as fp:
        obj=pickle.load(fp)
        roc_curves=obj["roc_curves"]
        pr_curves=obj["pr_curves"]

        ax=plt.gca()
        ax.set_rasterization_zorder(1)
        plt.text(0.07,0.40,"AUROC",color='k')

        for database,label_key,split_key,curve_label,color_key,score_y in configs["CURVE_LABELS"]:
            curves=roc_curves[(database,label_key,split_key)]
            main_fpr=curves["fpr"]
            main_tpr=curves["tpr"]
            sort_arr=np.argsort(main_fpr)
            main_fpr=main_fpr[sort_arr]
            main_tpr=main_tpr[sort_arr]
            tprs=[]
            auroc_main=skmetrics.auc(main_fpr,main_tpr)
            auroc_scores_var=[]
            for var_split in [split_key]:
                var_curve=roc_curves[(database,label_key,var_split)]
                var_fpr=var_curve["fpr"]
                var_tpr=var_curve["tpr"]
                sort_arr=np.argsort(var_fpr)
                var_fpr=var_fpr[sort_arr]
                var_tpr=var_tpr[sort_arr]
                tprs.append(scipy.interp(main_fpr,var_fpr,var_tpr))
                auroc_scores_var.append(skmetrics.auc(var_fpr,var_tpr))
            std_tprs=np.std(tprs,axis=0)
            tprs_upper=np.minimum(main_tpr+std_tprs,1)
            tprs_lower=np.maximum(main_tpr-std_tprs,0)
            plt.plot(curves["fpr"],curves["tpr"],rasterized=True,zorder=0,color=color_key,label=curve_label)
            plt.fill_between(main_fpr,tprs_lower,tprs_upper,alpha=0.2,rasterized=True,zorder=0,color=color_key)
            auroc_stdev=np.std(auroc_scores_var)
            plt.text(0.07,score_y,r'{:.3f}$\pm${:.3f}'.format(auroc_main,auroc_stdev),color=color_key)

        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.plot([0,1],[0,1],color="grey",lw=0.5,linestyle="--",rasterized=True,zorder=0)

        plt.legend(loc="lower right")
        ax=plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_aspect(1.0)
        ax.grid(which="both",lw=0.1)
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.tight_layout()
        plt.savefig(os.path.join(configs["plot_path"],"roc_main_{}.pdf".format(configs["plot_name"])))
        plt.savefig(os.path.join(configs["plot_path"],"roc_main_{}.png".format(configs["plot_name"])))
        plt.clf()

        # Create PR curve
        ax=plt.gca()
        ax.set_rasterization_zorder(1)
        plt.text(0.05,0.40,"AUPRC",color='k')

        for database,label_key,split_key,curve_label,color_key,score_y in configs["CURVE_LABELS"]:
            curves=pr_curves[(database,label_key,split_key)]
            main_rec=curves["recalls"]
            main_prec=curves["precs"]
            sort_arr=np.argsort(main_rec)
            main_rec=main_rec[sort_arr]
            main_prec=main_prec[sort_arr]
            precs=[]
            auprc_main=skmetrics.auc(main_rec,main_prec)
            auprc_scores_var=[]
            for var_split in [split_key]:
                var_curve=pr_curves[(database,label_key,var_split)]
                var_rec=var_curve["recalls"]
                var_prec=var_curve["precs"]
                sort_arr=np.argsort(var_rec)
                var_rec=var_rec[sort_arr]
                var_prec=var_prec[sort_arr]
                precs.append(scipy.interp(main_rec,var_rec,var_prec))
                auprc_scores_var.append(skmetrics.auc(var_rec,var_prec))
            std_precs=np.std(precs,axis=0)
            precs_upper=np.minimum(main_prec+std_precs,1)
            precs_lower=np.maximum(main_prec-std_precs,0)
            plt.plot(main_rec,main_prec,rasterized=True,zorder=0,color=color_key,label=curve_label)
            plt.fill_between(main_rec,precs_lower,precs_upper,alpha=0.2,rasterized=True,zorder=0,color=color_key)
            auprc_stdev=np.std(auprc_scores_var)
            plt.text(0.05,score_y,r'{:.3f}$\pm${:.3f}'.format(auprc_main,auprc_stdev),color=color_key)
            print("PR score: {:.3f}".format(auprc_main))

        plt.xlim((0,1))
        plt.ylim((0,1))
        
        plt.legend(loc="upper right")
        ax=plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_aspect(1.0)
        ax.grid(which="both",lw=0.1)
        plt.xlabel("Alarm recall")
        plt.ylabel("Alarm precision")
        plt.tight_layout()
        plt.savefig(os.path.join(configs["plot_path"],"pr_main_{}.pdf".format(configs["plot_name"])))
        plt.savefig(os.path.join(configs["plot_path"],"pr_main_{}.png".format(configs["plot_name"])))
        plt.clf()        

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--result_path", default="../../data/evaluation/time_point_based/raw_results.pickle",
                        help="Path where result curves should be loaded")

    # Output paths
    parser.add_argument("--plot_path", default="../../data/plots/model_selection",
                        help="Path where plots should be saved")

    # Arguments
    parser.add_argument("--plot_name", default="ef_extval", help="Name for plot")
    
    configs=vars(parser.parse_args())

    # Curves that should be loaded
    configs["CURVE_LABELS"]=[("hirid",'Label_hirid_merged_24h_deleted_4h_WorseStateFromZeroEVAL0To48Hours_joint_model_var_union_simple_lightgbm',"temporal_1","Joint (simple f., d=24)","C0",0.20),
                             ("hirid",'Label_hirid_merged_24h_deleted_4h_WorseStateFromZeroEVAL0To48Hours_joint_model_var_union_complex_lightgbm',"temporal_1","Joint (complex f., d=24)","C1",0.15),
                             ("hirid",'Label_hirid_merged_24h_deleted_4h_WorseStateFromZero0To48Hours_combined_model_from_separate_var_union_calibrated_lightgbm',"temporal_1","Separate (simple f., d=24)","C2",0.10),
                             ("hirid",'Label_hirid_merged_24h_deleted_4h_WorseStateFromZero0To48Hours_combined_model_from_separate_var_union_calibrated_complex_lightgbm',"temporal_1","Separate (complex f., d=24)","C3",0.05)]
    
    execute(configs)
