''' ML fitting utilities'''

import os
import os.path
import sys
import random
import timeit
import csv
import warnings
import argparse
import ipdb
import gc
import glob
import itertools
import json
import pickle
import psutil
import time
import gc
import gin
import math
import warnings

import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt

import scipy as sp
import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
import sklearn.feature_selection as skfselect
import sklearn.preprocessing as skpproc
import sklearn.impute as skimpute
import sklearn.dummy as skdummy
import sklearn.svm as sksvm
import sklearn.compose as skcompose
import sklearn.linear_model as sklm
import sklearn.ensemble as skens
import sklearn.neural_network as sknn
import shap
import numpy.random as nprand
import lightgbm
import xgboost
import catboost
import flaml

import akiews.utils.filesystem as mlhc_fs
import akiews.utils.memory as mlhc_memory
import akiews.utils.array as mlhc_array
import akiews.utils.io as mlhc_io

def predict(X_df, y_df=None, pid=None, df_static=None, ml_model=None, feat_order=None,
            std_arr=None, std_eps=None,configs=None, impute_model=None, low_var_model=None,
            feat_order_orig=None, encode_scale_model=None, return_X_y_inputs=False):
    ''' Predicts the labels of a Pandas data-frame. Allows to pass the data-frame with the true labels 
        so it can be appended on the output data-frame '''

    # Determine which label to use when deciding when to make a prediction
    target_label_key = configs["eval_label_key"] if configs["eval_label_key"] is not None else configs["label_key"]
    
    abs_dt=X_df["AbsDatetime"]
    rel_dt=X_df["RelDatetime"]

    if configs["custom_eval_metric"]=="multiclass":
        pred_all_vect=mlhc_array.empty_nan((abs_dt.size,3))
    else:
        pred_all_vect=mlhc_array.empty_nan(abs_dt.size)
        
    pred_all_vect_labels=mlhc_array.empty_nan(abs_dt.size)

    # Should only be true for multi-class tasks
    pred_everywhere=configs["pred_everywhere"]

    # Make predictions everywhere?
    if pred_everywhere:
        X_input=X_df
    else:

        try:
            X_input=X_df[~y_df[target_label_key].isnull()]
        except KeyError:
            target_label_key=target_label_key.replace("_hirid_","_mimic_")
            X_input=X_df[~y_df[target_label_key].isnull()]

    # No valid samples to use, hence we can not predict anything
    if X_input.shape[0]==0:
        return (None,None,None,None)

    df_static=pd.concat([df_static]*X_input.shape[0])
    df_full=pd.DataFrame(np.column_stack([X_input,df_static]),columns=X_input.columns.append(df_static.columns))

    ret_abs_ts=df_full["AbsDatetime"].values.tolist()
    
    X_input=df_full[feat_order_orig]

    if "Sex" in X_input.columns.values.tolist():
        X_input["Sex"]=X_input["Sex"].replace(configs["str_to_int_sex"])
    
    X_full=np.array(X_input)

    if configs["mean_impute_nans"]:
        X_full=impute_model.transform(X_full)  
    
    if configs["filter_low_variance"]:
        X_full=low_var_model.transform(X_full)

    if configs["scale_encode_data"] or configs["encode_data"]:
        X_full=encode_scale_model.transform(X_full)

    prob_output=ml_model.predict_proba(X_full)

    if configs["custom_eval_metric"]=="multiclass":
        pred_vect=prob_output
    else:
        pred_vect=prob_output[:,1]
        
    pred_vect_labels=ml_model.predict(X_full)
    df_out_dict={}        

    if configs["ml_model"]=="lightgbm":

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer=shap.TreeExplainer(ml_model)
            shap_values=explainer.shap_values(X_full)[1] # SHAP library changing encoding, [1] is the contributions to positive class

        assert(shap_values.shape[1]==len(feat_order))

        for cidx,cname in enumerate(feat_order):
            temp_vect=mlhc_array.empty_nan(abs_dt.size)

            if pred_everywhere:
                temp_vect=np.copy(shap_values[:,cidx])
            else:
                temp_vect[~y_df[target_label_key].isnull()]=shap_values[:,cidx]
                
            df_out_dict["RawShap_{}".format(cname)]=temp_vect

    if pred_everywhere:
        pred_all_vect=np.copy(pred_vect)
        pred_all_vect_labels=np.copy(pred_vect_labels)
    else:
        pred_all_vect[~y_df[target_label_key].isnull()]=pred_vect
        pred_all_vect_labels[~y_df[target_label_key].isnull()]=pred_vect_labels
        
    pid_vect=mlhc_array.value_empty(abs_dt.size, pid)
    cols_y=sorted(y_df.columns.values.tolist())
    y_vect=np.array(y_df[target_label_key])

    y_vect_finites=np.array(y_df[~y_df[target_label_key].isnull()][target_label_key])

    df_out_dict["PatientID"]=pid_vect

    if configs["custom_eval_metric"]=="multiclass":
        for class_id in range(3):
            df_out_dict["PredScore_{}".format(class_id)]=pred_all_vect[:,class_id]
    else:
        df_out_dict["PredScore"]=pred_all_vect
        
    df_out_dict["PredLabel"]=pred_all_vect_labels
    df_out_dict["TrueLabel"]=y_vect
    df_out_dict["AbsDatetime"]=abs_dt
    df_out_dict["RelDatetime"]=rel_dt
    df_out=pd.DataFrame(df_out_dict)

    if return_X_y_inputs:
        return (df_out,X_full,y_vect_finites,ret_abs_ts)
    else:
        return df_out


def construct_ml_model(model_type, n_estimators=None, num_leaves=None, learning_rate=None,
                       colsample_bytree=None, rowsample_bytree=None,
                       alpha=None, hidden_layer_size=None, configs=None):
    ''' Factory of a machine learning model with hyperparameters'''

    if model_type=="tree":
        ml_model=lightgbm.LGBMClassifier(n_estimators=1, n_jobs=1, random_state=2021,
                                         num_leaves=num_leaves, learning_rate=learning_rate, 
                                         colsample_bytree=1, verbose=-1,silent=True,metric="custom", class_weight=configs["class_weight"],
                                         is_unbalance=configs["lgbm_is_unbalanced"],objective=configs["clf_objective"],
                                         subsample_freq=0, subsample=1.0, max_depth=int(math.log2(num_leaves)),
                                         subsample_for_bin=1000000,min_child_samples=configs["lgbm_min_child_samples"],
                                         max_cat_to_onehot=100, cat_smooth=0.0,cat_l2=0.0)
    elif model_type=="extratrees":
        ml_model=skens.ExtraTreesClassifier(random_state=2021)
    elif model_type=="rforest":
        ml_model=skens.RandomForestClassifier(random_state=2021) 
    elif model_type=="logreg":
        ml_model=sklm.SGDClassifier(loss="log",penalty="l2",alpha=alpha,random_state=2021,learning_rate="optimal",class_weight="balanced")
    elif model_type=="mlp":
        ml_model=sknn.MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), activation="relu", solver="adam",
                                    alpha=alpha, batch_size=128,random_state=2021)

    elif model_type=="lightgbm":
        ml_model=lightgbm.LGBMClassifier(n_estimators=n_estimators,n_jobs=1,random_state=2021,
                                         num_leaves=num_leaves, learning_rate=learning_rate,
                                         colsample_bytree=colsample_bytree,verbose=-1,silent=True, class_weight=configs["class_weight"],
                                         is_unbalance=configs["lgbm_is_unbalanced"],metric="custom",objective=configs["clf_objective"],
                                         subsample_freq=1, subsample=rowsample_bytree, max_depth=int(math.log2(num_leaves)),
                                         subsample_for_bin=1000000,min_child_samples=configs["lgbm_min_child_samples"],
                                         max_cat_to_onehot=100,cat_smooth=0.0,cat_l2=0.0)
    elif model_type=="lgbm_flaml":
        ml_model=flaml.AutoML()
        #ml_model=flaml.default.LGBMClassifier(n_estimators=5000,n_jobs=1,random_state=2023,verbose=-1,
        #                                      silent=True, metric="custom",objective=configs["clf_objective"])

    return ml_model



def get_validation_scores(full_X_val, full_y_val, X_col_names=None, X_cat_cols=None, ml_model=None, configs=None):
    ''' Returns dictionary with scores on validation set'''
    der_X_val=full_X_val

    if configs["clf_objective"]=="binary":
        pred_vect=ml_model.predict_proba(der_X_val)[:,1]
        auroc_score=skmetrics.roc_auc_score(full_y_val,pred_vect)
        auprc_score=skmetrics.average_precision_score(full_y_val,pred_vect)
        return {"auroc": auroc_score, "auprc": auprc_score}
    elif configs["clf_objective"]=="multiclass":
        pred_vect=ml_model.predict_proba(der_X_val)
        pred_vect_labels=ml_model.predict(der_X_val)
        return {"acc": skmetrics.accuracy_score(full_y_val,pred_vect_labels),
                "auroc": skmetrics.roc_auc_score(full_y_val,pred_vect,multi_class="ovo")}
    

def get_train_scores(full_X,full_y, X_col_names=None, X_cat_cols=None, ml_model=None, configs=None):
    ''' Returns dictionary with scores on training set'''

    if configs["clf_objective"]=="binary":
        pred_vect=ml_model.predict_proba(full_X)[:,1]
        auroc_score=skmetrics.roc_auc_score(full_y,pred_vect)
        auprc_score=skmetrics.average_precision_score(full_y, pred_vect)
        return {"auroc": auroc_score, "auprc": auprc_score}
    elif configs["clf_objective"]=="multiclass":
        pred_vect=ml_model.predict_proba(full_X)
        pred_vect_labels=ml_model.predict(full_X)
        return {"acc": skmetrics.accuracy_score(full_y,pred_vect_labels),
                "auroc": skmetrics.roc_auc_score(full_y,pred_vect,multi_class="ovo")}
    else:
        print("ERROR: Wrong metric...")
        sys.exit(1)
        

def plot_tree(ml_model=None):
    ''' Plots a tree to a new MPL figure'''
    lightgbm.plot_tree(ml_model, tree_index=0, show_info=["split_gain"])


def feature_importances(full_X_val=None, full_y_val=None, ml_model=None):
    ''' Returns a list of feature importance representations from this model. Wraps the importances 
        as a Pandas data-frame table to re-associate with the variable names.'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer=shap.TreeExplainer(ml_model)
        raw_shap=explainer.shap_values(full_X_val)
        all_mats=[]
        for class_mat in raw_shap:
            global_class=np.mean(np.absolute(class_mat),axis=0)
            all_mats.append(global_class)
            
    global_vals=np.mean(all_mats,axis=0)
    return global_vals


def get_evaluation_trace(ml_model=None,configs=None):
    ''' Gets evaluation trace of the model'''
    if configs["custom_eval_metric"]=="auprc":
        return ml_model.evals_result_["valid_0"]["custom_auprc"]
    elif configs["custom_eval_metric"]=="auroc":
        return ml_model.evals_result_["valid_0"]["custom_auroc"]
    elif configs["custom_eval_metric"]=="multiclass":
        return ml_model.evals_result_["valid_0"]["custom_auroc"]

def custom_auprc_metric(y_true,y_pred):
    ''' Custom metric used for early stopping (AUPRC)'''
    return ("custom_auprc", skmetrics.average_precision_score(y_true,y_pred), True)

def custom_auroc_metric(y_true,y_pred):
    ''' Custom metric for early stopping (AUROC)'''
    return ("custom_auroc", skmetrics.roc_auc_score(y_true,y_pred), True)

def custom_mc_auroc_metric(y_true, y_pred):
    y_pred_multi=y_pred.reshape((y_true.size,-1),order="F")
    y_pred_multi_sum=np.sum(y_pred_multi,axis=1)
    y_pred_multi=y_pred_multi/y_pred_multi_sum[:,np.newaxis]
    metric_out=skmetrics.roc_auc_score(y_true,y_pred_multi,multi_class="ovo")
    return ("custom_auroc",metric_out,True)

def custom_bacc_metric(y_true,y_pred):
    y_pred_multi=y_pred.reshape((y_true.size,-1),order="F")
    y_pred_labels=np.argmax(y_pred_multi,axis=1).astype(np.int)
    metric_out=skmetrics.balanced_accuracy_score(y_true,y_pred_labels)
    return ("custom_bacc", metric_out, True)
    

