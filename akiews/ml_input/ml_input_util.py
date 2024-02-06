''' ML input script'''

import psutil
import os
import os.path
import random
import timeit
import gc
import time
import sys
import argparse
import ipdb

import tables
import warnings
warnings.simplefilter('ignore', tables.NaturalNameWarning)

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as spstats
import numba

import akiews.utils.filesystem as mlhc_fs
import akiews.utils.array as mlhc_array
import akiews.utils.memory as mlhc_memory
import akiews.utils.io as mlhc_io


def linregress(x, y=None):
    """
    Calculate a regression line

    This computes a least-squares regression for two sets of measurements.

    !!COPY!! of scipy.stats.linregress that is faster for the measured setting
    of small vectors.

    Parameters
    ----------
    x, y : array_like
        two sets of measurements.  Both arrays should have the same length.
        If only x is given (and y=None), then it must be a two-dimensional
        array where one dimension has length 2.  The two sets of measurements
        are then found by splitting the array along the length-2 dimension.

    Returns
    -------
    slope : float
        slope of the regression line
    intercept : float
        intercept of the regression line
    rvalue : float
        correlation coefficient

    """
    if y is None:  # x is a (2, N) or (N, 2) shaped array_like
        x = np.asarray(x)
        if x.shape[0] == 2:
            x, y = x
        elif x.shape[1] == 2:
            x, y = x.T
        else:
            msg = ("If only `x` is given as input, it has to be of shape "
                   "(2, N) or (N, 2), provided shape was %s" % str(x.shape))
            raise ValueError(msg)
    else:
        x = np.asarray(x)
        y = np.asarray(y)

    xmean = np.mean(x, None)
    ymean = np.mean(y, None)

    # average sum of squares:
    ssxm, ssxym, ssyxm, ssym = np.cov(x, y, bias=1).flat
    r_num = ssxym
    r_den = np.sqrt(ssxm * ssym)
    if r_den == 0.0:
        r = 0.0
    else:
        r = r_num / r_den
        # test for numerical error propagation
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0

    slope = r_num / ssxm
    intercept = ymean - slope * xmean

    return slope, intercept, r



def _trend_array_entire(orig_arr):
    ''' Generates a trend function over the entire stay'''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(1,orig_arr.size):
        search_arr=out_arr[:idx+1]
        out_arr[idx]=spstats.linregress(np.arange(search_arr.size),search_arr)[0]
    return out_arr

def _instable_array_8h(orig_arr,level_desc):
    ''' Generates an instability history feature across the last 8 hours'''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(orig_arr.size):
        if idx==0:
            out_arr[idx]=1 if orig_arr[idx]>=level_desc[0] and orig_arr[idx]<=level_desc[1] else 0
        else:
            subarr=orig_arr[max(0,idx-8*12):idx+1]
            subarr=subarr[np.isfinite(subarr)]
            if subarr.size>0:
                out_arr[idx]=np.sum((subarr>=level_desc[0]) & (subarr<=level_desc[1]))/subarr.size
    return out_arr

def _instable_array_entire(orig_arr,level_desc):
    ''' Generates an instability history feature across the entire stay'''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(orig_arr.size):
        subarr=orig_arr[:idx+1]
        subarr=subarr[np.isfinite(subarr)]
        if subarr.size>0:
            out_arr[idx]=np.sum((subarr>=level_desc[0]) & (subarr<=level_desc[1]))/subarr.size
    return out_arr

def _mean_array_entire(orig_arr):
    ''' Generates a mean function over the entire stay'''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(orig_arr.size):
        if idx==0:
            out_arr[idx]=orig_arr[idx]
        else:
            out_arr[idx]=np.nanmean(orig_arr[:idx+1])
    return out_arr

def _min_array_entire(orig_arr):
    ''' Generates a min function over the entire stay'''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(orig_arr.size):
        if idx==0:
            out_arr[idx]=orig_arr[idx]
        else:
            out_arr[idx]=orig_arr[:idx+1].min()
    return out_arr

def _max_array_entire(orig_arr):
    ''' Generates a max function over the entire stay'''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(orig_arr.size):
        if idx==0:
            out_arr[idx]=orig_arr[idx]
        else:
            out_arr[idx]=orig_arr[:idx+1].max()
    return out_arr

def _std_array_entire(orig_arr):
    ''' Generates a std function over the entire stay'''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(1,orig_arr.size):
        out_arr[idx]=np.std(orig_arr[:idx+1])
    return out_arr

@numba.jit(nopython=True)
def _median_iqr_array(orig_arr, back_hor):
    ''' Generates a median/iqr function with a back horizon'''
    out_arr_median=np.zeros_like(orig_arr)
    out_arr_iqr=np.zeros_like(orig_arr)
    for idx in range(0,orig_arr.size):
        search_arr=orig_arr[max(0,idx-int(back_hor*12)):idx+1]
        perc=np.nanpercentile(search_arr,[25,50,75])
        out_arr_median[idx]=perc[1]
        out_arr_iqr[idx]=perc[2]-perc[0]
    return (out_arr_median,out_arr_iqr)

def _mode_array(orig_arr, back_hor):
    ''' Generates a mode function with a back horizon'''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(0,orig_arr.size):
        search_arr=orig_arr[max(0,idx-int(back_hor*12)):idx+1]
        out_arr[idx]=spstats.mode(search_arr,axis=None,nan_policy="omit")[0][0]
    return out_arr

def _trend_array(orig_arr, back_hor):
    ''' Generates a trend function with a back horizon,
        if the horizon is not full, emit NAN '''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(0,orig_arr.size):
        search_arr=orig_arr[max(0,idx-int(back_hor*12)):idx+1]
        search_arr=np.isfinite(search_arr)
        if search_arr.size>0:
            out_arr[idx]=linregress(np.arange(search_arr.size),search_arr)[0]
    return out_arr

@numba.jit(nopython=True)
def _mean_array(orig_arr, back_hor):
    ''' Generates a mean function with a back horizon '''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(0,orig_arr.size):
        search_arr=orig_arr[max(0,idx-int(back_hor*12)):idx+1]
        out_arr[idx]=np.nanmean(search_arr)
    return out_arr    

@numba.jit(nopython=True)
def _meas_density_array(orig_arr, back_hor):
    ''' Generates a measure density function with a back horizon'''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(0,orig_arr.size):
        search_arr=orig_arr[max(0,idx-int(back_hor*12)):idx+1]
        out_arr[idx]=(search_arr[-1]-search_arr[0])/search_arr.size
    return out_arr

def _min_array(orig_arr, back_hor):
    ''' Generates a min function with a back horizon '''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(orig_arr.size):
        if idx==0:
            out_arr[idx]=orig_arr[idx]
        else:
            out_arr[idx]=orig_arr[max(0,idx-back_hor*12):idx+1].min()
    return out_arr

def _max_array(orig_arr, back_hor):
    ''' Generates a max function with a back horizon '''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(orig_arr.size):
        if idx==0:
            out_arr[idx]=orig_arr[idx]
        else:
            out_arr[idx]=orig_arr[max(0,idx-back_hor*12):idx+1].max()
    return out_arr

def _std_array(orig_arr, back_hor):
    ''' Generates a std function with a back horizon '''
    out_arr=np.zeros_like(orig_arr)
    for idx in range(1,orig_arr.size):
        out_arr[idx]=np.std(orig_arr[max(0,idx-back_hor*12):idx+1])
    return out_arr



def gen_features_df(df_pat,df_label_pat,df_ep_pat,pid=None,configs=None,imp_vars=None,
                    interval_dict=None, hirid_dict=None, bowen_hmm=None,
                    bowen_hmm_probs=None):
    ''' 
    Transforms the imputed data frame to a data frame with final features
    '''

    sample_idx=0
    start_ts=configs["min_history_hours"]*3600.0-configs["impute_grid_unit_secs"]
    samples_per_hour=int(3600.0/configs["impute_grid_unit_secs"])
    all_cols=list(filter(lambda col: "IMPUTE_STATUS" not in col, df_pat.columns.values.tolist()))
    label_cols=list(filter(lambda col: col not in ["AbsDatetime", "RelDatetime", "PatientID"], df_label_pat.columns.values.tolist()))
    rel_dt_col=df_pat["RelDatetime"]
    abs_dt_col=df_pat["AbsDatetime"]
    pid_col=df_pat["PatientID"]
    SYMBOLIC_MAX=configs["symbolic_max_time_ms"]
    current_value_cols=list(filter(lambda item: "_IMPUTED_" not in item and ("vm" in item or "pm" in item), all_cols))

    if df_pat.shape[0]==0 or df_label_pat.shape[0]==0:
        print("WARNING: Patient without information in labels or imputed data",flush=True)
        return (None,None)

    assert(df_pat.shape[0]==df_label_pat.shape[0])
    ts_col=np.array(df_pat["RelDatetime"])
    label_cols_dict={}

    for lcol in label_cols:
        label_cols_dict["{}".format(lcol)]=np.array(df_label_pat[lcol])

    for col in current_value_cols:
        arr_col=np.array(df_pat[col])
        if np.sum(np.isnan(arr_col))>0:
            assert(False,"Cannot have NAN in input to feature generation")

    X_df_dict={}
    X_df_dict["RelDatetime"]=rel_dt_col
    X_df_dict["AbsDatetime"]=abs_dt_col
    X_df_dict["PatientID"]=pid_col

    cum_fit_times={"meas_density": 0,
                   "median_iqr": 0,
                   "trend": 0,
                   "mean": 0,
                   "mode": 0}

    # Special treatment of endpoint-attached columns
    for col in configs["EP_COLS"]:
        col_id=col.replace("_","")
        X_df_dict["plain_{}".format(col_id)]=df_ep_pat[col]

    # Attach Bowen's columns as they are
    if bowen_hmm is not None:
        for hmm_col in bowen_hmm.columns.values.tolist():
            if bowen_hmm.shape[0]>0:
                X_df_dict["bowen_samples_{}".format(hmm_col)]=np.array(bowen_hmm[hmm_col])
            else:
                nan_vect=np.zeros(len(pid_col))
                nan_vect[:]=np.nan
                X_df_dict["bowen_samples_{}".format(hmm_col)]=nan_vect

    if bowen_hmm_probs is not None:
        for hmm_col in bowen_hmm_probs.columns.values.tolist():
            if bowen_hmm_probs.shape[0]>0:
                X_df_dict["bowen_probs_{}".format(hmm_col)]=np.array(bowen_hmm_probs[hmm_col])
            else:
                nan_vect=np.zeros(len(pid_col))
                nan_vect[:]=np.nan
                X_df_dict["bowen_probs_{}".format(hmm_col)]=nan_vect
    
    # Columns from endpoint frame
    for cix,col in enumerate(current_value_cols):
        col_key=col.strip()
        datatype=hirid_dict[(col_key,"Datatype")]
        exp_values_per_hour=3600/interval_dict[col_key] if col_key in interval_dict else 30.0
        if np.isnan(exp_values_per_hour):
            exp_values_per_hour=30.0

        col_arr=np.array(df_pat[col]) 
        cum_cnt_arr=np.array(df_pat["{}_IMPUTED_STATUS_CUM_COUNT".format(col)]) 
        
        # Current value
        X_df_dict["plain_{}".format(col)]=col_arr

        # Time to last measurement features            
        im_array=np.copy(cum_cnt_arr)
        im_array[im_array==-1.0]=SYMBOLIC_MAX
        X_df_dict["{}_time_to_last_ms".format(col)]=im_array

        for pct,pct_hours in configs["HIST_LENGTH_PCT"]:
            exp_values_horizon=int(exp_values_per_hour*pct_hours)

            # Do not produce statistics on horizons with less than 5 values on expectation,
            # do not skip the last horizon.
            if exp_values_horizon<5 and not pct==80:
                continue

            hour_key=int(pct_hours)
            
            # Measure density feature
            #t_begin=timeit.default_timer()
            X_df_dict["{}_meas_density_H{}".format(col,hour_key)]=_meas_density_array(cum_cnt_arr,pct_hours)
            #t_end=timeit.default_timer()
            #cum_fit_times["meas_density"]+=t_end-t_begin

            # Median/IQR features
            if datatype=="Ordinal":
                #t_begin=timeit.default_timer()
                med,iqr=_median_iqr_array(col_arr,pct_hours)                
                X_df_dict["{}_median_H{}".format(col,hour_key)]=med
                X_df_dict["{}_iqr_H{}".format(col,hour_key)]=iqr
                #t_end=timeit.default_timer()
                #cum_fit_times["median_iqr"]+=t_end-t_begin

            # Trend feature
            if datatype=="Ordinal":
                #t_begin=timeit.default_timer()
                X_df_dict["{}_trend_H{}".format(col,hour_key)]=_trend_array(col_arr,pct_hours)
                #t_end=timeit.default_timer()
                #cum_fit_times["trend"]+=t_end-t_begin

            # Mean feature
            if datatype=="Binary":
                #t_begin=timeit.default_timer()
                X_df_dict["{}_mean_H{}".format(col,hour_key)]=_mean_array(col_arr,pct_hours)
                #t_end=timeit.default_timer()
                #cum_fit_times["mean"]+=t_end-t_begin

            # Mode feature
            if datatype=="Categorical":
                #t_begin=timeit.default_timer()
                X_df_dict["{}_mode_H{}".format(col,hour_key)]=_mode_array(col_arr,pct_hours)
                #t_end=timeit.default_timer()
                #cum_fit_times["mode"]+=t_end-t_begin

            # Only compute one horizon for non-important variables
            if not col in imp_vars:
                break

    # Generate complex features in a separate pass
    for col in imp_vars:

        # Do not try to generate complex features for static variables
        if "static_" in col:
            continue

        if col in configs["EP_COLS"]:
            col_arr=np.array(df_ep_pat[col])
        else:

            if col not in df_pat.columns.values.tolist():
                ipdb.set_trace()
            
            col_arr=np.array(df_pat[col])

        col_id = col.replace("_","") if col in configs["EP_COLS"] else col

        if col in configs["SEVERITY_LEVELS"]:
            levels=configs["SEVERITY_LEVELS"][col]
            for lidx,level_desc in enumerate(levels):
                X_df_dict["{}_instable_l{}_8h".format(col_id,lidx+1)]=_instable_array_8h(col_arr,level_desc)
                X_df_dict["{}_instable_l{}_entire".format(col_id,lidx+1)]=_instable_array_entire(col_arr,level_desc)

    gc.collect()

    dict_ks=X_df_dict.keys()
    for k in dict_ks:
        if len(X_df_dict[k])==0:
            fill_arr=np.zeros_like(ts_col)
            fill_arr[:]=np.nan
            X_df_dict[k]=fill_arr
            
    X_df=pd.DataFrame(X_df_dict)

    y_df_dict={}
    y_df_dict["RelDatetime"]=rel_dt_col
    y_df_dict["AbsDatetime"]=abs_dt_col
    y_df_dict["PatientID"]=pid_col

    for lcol in label_cols_dict.keys():
        y_df_dict["Label_{}".format(lcol)]=label_cols_dict[lcol]

    y_df=pd.DataFrame(y_df_dict)

    return (X_df, y_df)

