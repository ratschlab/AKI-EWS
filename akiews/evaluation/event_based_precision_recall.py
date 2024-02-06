import os
import gc
import sys
import h5py
import json
import pickle 
import pandas as pd
import numpy as np
from sklearn import metrics 
from functools import reduce

import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("tab10")

import utils_data
    
import ipdb

use_cols = ["PatientID", "Datetime", "Prediction"]

def read_vars_imputed(datapath, batch_name):

    df = []
    df_iter = pd.read_hdf(os.path.join(datapath, batch_name), chunksize=10**5, mode="r", usecols=["PatientID", "AbsDatetime", "RelDatetime", "pm69"])
    for tmp in df_iter:
        pids_drug = tmp[tmp.pm69>0].PatientID.unique()
        if len(pids_drug) > 0:
            df.append(tmp[tmp.PatientID.isin(pids_drug)])
        gc.collect()
    df = pd.concat(df).reset_index(drop=True)
    df.rename(columns={"AbsDatetime": "Datetime"}, inplace=True)
    df.sort_values(["PatientID", "Datetime"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

def read_tg_pd(datapath, lst_batches, col_predscore=None):
    """
    Reading prediction scores from Thomas' prediction folder
    """    
    df = []
    for i, b in enumerate(lst_batches):
        df_b = pd.read_hdf(os.path.join(datapath, b))
        df_b = df_b.rename(columns={"geq1_48": "Label", 
                                    "geq0_48": "Label", 
                                    "point_est_set": "Dataset"})
        if col_predscore is None:
            # df_b = df_b.rename(columns={"point_est": "Prediction"})
            df_b = df_b.rename(columns={"raw_point_est": "Prediction"})
        else:
            df_b = df_b.rename(columns={col_predscore: "Prediction"})
        df_b = df_b.sort_values(["PatientID", "Datetime"]).reset_index(drop=True)
        for pid in df_b.PatientID.unique():
            df_p = df_b[df_b.PatientID==pid]
            # assert(df_p.Datetime.duplicated(keep=False).sum()==0)
            if df_p.Datetime.duplicated(keep=False).sum() > 0:
                df_b = df_b.drop(df_p.index[df_p.Datetime.duplicated(keep="last")])
        df.append(df_b[["PatientID", "Datetime", "Prediction", "Dataset"]])
        gc.collect()

        sys.stdout.write("Reading Thomas' prediction results: ")
        sys.stdout.write("Batch %02d \r"%(i+1))
        sys.stdout.flush()
    df = pd.concat(df, axis=0).reset_index(drop=True)
    print("\nFinished reading Thomas' results.")
    return df


def read_clinical_baseline(datapath, lst_batches, spo2=None, fio2=None):
    """
    Reading clinical baseline prediction
    """            
    usecols = ["PatientID", "AbsDatetime", "RelDatetime", "vm20", "vm58"]
    df = []
    for i, b in enumerate(lst_batches):
        batch_path = os.path.join(datapath, b)
        df_b = pd.read_hdf(batch_path, chunksize=10**5, mode="r")
        for tmp in df_b:
            df.append(tmp[usecols])
            gc.collect()
        sys.stdout.write("Reading clinical baseline results: ")
        sys.stdout.write("Batch %02d \r"%(i+1))
        sys.stdout.flush()
    df = pd.concat(df)

    df.loc[:,"Prediction"] = (df.vm20<spo2)|(df.vm58>fio2)
    df = df.rename(columns={"AbsDatetime": "Datetime"})
    df = df.sort_values(["PatientID", "Datetime"]).reset_index(drop=True)
    print("Finished reading clinical predictions.")
    return df[["PatientID", "Datetime", "Prediction"]]


def get_only_urine_ep(df_ep):
    gap2merge_m=1500
    df_ep.loc[:,"geq1_urine"] = df_ep["1.u"].copy()
    for pid in df_ep.PatientID.unique():
        tmp = df_ep[df_ep.PatientID==pid]
        if tmp.geq1.sum()==0:
            continue
        if tmp["1.b"].sum()==0 or tmp["1.i"].sum()==0:
            continue
        tdiff = tmp[tmp['geq1_urine']==1].AbsDatetime.diff()
        if np.sum((tdiff>np.timedelta64(5,"m"))&(tdiff<np.timedelta64(gap2merge_m,"m"))) > 0:
            loc_all_gap_end = tdiff[(tdiff>np.timedelta64(5,"m"))].index
            loc_all_gap_start = np.array([tmp[tmp.AbsDatetime==(tmp.loc[l].AbsDatetime-tdiff.loc[l])].index[0] for l in loc_all_gap_end])
            
            loc_short_gap_end = tdiff[(tdiff>np.timedelta64(5,"m"))&(tdiff<np.timedelta64(gap2merge_m,"m"))].index
            idx_short_gap_end = np.where(np.isin(loc_all_gap_end, loc_short_gap_end))[0]
            for i in idx_short_gap_end:
                df_ep.loc[loc_all_gap_start[i]:loc_all_gap_end[i],"geq1_urine"] = 1
    return df_ep

def get_only_creatinine_ep(df_ep):
    gap2merge_m=1500
    df_ep.loc[:,"geq1_creatinine"] = (df_ep["1.b"].astype(bool)|df_ep["1.i"].astype(bool))
    for pid in df_ep.PatientID.unique():
        tmp = df_ep[df_ep.PatientID==pid]
        if tmp.geq1.sum()==0:
            continue
        if tmp["1.u"].sum()==0:
            continue
        tdiff = tmp[tmp['geq1_creatinine']==1].AbsDatetime.diff()
        if np.sum((tdiff>np.timedelta64(5,"m"))&(tdiff<np.timedelta64(gap2merge_m,"m"))) > 0:
            loc_all_gap_end = tdiff[(tdiff>np.timedelta64(5,"m"))].index
            loc_all_gap_start = np.array([tmp[tmp.AbsDatetime==(tmp.loc[l].AbsDatetime-tdiff.loc[l])].index[0] for l in loc_all_gap_end])
            
            loc_short_gap_end = tdiff[(tdiff>np.timedelta64(5,"m"))&(tdiff<np.timedelta64(gap2merge_m,"m"))].index
            idx_short_gap_end = np.where(np.isin(loc_all_gap_end, loc_short_gap_end))[0]
            for i in idx_short_gap_end:
                df_ep.loc[loc_all_gap_start[i]:loc_all_gap_end[i],"geq1_urine"] = 1
    return df_ep

def read_ep_renal(datapath, pids, endpoint_name, no_urine=False, only_urine=False):
    """
    Read renal endpoints
    """
    df = []
    for f in os.listdir(datapath):
        if not( ".h5" in f or ".hdf5" in f):
            continue
        if "mimic" in datapath:
            try:
                df_b = pd.read_hdf(os.path.join(datapath, f), columns=["stay_id"])
                df_b = df_b.rename(columns={"stay_id": "PatientID"})
            except:
                continue
        else:
            try:
                df_b = pd.read_hdf(os.path.join(datapath, f), columns=["PatientID"])
            except:
                continue
        
        if len(set(pids)&set(df_b.PatientID))==0:
            del df_b
            gc.collect()
            continue
        
        df_b = pd.read_hdf(os.path.join(datapath, f))
        if "mimic" in datapath:
            df_b = df_b.rename(columns={"stay_id": "PatientID", "AbsDatetime": "Datetime"})
        df_b = df_b.drop(df_b.index[~df_b.PatientID.isin(pids)])
        gc.collect()

        if only_urine:
            df_b = df_b.rename(columns={"geq1_urine": "geq1",
                                        "geq2_urine": "geq2",
                                        "geq3_urine": "geq3"})
        elif no_urine:
            df_b = df_b.rename(columns={"geq1_creatinine": "geq1",
                                        "geq2_creatinine": "geq2",
                                        "geq3_creatinine": "geq3"})
        else:
            pass

        if endpoint_name == "012-3":
            df_b.loc[:,"Stable"] = (df_b.geq3 == False) & (df_b.geq2 == True)
            df_b.loc[:,"InEvent"] = (df_b.geq3 == True)
            df_b.loc[:,"Unknown"] = df_b.geq3.isnull() 
        elif endpoint_name == "01-23":
            df_b.loc[:,"Stable"] = (df_b.geq2 == False) & (df_b.geq1 == True)
            df_b.loc[:,"InEvent"] = (df_b.geq2 == True)
            df_b.loc[:,"Unknown"] = df_b.geq2.isnull()
        elif endpoint_name == "0-123":
            df_b.loc[:,"Stable"] = (df_b.geq1 == False)
            df_b.loc[:,"InEvent"] = (df_b.geq1 == True)
            df_b.loc[:,"Unknown"] = df_b.geq1.isnull()
        
        df.append(df_b)
    df = pd.concat(df, axis=0)
    df = df.sort_values(["PatientID", "Datetime"]).reset_index(drop=True)
    print("\nFinished reading endpoints.\n")
    return df[["PatientID", "Datetime", "Stable", "InEvent", "Unknown"]]


def read_ep_resp(datapath, lst_batches, endpoint_name, 
                 stable_status=None, unstable_status=None):
    """
    Read respiratory endpoints
    """
    df = []
    for i, b in enumerate(lst_batches):
        df_b = pd.read_hdf(os.path.join(datapath, b)).reset_index(drop=True)
        if endpoint_name=="resp_failure":
            str_stable = ["event_%s"%x for x in stable_status]
            str_fail = ["event_%s"%x for x in unstable_status]
            df_b.loc[:,"Stable"] = df_b.endpoint_status.isin(str_stable).values
            df_b.loc[:,"InEvent"] = df_b.endpoint_status.isin(str_fail).values
            
            status_of_interest = np.concatenate((stable_status, unstable_status))
            if len(np.unique(status_of_interest)) == 4:
                df_b.loc[:,"Unknown"] = (df_b.endpoint_status=="UNKNOWN").values
            else:
                unknown_status = set(["0","1","2","3"]) - set(status_of_interest)
                unknown_status = list(unknown_status) + ["UNKNOWN"]
                df_b.loc[:,"Unknown"] = df_b.endpoint_status.isin(unknown_status).values
                
        elif "extu" in endpoint_name and endpoint_name not in ["extu_failure", "extu_success"]:
            df_b.loc[:,"Stable"] = (df_b.readiness_ext==0).values
            df_b.loc[:,"InEvent"] = (df_b.readiness_ext==1).values
            df_b.loc[:,"Unknown"] = df_b.readiness_ext.isnull()
            
        elif "vent" in endpoint_name:
            df_b.loc[:,"Stable"] = (df_b.vent_period==0).values
            df_b.loc[:,"InEvent"] = (df_b.vent_period==1).values
            df_b.loc[:,"Unknown"] = df_b.vent_period.isnull()
            
        elif endpoint_name in ["extu_failure", "extu_success"]:
            for pid in df_b[df_b.ext_failure.notnull()].PatientID.unique():
                df_p = df_b[df_b.PatientID==pid]
                for idx in df_p[df_p.ext_failure.notnull()].index:

                    # create a short event of 30 minutes after the extubation 
                    # success or failure event
                    dt_extu =  df_p.loc[idx,"AbsDatetime"]
                    dt_extu_end = dt_extu + np.timedelta64(30,"m")
                    idx_tmp = df_p.index[( (df_p.AbsDatetime>dt_extu)
                                          &(df_p.AbsDatetime<dt_extu_end))]
                    df_b.loc[idx_tmp,"ext_failure"] = df_b.loc[idx,"ext_failure"]

                    # the prediction score only happen at the time point of 
                    # extubation
                    if "ExtubationFailure" in out_path:
                        df_b.loc[idx,"ext_failure"] = 0
                    else:
                        df_b.loc[idx,"ext_failure"] = 1

            if endpoint_name=="extu_failure":
                df_b.loc[:,"Stable"] = (df_b.ext_failure==0).values
                df_b.loc[:,"InEvent"] = (df_b.ext_failure==1).values
            else:
                df_b.loc[:,"Stable"] = (df_b.ext_failure==1).values
                df_b.loc[:,"InEvent"] = (df_b.ext_failure==0).values                    
            df_b.loc[:,"Unknown"] = df_b.ext_failure.isnull()
        else:
            raise Exception("endpoint %s is not defined!"%endpoint_name)
            exit(0)
            
        df.append(df_b)
        sys.stdout.write("Reading respiratory endpoint: Batch %02d \r"%(i+1))

    print("\nFinished reading endpoints.")
    df = pd.concat(df, axis=0)
    df = df.rename(columns={"AbsDatetime": "Datetime"})
    df = df.sort_values(["PatientID", "Datetime"]).reset_index(drop=True)
    return df[["PatientID", "Datetime", "Stable", "InEvent", "Unknown"]]

def align_time(df1, df2):
    df1.loc[:,"Datetime"] = df1.Datetime.dt.floor("5T")
    df2.loc[:,"Datetime"] = df2.Datetime.dt.floor("5T")
    if len(set(df1.Datetime)&set(df2.Datetime))==0:
        df1.loc[:,"Datetime"] = df1.Datetime.dt.round("1h")
        df2.loc[:,"Datetime"] = df2.Datetime.dt.round("1h")
        df1 = df1.drop_duplicates("Datetime", keep="last")
        df2 = df2.drop_duplicates("Datetime", keep="last")
    df1 = df1.set_index("Datetime")
    df2 = df2.set_index("Datetime")
    
    df = df1.merge(df2.drop(columns=["PatientID"]), how="outer", 
                   left_index=True, right_index=True)
    df.loc[:,"RelDatetime"] = (df.index-df.index[0])/np.timedelta64(1,"s")
    assert(df.RelDatetime.isnull().sum()==0)
    df = df.sort_values("RelDatetime")
    df = df.reset_index()
    df = df.rename(columns={"index": "Datetime"})
    df = df.set_index("RelDatetime")
    return df

def get_threshold(datapath,
                  configs, 
                  rec=0.9, 
                  prec=None, 
                  is_first_onset=False, 
                  is_random=False):

    df = utils_data.get_result(datapath, 
                               configs, 
                               RANDOM=is_random, 
                               onset_type="first" if is_first_onset else None)
    
    if rec is not None and prec is None:
        tmp = df.iloc[np.argmin(np.abs(df.rec.values-rec))]
    elif rec is None and prec is not None:
        tmp = df.iloc[np.argmin(np.abs(df.prec.values-prec))]
    return tmp.tau, tmp.rec, tmp.prec

def get_event_info(df, pred_interval, hour2ignore=None):
    df.loc[:,"Onset"] = False
    df.loc[:,"Onset_copy"] = False
    df.loc[:,"EventEnd"] = False
    df.loc[:,"Merged"] = False

    if df.InEvent.sum() > 0:
        dt_event = df.index[df.InEvent]
        beg_event = np.concatenate((dt_event[[0]],
                                    dt_event[np.where(np.diff(dt_event)>pred_interval)[0]+1]))
        end_event = np.concatenate((dt_event[np.where(np.diff(dt_event)>pred_interval)[0]],
                                    dt_event[[-1]]))

        if hour2ignore is not None:
            end_event = end_event[beg_event>hour2ignore*3600]
            beg_event = beg_event[beg_event>hour2ignore*3600]

        df.loc[beg_event,"Onset"] = True
        df.loc[end_event,"EventEnd"] = True


        df.loc[:,"Onset_copy"] = df.Onset.values.copy()

        # merge events with gap shorter than t_mingap
        if t_mingap > 0:
            gap_witdth = (beg_event[1:] - end_event[:-1])
            idx_gap2merge = np.where(gap_width <= t_mingap * 60)[0]
            for igap in idx_gap2merge:
                df.loc[end_event[igap]:beg_event[1+igap]-300,"Merged"] = True
                df.loc[beg_event[1+igap],"Onset"] = False
                df.loc[end_event[igap],"EventEnd"] = False
            beg_event = np.delete(beg_event, idx_gap2merge+1)
            end_event = np.delete(end_event, idx_gap2merge)

        lst_event_end = np.concatenate(([hour2ignore*3600-t_reset_sec], end_event))
        for kk, dt_onset in enumerate(beg_event):
            win_pre_event = df[np.logical_and(df.index>=max(dt_onset-max_sec,lst_event_end[kk]+t_reset_sec),
                                              df.index<dt_onset-min_sec)] 
            if win_pre_event.Prediction.notnull().sum()==0:
#             if len(win_pre_event) in [0, win_pre_event.Prediction.isnull().sum()]:
                # if prior to the event, the status are unknown and there is no prediction score,                                                                                                           
                # this event is listed unpredictable at all, hence delete it from the Onset list                                                                                                            
                # print("reset.")
                df.loc[dt_onset, "Onset"] = False
                df.loc[end_event[kk], "EventEnd"] = False
    try:
        assert(df.Onset.sum()==df.EventEnd.sum())
    except:
        ipdb.set_trace()

def get_alarm_info(df, tau, pred_interval):
    # Compute true alarms and false alarms                                                                                 
    df.loc[:,"GeqTau"] = (df.Prediction >= tau)
    df.loc[:,"Alarm"] = False
    df.loc[:,"TrueAlarm"] = False

    if df.GeqTau.sum() > 0:
        dt_geq_tau = df.index[df.GeqTau]
        if t_silence == 5 and t_reset == 0 or df.Prediction.notnull().sum() < 2:
            df.loc[dt_geq_tau,"Alarm"] = True
            df.loc[df.index[df.Merged],"Alarm"] = False

        elif df.Onset.sum()==0:
            dt = 0
            while dt <= dt_geq_tau[-1]:
                dt = dt_geq_tau[dt_geq_tau>=dt][0]
                reset = False
                while not reset:
                    if dt in df.index and df.loc[dt,"GeqTau"]:
                        df.loc[dt,"Alarm"] = True
                        dt += ts
                    else:
                        reset = True
            df.loc[df.index[df.Merged],"Alarm"] = False
        else:
            dt = 0
            while dt <= dt_geq_tau[-1]:
                dt = dt_geq_tau[dt_geq_tau>=dt][0]
                reset = False
                while not reset:
                    if dt in df.index and df.loc[dt,"GeqTau"]:
                        if dt < df.index[df.EventEnd].values[0]:
                        # before the first event
                            df.loc[dt,"Alarm"] = True
                            dt = min( dt+ts , df.index[df.EventEnd].values[0]+t_reset_sec)
                        else:
                            # after the first event
                            dist2event = dt - df.index[df.EventEnd].values
                            last_end = df.index[df.EventEnd].values[np.where(dist2event>0)[0][-1]]
                            
                            if np.where(dist2event>0)[0][-1] == df.EventEnd.sum()-1:
                            # after the last event
                                next_end = df.index[-1]
                            else:
                            # before the last event
                                next_end = df.index[df.EventEnd].values[np.where(dist2event>0)[0][-1]+1]

                            if dt < last_end+t_reset_sec:
                                # if the current dt falls in the reset time period
                                dt = last_end+t_reset_sec
                            else:
                                # otherwise
                                df.loc[dt,"Alarm"] = True
                                dt = min( dt+ts , next_end+t_reset_sec)
                    else:
                        reset = True
        df.loc[df.index[df.Merged],"Alarm"] = False
        for dt_alarm in df.index[df["Alarm"]]:
            win_post_alarm = df[np.logical_and(df.index<=dt_alarm+max_sec,df.index>dt_alarm+min_sec)]
            if len(win_post_alarm) in [0, win_post_alarm.Unknown.sum()]:
                # if there is no more time after the alarm or the status are completely unknown, we                        
                # consider the alarm neither true or false, hence disable it.                                              
                df.loc[dt_alarm,"Alarm"] = False
            else:
                df.loc[dt_alarm,"TrueAlarm"] = win_post_alarm.InEvent.sum()>0

    # Compute captured events and missed events                                                                            
    df.loc[:,"CatchedOnset"] = False
    if df.Onset.sum() > 0:
        lst_event_end = np.concatenate(([0], df.index[df.EventEnd]))
        for kk, dt_onset in enumerate(df.index[df.Onset]):
            win_pre_event = df[np.logical_and(df.index>=max(dt_onset-max_sec,lst_event_end[kk]),
                                              df.index<dt_onset-min_sec)]
            df.loc[dt_onset,"CatchedOnset"] = win_pre_event.Alarm.sum() > 0
    return df


def plot_ts(df_p):
    """
    plot the time series of a patient
    """
    tvals = df_p.index/3600
    plt.fill_between(tvals, df_p.Stable.astype(int)-0, step="pre", color="C2", alpha=0.2, zorder=1, label="Stable")
    plt.fill_between(tvals, df_p.InEvent.astype(int)-0, step="pre", color="C3", alpha=0.2, zorder=2, label="Renal Failure")
    plt.plot(tvals, df_p.Prediction, color="k", marker='.', markersize=2**2, zorder=3, label="Prediction score")
    plt.scatter(tvals[df_p.Onset]-300/3600, [1]*df_p.Onset.sum(), marker=11, color="C3", zorder=4, label="Onset")
#     plt.scatter(tvals[df_p.EventEnd], [1]*df_p.EventEnd.sum(), marker=11, color="C2", zorder=5, label="_nolegend_")
    plt.axhline(y=tau, linestyle="--", color='gray', zorder=6, label="Threshold")
    plt.scatter(tvals[df_p.Alarm], df_p.Prediction[df_p.Alarm]+0.05, marker='d', color="C2", s=5**2, zorder=7, label="Alarm")
    plt.scatter(tvals[df_p.TrueAlarm], df_p.Prediction[df_p.TrueAlarm]+0.05, marker='d', color="C3", zorder=8, label="True Alarm")
    plt.scatter(tvals[df_p.CatchedOnset]-300/3600, [0.85]*df_p.CatchedOnset.sum(), marker='*', s=16**2, color="C3", zorder=9, label="Catched Onset")
    plt.xlim(tvals.min()-0.05, tvals.max()+0.05)
    if tvals.max() > 48:
        tickvals = np.arange(0,tvals.max(),24)
    else:
        tickvals = np.arange(0,tvals.max(),4)
    plt.xticks(tickvals, ["%d"%x for x in tickvals])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Time since admission (h)')
    plt.ylabel('Prediction score')

def sys_print(line):
    sys.stdout.write(line+"\n")
    sys.stdout.flush()
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    ### essential parameters
    parser.add_argument("-ep_path")
    parser.add_argument("--pd_path", default=None)
    parser.add_argument("-out_path")
    parser.add_argument("-split_file", default="")

    parser.add_argument("--t_delta", type=int, default=0)
    parser.add_argument("--t_window", type=int, default=480)
    parser.add_argument("--t_silence", type=int, default=30)
    parser.add_argument("--t_mingap", type=int, default=0)
    parser.add_argument("--t_reset", type=int, default=0)
    parser.add_argument("--temporal_split", default="temporal_1")

    parser.add_argument("--RANDOM", action="store_true")
    parser.add_argument("--FIRST_ONSET", action="store_true")
    parser.add_argument("--random_seed", type=int, default=2021)

    ### types of enpoints
    parser.add_argument("--stable_status", nargs="+", default=["0", "1"])
    parser.add_argument("--unstable_status", nargs="+", default=["2", "3"])
    parser.add_argument("--ep_problem", default="renal", choices=["respiratory", "renal"])
    parser.add_argument("--ep_type", default="resp_failure")

    ### baseline parameters
    parser.add_argument("--spo2", type=float, default=90)
    parser.add_argument("--fio2", type=float, default=60)
    parser.add_argument("--THRESHOLD_BASELINE", action="store_true")

    ### whether to read ensemble and if there are more than one prediction results, how to concatenate them
    parser.add_argument("--model_json", default=None)
    
    ### intersection parameters
    parser.add_argument("--tg_path", default=None)
    parser.add_argument("--mh_path", default=None)
    parser.add_argument("--bw_rf_path", default=None)
    parser.add_argument("--bw_hmm_path", default=None)
    parser.add_argument("--tg_ep_type", default="1_1")
    parser.add_argument("--set_type", default="test")
    parser.add_argument("--idx_method", type=int, nargs="+", default=[0,1])
    parser.add_argument("--lst_w", type=float, nargs="+", default=[0.333,0.333,0.334])

    ### parallelization parameters
    parser.add_argument("--idx_batch", type=int, default=None)
    parser.add_argument("--idx_threshold", type=int, default=None)

    parser.add_argument("--kdigo_intervent_type", default=None, choices=["furosemide", "fluid", "no_furosemide", "no_fluid", "wo_invasive_bp", "w_invasive_bp"])

    ### parameters for getting alarm information with fixed precision/recall
    parser.add_argument("--fixed_recall", type=float, default=None)
    parser.add_argument("--fixed_precision", type=float, default=None)
    parser.add_argument("--alarm_path", default=None)
    parser.add_argument("--DEBUG", action="store_true")
    parser.add_argument("--correct_false_alarm", action="store_true")
    parser.add_argument("--correct_false_alarm_win", default=4, type=int)
    parser.add_argument("--correct_false_alarm_mode", default="FA2TA", choices=["DelFA", "FA2TA"])

    parser.add_argument("--select_cohort", default=None)
    parser.add_argument("--bootstrap_seed", default=None, type=int)

    parser.add_argument("--no_urine", action="store_true")
    parser.add_argument("--only_urine", action="store_true")

    parser.add_argument("--create_ensemble", action="store_true")
    parser.add_argument("--calibrated", action="store_true")
    parser.add_argument("--mix_alpha", type=float, default=1)

    args = parser.parse_args()
    configs = vars(args)
    for key, val in configs.items():
        exec("%s=%s"%(key, "'%s'"%val if type(val)==str else val))
    t_reset_sec = t_reset*60
    lst_w = np.array(lst_w) / 10

    ts = t_silence * 60
    max_sec = (t_delta+t_window) * 60
    min_sec = t_delta * 60
    t_unit = np.timedelta64(1,"s")

    with open(split_file,"rb") as tmp:
        pid_mapping = pickle.load(tmp)
    pids = pid_mapping[temporal_split][set_type]
    
    if ep_problem=="respiratory":
        if THRESHOLD_BASELINE:
            df_pd = read_clinical_baseline(pd_path, lst_files, spo2=spo2, fio2=fio2)
        else:
            df_pd = utils_data.read_model_data(model_json, pids)        
        df_ep = read_ep_resp(ep_path, lst_files, ep_type, stable_status=stable_status, unstable_status=unstable_status)
        
    else:
        df_pd = utils_data.read_model_data(model_json, pids)
        df_ep = read_ep_renal(ep_path, pids, ep_type, no_urine=no_urine, only_urine=only_urine)
        
    if kdigo_intervent_type is not None:
        if kdigo_intervent_type in ["furosemide", "no_furosemide"]:
            pids_with_intervention = pd.read_csv("patient_list_with_furosemide.csv").PatientID.values
        elif kdigo_intervent_type in ["fluid", "no_fluid"]:
            pids_with_intervention = pd.read_csv("patient_list_with_fluid.csv").PatientID.values
        elif kdigo_intervent_type in ["wo_invasive_bp", "w_invasive_bp"]:
            pids_with_intervention = pd.read_csv("pids_without_invasive_bp.csv").PatientID.values
            
        if "no" in kdigo_intervent_type or kdigo_intervent_type == "w_invasive_bp":
            df_ep.drop(df_ep.index[df_ep.PatientID.isin(pids_with_intervention)], inplace=True)
            df_pd.drop(df_pd.index[df_pd.PatientID.isin(pids_with_intervention)], inplace=True)      
        else:
            df_ep.drop(df_ep.index[~df_ep.PatientID.isin(pids_with_intervention)], inplace=True)
            df_pd.drop(df_pd.index[~df_pd.PatientID.isin(pids_with_intervention)], inplace=True)      

    pred_interval = df_pd[df_pd.PatientID==df_pd.PatientID.unique()[0]].Datetime.diff() / np.timedelta64(1,"s")
    pred_interval = pred_interval.values[1]

    lst_pid = np.array(list(set(df_pd.PatientID)&set(df_ep.PatientID)))
    if DEBUG:
        lst_pid = lst_pid[:20]
        
    if bootstrap_seed is not None:
        lst_pid = np.random.permutation(lst_pid)[:int(len(lst_pid)*0.5)]
        
    np.random.seed(random_seed)
    
    df_with_event =  []
    max_score = 0
    for i, pid in enumerate(lst_pid):
        
        df = align_time(df_ep[df_ep.PatientID==pid].copy(), df_pd[df_pd.PatientID==pid].copy())
        df.loc[:,"InEvent"] = df["InEvent"].fillna(method="ffill").fillna(False)
        df.loc[:,"PatientID"] = pid
            
        df.loc[df.index[~df.Stable.astype(bool)], 'Prediction'] = np.nan
            
        if df.Prediction.notnull().sum()==0:
            print("%d, PatientID %d does not have valid prediction score"%(i, pid))
            continue

        if RANDOM:
            df.loc[df.index[df.Prediction.notnull()], "Prediction"] = np.random.rand(df.Prediction.notnull().sum())
            
        get_event_info(df, pred_interval, hour2ignore=2)

        for i, dt in enumerate(df.index[df.Onset]):
            if i==0:
                max_score = max(df[(df.index>=0)&(df.index>=dt-max_sec)&(df.index<dt-min_sec)].Prediction.max(), max_score)
            else:
                max_score = max(df[(df.index>=df.index[df.EventEnd][i-1])&(df.index>=dt-max_sec)&(df.index<dt-min_sec)].Prediction.max(), max_score)
        df_with_event.append(df)
        
    df_with_event = pd.concat(df_with_event).reset_index()

    if THRESHOLD_BASELINE:
        thresholds = [0,0.5,1.1]
    elif fixed_recall is not None and fixed_precision is None:
        tau, rec, prec = get_threshold(out_path, configs, rec=fixed_recall)
        thresholds = [tau]
        print(thresholds)
    elif fixed_recall is None and fixed_precision is not None:
        tau, rec, prec = get_threshold(out_path, configs, rec=None, prec=fixed_precision, FIRST_ONSET=FIRST_ONSET)
        thresholds = [tau]
    else:
        pa_with_events = df_with_event[df_with_event.InEvent==True].PatientID.unique()
        pred_before_onset = []
        for pid in pa_with_events:
            df_tmp  = df_with_event[df_with_event.PatientID==pid]
            for dt in df_tmp.RelDatetime[df_tmp.Onset]:
                pred_tmp = df_tmp[(df_tmp.RelDatetime>=dt-max_sec)&(df_tmp.RelDatetime<dt-min_sec)].Prediction
                pred_tmp = pred_tmp[pred_tmp.notnull()]
                pred_before_onset.extend(pred_tmp)
        thresholds = [np.nanpercentile(pred_before_onset, i) for i in range(100)]        
        thresholds = np.concatenate(([0], thresholds, [max_score, df_with_event.Prediction.max()]))
        thresholds = np.interp(np.arange(0, len(thresholds)+0.25, 0.25), range(len(thresholds)), thresholds)        
        thresholds = np.concatenate((thresholds[:315:3], thresholds[315:]))
        thresholds = thresholds[::2]

        if idx_threshold >= len(thresholds):
            exit(0)

    stable_frac = [df_with_event[df_with_event.PatientID==pid].Stable.sum()/df_with_event[df_with_event.PatientID==pid].shape[0] for pid in df_with_event.PatientID.unique()]
    inevent_frac = [df_with_event[df_with_event.PatientID==pid].InEvent.sum()/df_with_event[df_with_event.PatientID==pid].shape[0] for pid in df_with_event.PatientID.unique()]
    unknown_frac = [df_with_event[df_with_event.PatientID==pid].Unknown.sum()/df_with_event[df_with_event.PatientID==pid].shape[0] for pid in df_with_event.PatientID.unique()]
    event_ratio = np.array(inevent_frac) / (np.array(inevent_frac)+np.array(stable_frac))
        
    if fixed_recall is None and fixed_precision is None:
        TA = []
        FA = []
        CE = []
        ME = []

        TA_first =  []
        FA_first =  []
        CE_first =  []
        ME_first =  []
        if idx_threshold is None:
            pass
        else:
            thresholds  = thresholds[[idx_threshold]]
    else:
        res_f = ("rand%d_"%random_seed if RANDOM else "")+"tg-%d_tr-%d_dt-%d_ws-%d_ts-%d"%(t_mingap, t_reset, t_delta, t_window, t_silence)
        if idx_threshold is None and idx_batch is None:
            pass
        elif idx_batch is None:
            res_f = res_f + "_cnts_i%d"%idx_threshold
        else:
            res_f = res_f + "_batch_i%d"%idx_batch            
            
        if not os.path.exists(alarm_path):
            os.mkdir(alarm_path)            
            
        if fixed_recall is not None and fixed_precision is None:
            alarm_path = os.path.join(alarm_path, (res_f+"_rec-%g.h5"%fixed_recall))
        elif fixed_recall is None and fixed_precision is not None:
            alarm_path = os.path.join(alarm_path, (res_f+"_prec-%g.h5"%fixed_precision))

    if correct_false_alarm and ep_problem=="renal": 
        df_in = pd.concat([read_vars_imputed("/cluster/work/grlab/clinical/hirid2/research/5c_imputed_resp/210407_noimpute/reduced/point_est", f) for f in lst_files]).reset_index(drop=True)

    df_static = pd.read_hdf("/cluster/work/grlab/clinical/hirid2/research/1a_hdf5_clean/v8/static.h5")
    if select_cohort is None:
        pass
    elif select_cohort.lower() in ['all', 'none']:
        pass
    elif select_cohort=="case":
        lst_pid = np.load("lst_pids_furo_new.npy")
    elif select_cohort=="ctrl":
        lst_pid = np.load("lst_pids_no_furo_new.npy")
    elif select_cohort=="male":
        lst_pid = [pid for pid in lst_pid if pid in df_static[df_static.Sex=="M"].PatientID.values]
    elif select_cohort=="female":
        lst_pid = [pid for pid in lst_pid if pid in df_static[df_static.Sex=="F"].PatientID.values]
    elif select_cohort=="emergency":
        lst_pid = [pid for pid in lst_pid if pid in df_static[df_static.Emergency==1].PatientID.values]
    elif select_cohort=="nonemergency":
        lst_pid = [pid for pid in lst_pid if pid in df_static[df_static.Emergency==0].PatientID.values]
    elif select_cohort=="surgical":
        lst_pid = [pid for pid in lst_pid if pid in df_static[df_static.Surgical>0].PatientID.values]
    elif select_cohort=="nonsurgical":
        lst_pid = [pid for pid in lst_pid if pid in df_static[df_static.Surgical==0].PatientID.values]
    elif select_cohort=="furosemide":
        pids_with_intervention = pd.read_csv("patient_list_with_furosemide.csv").PatientID.values
        lst_pid = [pid for pid in lst_pid if pid in pids_with_intervention]
    elif select_cohort=="no_furosemide":
        pids_with_intervention = pd.read_csv("patient_list_with_furosemide.csv").PatientID.values
        lst_pid = [pid for pid in lst_pid if pid not in pids_with_intervention]
    elif 'apachepatgroup' in select_cohort:
        lst_pid = [pid for pid in lst_pid if pid in df_static[df_static.APACHEPatGroup==int(select_cohort[14:])].PatientID.values]
    elif 'age' in select_cohort:
        min_age = int(select_cohort.split("_")[1])
        max_age = int(select_cohort.split("_")[2])
        lst_pid = [pid for pid in lst_pid if pid in df_static[(df_static.Age>=min_age)&(df_static.Age<=max_age)].PatientID.values]
    else:
        raise Exception("Not implemented.")

    for tau in thresholds:
        for pid in lst_pid:
            if (df_with_event.PatientID==pid).sum() == 0:
                continue
            df = df_with_event[df_with_event.PatientID==pid].set_index("RelDatetime")
            if correct_false_alarm and ep_problem=="renal" and correct_false_alarm_mode=="DelFA": 
                df_in_tmp  = df_in[df_in.PatientID==pid]
                if (df_in_tmp.pm69>0).sum()>0:
                    dt_drug = df_in_tmp[df_in_tmp.pm69>0].RelDatetime.values
                    dt_drug_beg = dt_drug[1:][np.diff(dt_drug)>300]
                    dt_drug_beg = np.concatenate((dt_drug[:1], dt_drug_beg))
                    
                    dt_drug_end = dt_drug[:-1][np.diff(dt_drug)>300]
                    dt_drug_end = np.concatenate((dt_drug_end, dt_drug[-1:]))

                    
                    for i in range(len(dt_drug_beg)):
                        if i==0:
                            tmp  = df[(df.index<dt_drug_beg[i]+correct_false_alarm_win*3600)&(df.index>=dt_drug_beg[i])]
                        else:
                            tmp  = df[(df.index>=dt_drug_end[i-1])&(df.index<dt_drug_beg[i]+correct_false_alarm_win*3600)&(df.index>=dt_drug_beg[i])]
                        df.loc[tmp.index, "Prediction"] = np.nan

            if len(df)==0:
                continue
            get_alarm_info(df, tau, pred_interval)
            
            if correct_false_alarm and ep_problem=="renal" and correct_false_alarm_mode=="FA2TA": 
                df_in_tmp  = df_in[df_in.PatientID==pid]
                if (df_in_tmp.pm69>0).sum()>0:
                    dt_drug = df_in_tmp[df_in_tmp.pm69>0].RelDatetime.values
                    dt_drug_beg = dt_drug[1:][np.diff(dt_drug)>300]
                    dt_drug_beg = np.concatenate((dt_drug[:1], dt_drug_beg))
                    
                    dt_drug_end = dt_drug[:-1][np.diff(dt_drug)>300]
                    dt_drug_end = np.concatenate((dt_drug_end, dt_drug[-1:]))

                    
                    for i in range(len(dt_drug_beg)):
                        if i==0:
                            tmp  = df[(df.Alarm)&(df.index<dt_drug_beg[i]+correct_false_alarm_win*3600)&(df.index>=dt_drug_beg[i])]
                        else:
                            tmp  = df[(df.Alarm)&(df.index>=dt_drug_end[i-1])&(df.index<dt_drug_beg[i]+correct_false_alarm_win*3600)&(df.index>=dt_drug_beg[i])]
                            
                        if ((tmp.Alarm)&(~tmp.TrueAlarm)).sum() > 0:
                            df.loc[tmp.index,"TrueAlarm"] = True

            # if correct_false_alarm and ep_problem=="renal" and correct_false_alarm_mode=="DelFA":
            #     for dt in df[df.CatchedOnset].index:
            #         if df.loc[dt-max_sec:dt-min_sec,"Alarm"].sum() == 0:
            #             df.loc[dt,"Onset"] = False
            #             df.loc[dt,"CatchedOnset"] = False
            #         else:
            #             pass

            # df = df[df.index>48*3600]
            if len(df) == 0:
                continue
            
            if fixed_recall is None and fixed_precision is None:
                TA.append([tau, pid, df.TrueAlarm.sum()])
                FA.append([tau, pid, df.Alarm.sum() - df.TrueAlarm.sum()])
                CE.append([tau, pid, df.CatchedOnset.sum()])
                ME.append([tau, pid, df.Onset.sum() - df.CatchedOnset.sum()])

                if df.Onset.sum() > 0:
                    dt_before_first = df.index[df.index<=df.index[df.Onset][0]]
                    TA_first.append([tau, pid, df.loc[dt_before_first].TrueAlarm.sum()])
                    FA_first.append([tau, pid, df.loc[dt_before_first].Alarm.sum() - df.loc[dt_before_first].TrueAlarm.sum()])
                    CE_first.append([tau, pid, df.loc[dt_before_first].CatchedOnset.sum()])
                    ME_first.append([tau, pid, df.loc[dt_before_first].Onset.sum() - df.loc[dt_before_first].CatchedOnset.sum()])
                else:
                    TA_first.append([tau, pid, 0])
                    FA_first.append([tau, pid, df.Alarm.sum()])
                    CE_first.append([tau, pid, 0])
                    ME_first.append([tau, pid, 0])
                    
            else:
                df_alarm = df[df.Alarm].copy()
                if len(df_alarm) > 0:
                    df_alarm.loc[:,"t2onset_h"] = np.nan
                for i, dt_onset in enumerate(df.index[df.Onset]):
                    if i == 0:
                        idx_alarm = df_alarm.index[df_alarm.index<dt_onset]
                    else:
                        idx_alarm = df_alarm.index[(df_alarm.index>df.index[df.Onset][i-1])&(df_alarm.index<dt_onset)]
                    df_alarm.loc[idx_alarm,"t2onset_h"] = (dt_onset - idx_alarm) / 3600
                if len(df_alarm) > 0:
                    df_alarm = pd.concat([df_alarm, df[df.Onset|df.EventEnd]]).sort_index()
                else:
                    df_alarm =  df[df.Onset|df.EventEnd].copy()
                    if len(df_alarm) > 0:
                        df_alarm.loc[:,"t2onset_h"] = np.nan
                
                if len(df_alarm) == 0:
                    df_alarm = df.iloc[[-1]].copy()
                    df_alarm.loc[:,"t2onset_h"] = np.nan

                df_alarm = df_alarm[["PatientID", "Alarm","TrueAlarm","Onset","CatchedOnset", "EventEnd", "t2onset_h", "Prediction", "Datetime"]]
                df_alarm.loc[:,"DiscRelDatetime"] = df.index[-1]
                df_alarm.to_hdf(alarm_path, "p%d"%pid, complevel=5, complib="blosc:lz4")

    if fixed_recall is None and fixed_precision is None:
        TA = pd.DataFrame(TA, columns=['tau','PatientID', 'TA'])
        FA = pd.DataFrame(FA, columns=['tau','PatientID', 'FA'])
        ME = pd.DataFrame(ME, columns=['tau','PatientID', 'ME'])
        CE = pd.DataFrame(CE, columns=['tau','PatientID', 'CE'])

        TA = TA[['tau', 'TA']].groupby('tau').sum()
        FA = FA[['tau', 'FA']].groupby('tau').sum()
        ME = ME[['tau', 'ME']].groupby('tau').sum()
        CE = CE[['tau', 'CE']].groupby('tau').sum()

        TA_first = pd.DataFrame(TA_first, columns=['tau','PatientID', 'TA'])
        FA_first = pd.DataFrame(FA_first, columns=['tau','PatientID', 'FA'])
        ME_first = pd.DataFrame(ME_first, columns=['tau','PatientID', 'ME'])
        CE_first = pd.DataFrame(CE_first, columns=['tau','PatientID', 'CE'])

        TA_first = TA_first[['tau', 'TA']].groupby('tau').sum()
        FA_first = FA_first[['tau', 'FA']].groupby('tau').sum()
        ME_first = ME_first[['tau', 'ME']].groupby('tau').sum()
        CE_first = CE_first[['tau', 'CE']].groupby('tau').sum()


        cnts = pd.concat([TA,FA,CE,ME], axis=1).reset_index()
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        res_f = ("rand%d_"%random_seed if RANDOM else "")+"tg-%d_tr-%d_dt-%d_ws-%d_ts-%d"%(t_mingap, t_reset, t_delta, t_window, t_silence)
        if idx_threshold is None and idx_batch is None:
            pass
        elif idx_batch is None:
            res_f = res_f + "_cnts_i%d"%idx_threshold
        else:
            res_f = res_f + "_batch_i%d"%idx_batch            
        if not DEBUG:
            cnts.to_csv(os.path.join(out_path, (res_f+".csv")), index=False)
        print(os.path.join(out_path, res_f))


        cnts_first = pd.concat([TA_first,FA_first,CE_first,ME_first], axis=1).reset_index()
        res_f_first = ("rand%d_"%random_seed if RANDOM else "")+"tg-%d_tr-%d_dt-%d_ws-%d_ts-%d_first"%(t_mingap, t_reset, t_delta, t_window, t_silence)
        if idx_threshold is None and idx_batch is None:
            pass
        elif idx_batch is None:
            res_f_first = res_f_first + "_cnts_i%d"%idx_threshold
        else:
            res_f_first = res_f_first + "_batch_i%d"%idx_batch            
