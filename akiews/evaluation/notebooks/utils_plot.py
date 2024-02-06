import pandas as pd
import numpy as np
import os
import h5py
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
import seaborn as sns

plot_alpha=0.4

def read_event_based_pr_single_split(res_path, 
                                     pred_win=1440, 
                                     min_event_gap=0, 
                                     t_silence=30,
                                     t_buffer=0,
                                     t_reset=30,
                                     calibration_scaler=1,
                                     random_classifier=False):
    """
    res_path: path to the event-based evaluation results
    pred_win: future prediction window size (in minutes)
    min_event_gap: the minimal event gap length (in minutes), any gap smaller should be closed
    t_silence: alarm silencing time (in minutes)
    t_buffer: minimal buffer time before event
    t_reset: alarm reset time after patient recovers from failure event
    calibration_scaler: scaler to calibrate the prevalence (AUPRC of the random classifier)
    """
    
    prefix_str = 'tg-%d_tr-%d_dt-%d_ws-%d_ts-%d'%(min_event_gap,
                                                  t_reset,
                                                  t_buffer,
                                                  pred_win,
                                                  t_silence) # prefix for different configuration
    res = []
    for f in os.listdir(res_path):
        if prefix_str+"_" in f or prefix_str+"." in f:
            if random_classifier and 'rand' in f:
                res.append(pd.read_csv(os.path.join(res_path,f)))
            elif not random_classifier and 'rand' not in f:
                res.append(pd.read_csv(os.path.join(res_path,f)))     
    try:            
        res = pd.concat(res)
    except:
        raise Exception('%s'%res_path)
    res.loc[:,'FA'] = calibration_scaler * res.FA # calibrate the false alarms using the scale
    
    res.loc[:,"recall"] = res.CE / (res.CE+res.ME)
    res.loc[:,"precision"] = res.TA / (res.TA+res.FA)
    res = res.sort_values(['tau', 'recall', 'precision'])
    res = res.drop_duplicates('recall', keep='last')
    res = res.reset_index(drop=True)
    return res

def read_event_based_pr_multi_splits(res_path, 
                                     splits,
                                     pred_win=1440, 
                                     min_event_gap=0, 
                                     t_silence=30,
                                     t_buffer=0,
                                     t_reset=30,
                                     calibration_scaler=1,
                                     random_classifier=False):
    """
    res_path: path to the event-based evaluation results
    splits: list of splits 
    pred_win: future prediction window size (in minutes)
    min_event_gap: the minimal event gap length (in minutes), any gap smaller should be closed
    t_silence: alarm silencing time (in minutes)
    t_buffer: minimal buffer time before event
    t_reset: alarm reset time after patient recovers from failure event
    calibration_scaler: scaler to calibrate the prevalence (AUPRC of the random classifier)
    """
    if type(splits) is not list and type(split)==str:
        splits = [splits]
        
    res = dict()
    for split in splits:
        res.update({split: read_event_based_pr_single_split(os.path.join(res_path, split),
                                                             pred_win=pred_win,
                                                             min_event_gap=min_event_gap,
                                                             t_silence=t_silence,
                                                             t_buffer=t_buffer,
                                                             t_reset=t_reset,
                                                             random_classifier=random_classifier,
                                                             calibration_scaler=calibration_scaler)})
        
    return res


def plot_event_based_prc(curves, fixed_rec=0.8):
    """
    curves: a dictionary containing the configuration of all curves in the same plot
    """
    dict_res = dict()
    for i, model in enumerate(curves.keys()):
        if 'calibration_scaler' in curves[model]:
            calibration_scaler = curves[model]['calibration_scaler']
        else:
            calibration_scaler = 1
        res = read_event_based_pr_multi_splits(curves[model]['res_path'], 
                                               curves[model]['splits'], 
                                               pred_win=curves[model]['pred_win'],
                                                 min_event_gap=curves[model]['min_event_gap'],
                                                 t_silence=curves[model]['t_silence'],
                                                 t_buffer=curves[model]['t_buffer'],
                                                 t_reset=curves[model]['t_reset'],
                                                 random_classifier=curves[model]['random_classifier'],
                                                 calibration_scaler=calibration_scaler)
        aggr_res = [] # aggregated results from all splits
        for k, v in res.items():
            aggr_res.append(v.set_index('recall').sort_index().rename(columns={'precision':k})[[k]])
            
        
        aggr_res = pd.concat(aggr_res, axis=1)
        aggr_res = aggr_res.sort_index()
        aggr_res = aggr_res.interpolate(method='index')
        aggr_res = aggr_res[aggr_res.isnull().sum(axis=1)==0]
        
        dict_res.update({model:aggr_res})
        
        if 'single_point' in curves[model] and curves[model]['single_point']:
            
            aggr_res =  aggr_res[aggr_res.index<1]
            precision_mean = aggr_res.mean(axis=1)
            precision_std = aggr_res.std(axis=1) if aggr_res.shape[1]>1 else 0
            
            plt.errorbar(aggr_res.index,
                         precision_mean,
                         yerr=precision_std,
                         color=curves[model]['color'],
                         marker=curves[model]['marker_style'],
                         label=model)
            
        else:
            aucs = [metrics.auc(aggr_res.index,aggr_res[k]) for k in aggr_res.columns]

            precision_mean = aggr_res.mean(axis=1)
            precision_std = aggr_res.std(axis=1)

            esti_rec = aggr_res.index[np.argmin(np.abs(aggr_res.index-fixed_rec))]
            
            plt.plot(aggr_res.index, 
                     precision_mean,
                     color=curves[model]['color'],
                     linestyle=curves[model]['linestyle'],
                     label=model)
            plt.fill_between(aggr_res.index,
                             precision_mean-precision_std,
                             precision_mean+precision_std,
                             color=curves[model]['color'],
                             alpha=plot_alpha)
            
            scale = 0.06
            if np.std(aucs)==0:
                plt.text(0.03, 
                         scale*(len(curves)-i), 
                         '%3.3f (prec@{rec=%d%%}: %d%%)'%(np.mean(aucs), esti_rec*100, precision_mean.loc[esti_rec]*100),
                         color=curves[model]['color'])
                
            else:
                plt.text(0.03, 
                         scale*(len(curves)-i), 
                         '%3.3f$\pm$%3.3f'%(np.mean(aucs), np.std(aucs)),
                         color=curves[model]['color'])
                # plt.text(0.03, 
                #          scale*(len(curves)-i), 
                #          '%3.3f$\pm$%3.3f (prec@{rec=%d%%}: %d%%)'%(np.mean(aucs), np.std(aucs), esti_rec*100, precision_mean.loc[esti_rec]*100),
                #          color=curves[model]['color'])
            
    plt.text(0.03, 
             scale*(len(curves)+1), 
             'AUPRC')
    plt.grid(alpha=0.5)    
    plt.axis('equal')
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.legend(loc=1, facecolor='white')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    prev = precision_mean.loc[esti_rec] if fixed_rec==1.0 else None
    return dict_res, prev

    
def plot_metric_vs_setting(curves, ylabel='auprc', xlabel='t_silence', fixed_rec=0.8):
    """
    curves: a dictionary containing the configuration of all curves in the same plot
    """
    xtick_vals = []
    metric_vals = []
    all_auc = []
    for i, model in enumerate(curves.keys()):
        
        if 'calibration_scaler' in curves[model]:
            calibration_scaler = curves[model]['calibration_scaler']
        else:
            calibration_scaler = 1
            
        res = read_event_based_pr_multi_splits(curves[model]['res_path'], 
                                               curves[model]['splits'], 
                                               pred_win=curves[model]['pred_win'],
                                                 min_event_gap=curves[model]['min_event_gap'],
                                                 t_silence=curves[model]['t_silence'],
                                                 t_buffer=curves[model]['t_buffer'],
                                                 t_reset=curves[model]['t_reset'],
                                               random_classifier=curves[model]['random_classifier'],
                                                 calibration_scaler=calibration_scaler)
        
        
        aggr_res = [] # aggregated results from all splits
        for k, v in res.items():
            aggr_res.append(v.set_index('recall').sort_index().rename(columns={'precision':k})[[k]])
            
        aggr_res = pd.concat(aggr_res, axis=1)
        aggr_res = aggr_res.sort_index()
        aggr_res = aggr_res.interpolate(method='index')
        aggr_res = aggr_res[aggr_res.isnull().sum(axis=1)==0]
        
        if 'single_point' in curves[model] and curves[model]['single_point']:
            aggr_res =  aggr_res[aggr_res.index<1]
            precision_mean = aggr_res.mean(axis=1)
            precision_std = aggr_res.std(axis=1) if aggr_res.shape[1]>1 else 0
                        
        else:
            aucs = [metrics.auc(aggr_res.index,aggr_res[k]) for k in aggr_res.columns]

            precision_mean = aggr_res.mean(axis=1)
            precision_std = aggr_res.std(axis=1)

            esti_rec = aggr_res.index[np.argmin(np.abs(aggr_res.index-fixed_rec))]
                            
        xtick_vals.append(curves[model][xlabel])
        if ylabel=='auprc':
            metric_vals.append([np.mean(aucs), np.std(aucs)])
            
        else:
            metric_vals.append([precision_mean.loc[esti_rec], 
                                precision_std.loc[esti_rec]])
            
        aucs = pd.DataFrame(np.reshape(aucs,(-1,1)), columns=["metric"])
        aucs.loc[:,xlabel] = curves[model][xlabel]
        all_auc.append(aucs) 
    all_auc = pd.concat(all_auc).reset_index(drop=True)
    metric_vals = np.array(metric_vals)
    plt.plot(xtick_vals, metric_vals[:,0])   
    plt.fill_between(xtick_vals, 
                     metric_vals[:,0]-metric_vals[:,1],
                     metric_vals[:,0]+metric_vals[:,1],
                     alpha=0.5)
    
    plt.xticks(xtick_vals, xtick_vals)
    plt.xlim([0,xtick_vals[-1]])
    plt.grid(alpha=0.5)    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    prev = metric_vals[:,0] if fixed_rec==1.0 else None
    return all_auc, prev

def get_keystr(configs):
    try:
        f_key = "tg-%d"%configs["t_mingap_m"]
        f_key += "_tr-%d"%configs["t_reset_m"]
        f_key += "_dt-0"
        f_key += "_ws-%d"%configs["t_window_m"]
        f_key += "_ts-%d"%configs["t_silence_m"]
    except:
        f_key = "tg-%d"%configs["t_mingap"]
        f_key += "_tr-%d"%configs["t_reset"]
        f_key += "_dt-0"
        f_key += "_ws-%d"%configs["t_window"]
        f_key += "_ts-%d"%configs["t_silence"]
    return f_key
