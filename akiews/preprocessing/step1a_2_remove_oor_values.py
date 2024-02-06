#!/usr/bin/env python
import pandas as pd
import numpy as np
import pickle
import gc
import os

import sys
sys.path.append('../utils')
import preproc_utils
import matplotlib.pyplot as plt


cumul_urine_id_lst = [10015000, 10015033, 10015034, 10015035, 10015036, 10015037, 10015038]
dict_interval_urine_meas = {key: [] for key in cumul_urine_id_lst}
def convert_non_urine_variables(df, variable_ids, hr_start, adm):
    
    df_urine = df[df.VariableID.isin(variable_ids)]

    if len(df_urine) == 0:
        return df, df_urine[['PatientID', 'Datetime', 'Value']].copy()
    else:
        for vid in df_urine.VariableID.unique():
            df_tmp = df_urine[df_urine.VariableID==vid]
            if df.iloc[0].PatientID in hr_start.index and df.iloc[0].PatientID in adm.index:
                adm_time = min(hr_start.loc[df.iloc[0].PatientID].Datetime, adm.loc[df.iloc[0].PatientID].AdmissionTime)
            elif df.iloc[0].PatientID not in hr_start.index and df.iloc[0].PatientID in adm.index:
                adm_time = adm.loc[df.iloc[0].PatientID].AdmissionTime
            elif df.iloc[0].PatientID in hr_start.index and df.iloc[0].PatientID not in adm.index:
                adm_time = hr_start.loc[df.iloc[0].PatientID].Datetime
            else:
                adm_time = df_tmp.iloc[0].Datetime
            nrec_pre_hr = (df_tmp.Datetime < adm_time).sum()
            if nrec_pre_hr == 0:
                pass
            elif nrec_pre_hr ==1:
                df.loc[df_tmp.index[0], 'Datetime'] = adm_time
            else:
                df.drop(df_tmp.index[df_tmp.Datetime < adm_time][:-1], inplace=True)
                df.loc[df_tmp.index[df_tmp.Datetime < adm_time][-1], 'Datetime'] = adm_time
            df_tmp = df[df.VariableID==vid]
            df.drop(df_tmp.index[df_tmp.Datetime.duplicated(keep='last')], inplace=True) 
            df_tmp = df[df.VariableID==vid]
           
            if len(df_tmp) == 1:
                df.loc[df_tmp.index[0],'Value'] = 0
                continue
            t_reset = df_tmp[(df_tmp.Value.diff()<0)|(df_tmp.index==df_tmp.index[0])].Datetime
            idx_not_reset = df_tmp[(df_tmp.Value.diff()>=0)&(df_tmp.index!=df_tmp.index[0])].index
            df_old = df_tmp.copy()
            for i in np.arange(1, len(t_reset)):
                if i == len(t_reset)-1:
                    idx_between_reset = df_tmp.index[df_tmp.Datetime>=t_reset.iloc[i]]
                else:
                    idx_between_reset = df_tmp.index[(df_tmp.Datetime>=t_reset.iloc[i])&(df_tmp.Datetime<t_reset.iloc[i+1])]
                df.loc[idx_between_reset, 'Value'] += df.loc[df_tmp.index[df_tmp.Datetime<t_reset.iloc[i]][-1], 'Value']
            df_tmp = df[df.VariableID==vid]
            tdiff = (df_tmp.Datetime.diff().iloc[1:] / np.timedelta64(3600, 's'))
            if (tdiff < 0.5).sum() > 0:
                df.drop(df_tmp.index[1:][tdiff.values<0.5], inplace=True)


            df_tmp = df[df.VariableID==vid]
            vdiff = df_tmp.Value.diff().iloc[1:]
            # if (vdiff < 0).sum() > 0:
            #     import ipdb
            #     ipdb.set_trace()
            #     df.drop(df_tmp.index[:-1][vdiff.values<0], inplace=True)


        gc.collect()
        
        for vid in df_urine.VariableID.unique():
            df_tmp = df[df.VariableID==vid]
            if len(df_tmp) == 1:
                df.loc[df_tmp.index[0],'Value'] = 0
                continue
            idx_not_reset = df_tmp[df_tmp.Value.diff()>=0].index
            tdiff = (df_tmp.Datetime.diff().iloc[1:] / np.timedelta64(3600, 's'))
            # df.loc[df_tmp.index[:-1], 'Value'] = (np.diff(df_tmp.Value.values) / tdiff).values
            # df.loc[df_tmp.index[-1], 'Value'] = df.loc[df_tmp.index[-2], 'Value']
            df.loc[df_tmp.index[1:], 'Value'] = (np.diff(df_tmp.Value.values) / tdiff).values
            df.loc[df_tmp.index[0], 'Value'] = df.loc[df_tmp.index[1], 'Value']
                                                             
        # if (df[df.VariableID.isin(variable_ids)].Value<0).sum() > 0 :
        #     import ipdb
        #     ipdb.set_trace()
        return df

def convert_urine_variables(df, hr_start, adm):
    
    df_urine = df[df.VariableID.isin(cumul_urine_id_lst)]

    if len(df_urine) == 0:
        return df, df_urine[['PatientID', 'Datetime', 'Value']].copy()
    else:
        for vid in df_urine.VariableID.unique():
            df_tmp = df_urine[df_urine.VariableID==vid]
            if df.iloc[0].PatientID in hr_start.index and df.iloc[0].PatientID in adm.index:
                adm_time = min(hr_start.loc[df.iloc[0].PatientID].Datetime, adm.loc[df.iloc[0].PatientID].AdmissionTime)
            elif df.iloc[0].PatientID not in hr_start.index and df.iloc[0].PatientID in adm.index:
                adm_time = adm.loc[df.iloc[0].PatientID].AdmissionTime
            elif df.iloc[0].PatientID in hr_start.index and df.iloc[0].PatientID not in adm.index:
                adm_time = hr_start.loc[df.iloc[0].PatientID].Datetime
            else:
                adm_time = df_tmp.iloc[0].Datetime
            nrec_pre_hr = (df_tmp.Datetime < adm_time).sum()
            if nrec_pre_hr == 0:
                pass
            elif nrec_pre_hr ==1:
                df.loc[df_tmp.index[0], 'Datetime'] = adm_time
            else:
                df.drop(df_tmp.index[df_tmp.Datetime < adm_time][:-1], inplace=True)
                df.loc[df_tmp.index[df_tmp.Datetime < adm_time][-1], 'Datetime'] = adm_time
            df_tmp = df[df.VariableID==vid]
            df.drop(df_tmp.index[df_tmp.Datetime.duplicated(keep='last')], inplace=True) 
            df_tmp = df[df.VariableID==vid]
           
            if len(df_tmp) == 1:
                df.loc[df_tmp.index[0],'Value'] = 0
                continue
            t_reset = df_tmp[(df_tmp.Value.diff()<0)|(df_tmp.index==df_tmp.index[0])].Datetime
            idx_not_reset = df_tmp[(df_tmp.Value.diff()>=0)&(df_tmp.index!=df_tmp.index[0])].index
            df_old = df_tmp.copy()
            for i in np.arange(1, len(t_reset)):
                if i == len(t_reset)-1:
                    idx_between_reset = df_tmp.index[df_tmp.Datetime>=t_reset.iloc[i]]
                else:
                    idx_between_reset = df_tmp.index[(df_tmp.Datetime>=t_reset.iloc[i])&(df_tmp.Datetime<t_reset.iloc[i+1])]
                df.loc[idx_between_reset, 'Value'] += df.loc[df_tmp.index[df_tmp.Datetime<t_reset.iloc[i]][-1], 'Value']
            df_tmp = df[df.VariableID==vid]
            tdiff = (df_tmp.Datetime.diff().iloc[1:] / np.timedelta64(3600, 's'))
            if (tdiff < 0.5).sum() > 0:
                df.drop(df_tmp.index[1:][tdiff.values<0.5], inplace=True)


            df_tmp = df[df.VariableID==vid]
            vdiff = df_tmp.Value.diff()
            if (vdiff < 0).sum() > 0:
                import ipdb
                ipdb.set_trace()
                df.drop(df_tmp.index[:-1][vdiff.values<0], inplace=True)


        gc.collect()
        df_urine = df[df.VariableID.isin(cumul_urine_id_lst)].copy()
        if len(df_urine.VariableID.unique()) > 1:
            df_pivot = pd.pivot_table(df_urine, values='Value', index=['PatientID', 'Datetime'], columns='VariableID')
            df_urine_all = df_pivot.interpolate(method='linear').sum(axis=1).reset_index()
            df_urine_all.rename(columns={df_urine_all.columns[2]: 'Value'}, inplace=True)
        
        for vid in df_urine.VariableID.unique():
            df_tmp = df[df.VariableID==vid]
            if len(df_tmp) == 1:
                df.loc[df_tmp.index[0],'Value'] = 0
                continue
            idx_not_reset = df_tmp[df_tmp.Value.diff()>=0].index
            tdiff = (df_tmp.Datetime.diff().iloc[1:] / np.timedelta64(3600, 's'))
            # df.loc[df_tmp.index[:-1], 'Value'] = (np.diff(df_tmp.Value.values) / tdiff).values
            # df.loc[df_tmp.index[-1], 'Value'] = df.loc[df_tmp.index[-2], 'Value']
            df.loc[df_tmp.index[1:], 'Value'] = (np.diff(df_tmp.Value.values) / tdiff).values
            df.loc[df_tmp.index[0], 'Value'] = df.loc[df_tmp.index[1], 'Value']
            dict_interval_urine_meas[vid].extend(tdiff.loc[idx_not_reset])
                                                 
                                                
        if len(df_urine.VariableID.unique()) > 1:
            tdiff = (df_urine_all.Datetime.diff().iloc[1:] / np.timedelta64(3600, 's'))
            # df_urine_all.loc[df_urine_all.index[:-1], 'Value'] = (np.diff(df_urine_all.Value.values) / tdiff).values
            # df_urine_all.loc[df_urine_all.index[-1], 'Value'] = df_urine_all.loc[df_urine_all.index[-2], 'Value']
            df_urine_all.loc[df_urine_all.index[1:], 'Value'] = (np.diff(df_urine_all.Value.values) / tdiff).values
            df_urine_all.loc[df_urine_all.index[0], 'Value'] = df_urine_all.loc[df_urine_all.index[1], 'Value']
        else:
            df_urine_all = df[df.VariableID.isin(cumul_urine_id_lst)][['PatientID', 'Datetime', 'Value']].copy()
            
        if (df[df.VariableID.isin(cumul_urine_id_lst)].Value<0).sum() > 0 :
            import ipdb
            ipdb.set_trace()
        return df, df_urine_all

def remove_records_with_invalid_status(df, tbl):
    status_set = df.Status.unique()
    if tbl == 'pharmarec':
        status_binary = ['{0:10b}'.format(s)[::-1] for s in status_set]
    else:
        status_binary = ['{0:11b}'.format(s)[::-1] for s in status_set]
    invalid_status_set = status_set[np.where( [x[1]=='1' for x in status_binary])]
    if len(invalid_status_set) > 0:
        df.drop(df.index[df.Status.isin(invalid_status_set)], inplace=True)
    return df


def change_arterial_to_venous(labres, voi, tbl, batch_id, monvals_svo2_path):
    vid_arterial_to_venous = {24000529: 24000740, 20001300: 24000739, 24000526: 24000837, 24000548: 24000836, 
                              20004200: 24000731, 24000867: 24000833, 24000524: 24000732, 24000866: 24000835, 
                              24000525: 24000735, 24000513: 24000733, 24000514: 24000736, 24000512: 24000734, 
                              20000800: 24000737, 24000530: 24000738}
    arterial_vids_with_venous = [key for key in vid_arterial_to_venous.keys()]
    arterial_vids_wo_venous = [20000200, 20000300, 20001200, 24000426, 24000521, 24000522, 24000549]
    
    patient_ids = labres.PatientID.unique()
    if 20000800 not in labres.VariableID.unique():
        return labres
    svo2 = pd.read_hdf(monvals_svo2_path, where='PatientID=%d'%labres.iloc[0].PatientID, mode='r')
    if len(svo2) == 0:
        return labres
    
    sao2_id = 20000800
    sao2_dt = labres[labres.VariableID==sao2_id].Datetime.values.reshape((-1, 1))
    sao2_dt_tiled = np.tile(sao2_dt, (1, len(svo2)))
    svo2_dt = svo2.Datetime.values.reshape((1, -1))
    tdiff = np.abs(sao2_dt_tiled - svo2_dt) / np.timedelta64(1, 'm')
    tdiff_min = np.min(tdiff, axis=1)
    tdiff_argmin = np.argmin(tdiff, axis=1)

    if len(np.where(tdiff_min <= 4)[0])==0:
        return labres

    for i in np.where(tdiff_min <= 4)[0]:
        svo2_dt_tmp = svo2_dt[0,tdiff_argmin[i]]
        svo2_value = svo2[svo2.Datetime==svo2_dt_tmp].iloc[0].Value

        sao2_dt_tmp = sao2_dt[i,0]
        sao2_et_tmp = labres[np.logical_and(labres.VariableID==sao2_id, labres.Datetime==sao2_dt_tmp)].Entertime
        for et in sao2_et_tmp:
            sao2_value = labres[np.logical_and(labres.VariableID==sao2_id, labres.Entertime==et)].iloc[0].Value

            if (sao2_value - svo2_value) / 8.454 < 0.1:
                print('arterial_to_venous, patient %d'%patient_ids[0])
                print('patient %d: time %s'%(patient_ids[0], et))
                # look for all variables that is within 2 min window centered at the current datetime
                possible_venous = labres[labres.Datetime==sao2_dt_tmp]

                possible_venous = possible_venous[np.logical_and(possible_venous.Entertime >= et-np.timedelta64(5, 's'), 
                                                                 possible_venous.Entertime <= et+np.timedelta64(5, 's'))]
                idx_no_venous = possible_venous.index[possible_venous.VariableID.isin(arterial_vids_wo_venous)]
                if len(idx_no_venous) > 0:
                    labres.drop(idx_no_venous, inplace=True)
                possible_venous = possible_venous[possible_venous.VariableID.isin(arterial_vids_with_venous)]
                assert(np.sum(possible_venous.VariableID.value_counts()>1)==0)
                if len(possible_venous) > 0:
                    labres.loc[possible_venous.index, 'VariableID'] = possible_venous.VariableID.apply(lambda x: vid_arterial_to_venous[x])

    return labres

def remove_out_of_range(df, val_range, tbl, batch_id):
    vid_list = df.VariableID.unique()
    if 120 in vid_list or 170 in vid_list:
        for vid in [120, 170]:
            tmp = df[df.VariableID==vid]
            if len(tmp) == 0:
                continue
            et_oor = tmp[np.logical_or(tmp.Value > val_range.loc[vid].UpperBound, 
                                       tmp.Value < val_range.loc[vid].LowerBound)].Datetime
            for et in et_oor:
                df_dt_oor = df[np.logical_and(df.Datetime >= et - np.timedelta64(30,'s'),
                                              df.Datetime <  et + np.timedelta64(30,'s'))]
                
                df_spike = df.loc[df_dt_oor.index[df_dt_oor.VariableID.isin([vid-20, vid-10, vid])]].copy()
                df.drop(df_dt_oor.index[df_dt_oor.VariableID.isin([vid-20, vid-10, vid])], inplace=True)

    for vid in df.VariableID.unique():
        if vid in val_range.index[val_range.LowerBound.notnull()|val_range.UpperBound.notnull()]:
            tmp = df[df.VariableID==vid]
            if vid == 15004752:
                if tmp.iloc[0].Datetime <= np.datetime64('2016-11-08'):
                    ubound = 18
                    lbound = 0
                else:
                    ubound = 100
                    lbound = 21
                index_oor = tmp.index[np.logical_or(tmp.Value > ubound, 
                                                    tmp.Value < lbound)]
            else:
                index_oor = tmp.index[np.logical_or(tmp.Value > val_range.loc[vid].UpperBound, 
                                                    tmp.Value < val_range.loc[vid].LowerBound)]
            if len(index_oor) > 0:
                df.drop(index_oor, inplace=True) 
        else:
            pass
    gc.collect()
    return df

def increase_categorical_counter_to_merge(df):
    """
    The urine cluture variable will be merged with the (categorical) blood cluture location variable.
    The blood cluture location varibale conatins values 0-4. Urine contains values 1-2
    The urine values will be incremented by 4
    """
    df.loc[df.VariableID == 15002175,'Value'] = df.loc[df.VariableID == 15002175,'Value'].astype(float)+4
    return df


def correct_weight_height(df):
    # Height:10000450; Weight:10000400
    cols = {'Height': 10000450, 'Weight': 10000400}

    # delete height record if the height is higher than 220 cm
    df.loc[df.index[np.logical_and(df.VariableID==10000450, df.Value>10000)], 'Value'] /= 100
    df.drop(df.index[np.logical_and(df.VariableID==10000450, df.Value>240)], inplace=True)
    # delete weight record if the weight is heavier than 500 kg
    df.drop(df.index[np.logical_and(df.VariableID==10000400, df.Value>500)], inplace=True)

    df_tmp = df[df.VariableID.isin([10000450, 10000400])]
    if len(df_tmp)== 0:
        return df

    height_weight = pd.pivot_table(df_tmp.copy(), values='Value', index='Entertime', columns='VariableID')
    if height_weight.shape[1] == 1:
        if height_weight.columns[0] == cols['Height']:
            df.drop(df.index[np.logical_and(df.VariableID==10000450, df.Value<130)], inplace=True)
        else:
            pass
        return df
            
    imputed = height_weight.copy().fillna(method='ffill')
    imputed.fillna(method='bfill', inplace=True)
    for i in range(len(imputed)):
        height = imputed.iloc[i][cols['Height']]
        weight = imputed.iloc[i][cols['Weight']]

        if height <= 120 and weight >= 120:
            imputed.loc[imputed.index[i], cols['Weight']] = height
            imputed.loc[imputed.index[i], cols['Height']] = weight
            height = weight
            weight = imputed.loc[imputed.index[i], cols['Weight']]

        elif height < 100 and height >= 30:
            height += 100
            imputed.loc[imputed.index[i], cols['Height']] = height


        if height < 130:
            imputed.loc[imputed.index[i], cols['Height']] = float('NaN')
        else:
            bmi = weight / ((height/100)**2)
            if bmi > 60 or bmi < 10:
                imputed.loc[imputed.index[i], cols['Weight']] = float('NaN')
                imputed.loc[imputed.index[i], cols['Height']] = float('NaN')
            
            
    for i in range(len(df_tmp)):
        vid = df_tmp.iloc[i].VariableID
        df.loc[df_tmp.index[i], 'Value'] = imputed[imputed.index==df_tmp.iloc[i].Entertime][vid].values[0]

    df.dropna(how='any', inplace=True)
    return df   


def main():

    dict_type = dict(monvals='Monitored', dervals='Derived', observrec='Observed',
                     pharmarec='Pharma')
    # Load the global std values for all variables
    
    voi = pd.read_csv(os.path.join(voi_path, 'labref_excel_%s.tsv'%version if tbl=='labres' else 'varref_excel_%s.tsv'%version), 
                      sep='\t', encoding='cp1252')
    if tbl!='labres':
        voi = voi[voi.Type==dict_type[tbl]]
    voi.drop(voi.index[(voi.VariableID.isnull())|(voi.VariableID=='Vm23')], inplace=True)
    voi.drop_duplicates('VariableID', inplace=True)
    voi.loc[:,'VariableID'] = voi.VariableID.astype(np.int64)
    voi.set_index('VariableID', inplace=True)
    
    if tbl in ['observrec', 'dervals']:
        monvals_path = input_path.replace(tbl, 'monvals')
        df_hr_tstart = pd.read_hdf(monvals_path, where='VariableID=200', mode='r').drop_duplicates('PatientID', keep='first').set_index('PatientID')


    df_idx_start = 0
    num_pid = len(pid_list)
    df_height = []
    df_urine_sum = []
    cnt_pid_urine = 0
    for nn, pid in enumerate(pid_list):
        df = pd.read_hdf(input_path, where='PatientID=%d'%pid, mode='r')
        if len(df) == 0:
            print('Patient', pid, 'have no data in %s'%tbl)
            continue
            
        # rename columns of the pharmarec table
        if tbl =="pharmarec":
            df.rename(columns={'PharmaID': 'VariableID', 'GiveDose': 'Value', 'DateTime': 'Datetime', 
                               'SampleTime': 'Entertime'}, inplace=True)
        else:
            df.rename(columns={'EnterTime': 'Entertime'}, inplace=True)

        # select only variables of interest
        vid_intersect = set(df.VariableID) & set(voi.index)

        # add height (10000450) to the variables of interests if the table is 'observrec'
        if tbl == 'observrec':
            vid_intersect |= {10000450}
        elif tbl == 'dervals':
            vid_intersect |= {830005420, 30015110, 30015010, 30015075, 30015080}

        df.drop(df.index[~df.VariableID.isin(vid_intersect)], inplace=True)
        gc.collect()

        if len(df) == 0:
            print('Patient', pid, 'have no data of interest in %s'%tbl)
            continue

        # Only remove value 0 for variables for which 0 doesn't have any clinical meaning

        # remove records with status containing 2 (invalidated)
        df = remove_records_with_invalid_status(df, tbl)

        df.sort_values(by=['Datetime', 'VariableID', 'Entertime'], inplace=True)

        # remove identical records
        df.drop_duplicates(['Datetime', 'VariableID', 'Value', 'Status'], inplace=True)

        if tbl == 'labres':
            monvals_svo2_path = input_path.replace(tbl, 'monvals_svo2')
            df = change_arterial_to_venous(df, voi, tbl, batch_id, monvals_svo2_path)
            df.drop_duplicates(['Datetime', 'VariableID', 'Value'], inplace=True)
            # fixed troponin conversion
            if 24000538 in df.VariableID.unique():
                df.loc[df.index[df.VariableID==24000538], 'Value'] = df[df.VariableID==24000538].Value.values * 1000
            if 24000806 in df.VariableID.unique():
                if df[df.VariableID==24000806].Datetime.min() <= np.datetime64('2016-05-01'):
                    idx_to_convert = df.index[np.logical_and(df.VariableID==24000806, df.Datetime <= np.datetime64('2016-05-01'))]
                    df.loc[idx_to_convert,'Value'] = df.loc[idx_to_convert, 'Value'] * 1000

        elif tbl == 'dervals':
            cumulative_variable_ids = set(voi.index[voi.VariableName.apply(lambda x: 'cumul' in x or '/c' in x)])
            cumulative_variable_ids &= set(df.VariableID.tolist())
            if len(cumulative_variable_ids) == 0:
                pass
                # print('Patient', pid, 'does not have cumulative dervals variables.')
            else:
                cumulative_variable_ids = np.sort(list(cumulative_variable_ids))
                df = convert_non_urine_variables(df, cumulative_variable_ids, df_hr_tstart, gd)

        elif tbl == 'observrec':
            df = correct_weight_height(df)
            df_height.append(df[df.VariableID==10000450].copy())
            df.sort_values(['VariableID', 'Datetime', 'Value'], inplace=True)
            df.reset_index(inplace=True, drop=True)
            df, tmp_urine_sum = convert_urine_variables(df, df_hr_tstart, gd)
            cumulative_variable_ids = set(voi.index[voi.VariableName.apply(lambda x: 'cumul' in x or '/c' in x)])
            cumulative_variable_ids -= set(cumul_urine_id_lst)
            cumulative_variable_ids &= set(df.VariableID.tolist())
            if len(cumulative_variable_ids) == 0:
                pass
                # print('Patient', pid, 'have cumulative observrec variables.')
            else:
                df = convert_non_urine_variables(df, cumulative_variable_ids, df_hr_tstart, gd)
                assert(df.Value.max()!=float('Inf'))
                df.drop(df.index[df.VariableID==10000450], inplace=True)
                df = increase_categorical_counter_to_merge(df)
        
            if len(tmp_urine_sum) > 0:
                df_urine_sum.append(tmp_urine_sum)
        df = remove_out_of_range(df, voi, tbl, batch_id)
        df.drop(df.index[~df.VariableID.isin(vid_intersect)], inplace=True)
        df.set_index(np.arange(df_idx_start, df_idx_start+len(df)), drop=True, inplace=True)

        df_idx_start += len(df)
            
        if write_to_disk:
            df.loc[:, 'VariableID'] = df.VariableID.astype(np.int64)
            df.to_hdf(output_path, 'data', append=True, format='table', data_columns=['PatientID', 'VariableID'], complevel=5, complib='blosc:lz4')

        # sys.stdout.write('# patients processed: %3d / %3d\r'%(nn+1, len(pid_list)))
        # sys.stdout.flush()
        gc.collect()
        
    if tbl == 'observrec' and write_to_disk:
        height_info_path = output_path.replace(tbl, 'height')
        df_height = pd.concat(df_height, axis=0)
        df_height.reset_index(inplace=True, drop=True)
        df_height.to_hdf(height_info_path, 'data', data_columns=['PatientID'], complevel=5, complib='blosc:lz4')

        urine_sum_info_path = output_path.replace(tbl, 'urine_sum')
        df_urine_sum = pd.concat(df_urine_sum, axis=0)
        df_urine_sum.reset_index(inplace=True, drop=True)
        df_urine_sum.to_hdf(urine_sum_info_path, 'data', data_columns=['PatientID'], complevel=5, complib='blosc:lz4')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-tbl')
    parser.add_argument('-input_path')
    parser.add_argument('-output_path')
    parser.add_argument('-chunking_info_file')
    parser.add_argument('-voi_path')
    parser.add_argument('-batch_id', type=int)
    parser.add_argument('-version')
    parser.add_argument('--write_to_disk', action='store_true')
    args = parser.parse_args()

    tbl = args.tbl
    input_path = args.input_path
    output_path = args.output_path
    chunking_info_file = args.chunking_info_file
    voi_path = args.voi_path
    batch_id = args.batch_id
    version = args.version
    write_to_disk = args.write_to_disk


    chunking_info = pd.read_csv(chunking_info_file)
    chunking_info.rename(columns={'ChunkfileIndex': 'BatchID'}, inplace=True)
    if 'PatientID' in chunking_info.columns:
        pid_list = chunking_info.PatientID[chunking_info.BatchID==batch_id].values
    else:
        pid_list = chunking_info.index[chunking_info.BatchID==batch_id].values

    if 'hirid2' not in input_path:
        gd = pd.read_hdf(os.path.join("/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/1_hdf5_consent/200831", 'generaldata.h5'), mode='r')
    else:
        tbl_path = '/cluster/work/grlab/clinical/hirid2/pg_db_export/p_generaldata'
        gd_file = [f for f in os.listdir(tbl_path) if 'crc' not in f and 'parquet' in f][0]
        gd = pd.read_parquet(os.path.join(tbl_path, gd_file), engine='pyarrow')
        gd.loc[:,'admissiontime'] = pd.to_datetime(gd.admissiontime).dt.floor(freq='s')
        gd.rename(columns={'patientid': 'PatientID', 'birthyear': 'birthYear', 'sex': 'Sex',
                               'admissiontime': 'AdmissionTime', 'status': 'Status', 'patgroup': 'PatGroup'}, inplace=True)
        gd.set_index('PatientID', inplace=True)
    gc.collect()

    input_path = os.path.join(input_path, tbl)
    input_path = os.path.join(input_path, [f for f in os.listdir(input_path) if '%s_%d_'%(tbl, batch_id) in f][0])
    output_path = os.path.join(output_path, tbl)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, '%s_%d_%d--%d.h5'%(tbl, 
                                                               batch_id, 
                                                               np.min(pid_list), 
                                                               np.max(pid_list)))
    if write_to_disk and os.path.exists(output_path):
        print('Are you sure you want to append to the current existing file?')
        print('Please delete before rewrite')
        print(output_path)
    
    main()
    # with open('meas_interval_%d.pkl'%batch_id, 'wb') as f:
    #     pickle.dump(dict_interval_urine_meas, f)
