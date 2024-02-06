#!/usr/bin/env python
import os
import gc
import sys
from copy import copy
import pandas as pd
import numpy as np

def date_correction(new_year, new_month, new_day, old_month):
    month30 = [4, 6, 9, 11]
    month31 = [1, 3, 5, 7, 8, 10, 12]
    is_leap_year = lambda year: year%4 == 0
    
    # Correct the day value if the new day is equal to 0 or larger than 31.
    if new_day == 0:
        new_month-=1
        if new_month in month30: 
            new_day = 30 
        elif new_month in month31:
            new_day = 31
        else:
            new_day = 29 if is_leap_year(new_year) else 28

    elif new_day > 31:
        if new_month in month30:       
            new_day -= 30 
        elif new_month in month31:
            new_day -= 31
        else:
            new_day -= 29 if is_leap_year(new_year) else 28
        new_month+=1
            
    # Correct the month value is the new month is non-positive or larger than 12
    if new_month <=0:
        new_month += 12
        new_year -= 1
        
    elif new_month > 12:
        new_month = new_month%12
        new_year += 1
    
    # If the old date is the end of the old month, make the new date the end of the new month to be consistent
    if new_day in [30, 31] and new_month == 2:
        new_day = 29 if is_leap_year(new_year) else 28
        
    elif new_day == 31 and new_month in month30:
        new_day = 30
        
    elif new_day == 30 and new_month in month31 and old_month in month30:
        new_day = 31
        
    elif new_day == 28 and new_month!=2 and old_month == 2:
        new_day = 31 if new_month in month31 else 30
        
    return new_year, new_month, new_day


def DISPLAY_BEFORE(idx, show_case, case, where='pre'):
    if show_case in [0, case]:
        print('===== Patient %d | CASE %d | Admission Time %s ====='%(pid, case, gd.loc[pid].AdmissionTime))
        print('----- BEFORE -----')
        if type(idx) in [int, np.int64]:
            idx = [idx]
        for i in idx:
            if where=='pre':
                display(df.iloc[:i+6])
            elif where=='center':
                display(df.iloc[i-5:i+6] if i>5 else df.iloc[:i+6])
            elif where=='post':
                display(df.iloc[i-5:])
        if tbl == 'pharmarec':
            for i in idx:
                vid = df.iloc[i+1].VariableID
                iid = df.iloc[i+1].InfusionID
                display(df[np.logical_and(df.VariableID==vid, df.InfusionID==iid)])


def DISPLAY_AFTER(idx, show_case, case, where='pre'):
    if show_case in [0, case]:
        print('----- AFTER -----')
        if type(idx) in [int, np.int64]:
            idx = [idx]
        for i in idx:
            if where=='pre':
                display(df.iloc[:i+6])
            elif where=='center':
                display(df.iloc[i-5:i+6] if i>5 else df.iloc[:i+6])
            elif where=='post':
                display(df.iloc[i-5:])
        if tbl == 'pharmarec':
            for i in idx:
                vid = df.iloc[i+1].VariableID
                iid = df.iloc[i+1].InfusionID
                display(df[np.logical_and(df.VariableID==vid, df.InfusionID==iid)])

fix_year = lambda x, new_year: x.replace(year=new_year)
fix_month = lambda x, new_month: x.replace(month=new_month)


def fix_df_with_very_long_gap(show_case=None):
    AdmTime = np.datetime64(gd.loc[pid].AdmissionTime.date())
    index_long_gap = np.where(diff_dt > 31)[0]
        
    if len(index_long_gap)==1:
        # If there is only one such gap 
        idx = index_long_gap[0]
        
        # Compute the difference in the year, month and day of the Datetime right before and right after the gap
        dt_gap_yr = df.iloc[idx+1].Datetime.year - df.iloc[idx].Datetime.year 
        dt_gap_mo = df.iloc[idx+1].Datetime.month - df.iloc[idx].Datetime.month
        dt_gap_dd = df.iloc[idx+1].Datetime.day - df.iloc[idx].Datetime.day    

        # Datetime and Entertime difference before the gap
        be_dt_et_yr = df.iloc[idx].Datetime.year - df.iloc[idx].Entertime.year
        be_dt_et_mo = df.iloc[idx].Datetime.month - df.iloc[idx].Entertime.month
        be_dt_et_dd = df.iloc[idx].Datetime.day - df.iloc[idx].Entertime.day
        be_dt_et_hr = df.iloc[idx].Datetime.hour - df.iloc[idx].Entertime.hour

        # Datetime and Entertime difference after the gap
        af_dt_et_yr = df.iloc[idx+1].Datetime.year - df.iloc[idx+1].Entertime.year
        af_dt_et_mo = df.iloc[idx+1].Datetime.month - df.iloc[idx+1].Entertime.month
        af_dt_et_dd = df.iloc[idx+1].Datetime.day - df.iloc[idx+1].Entertime.day

        # DT is short for Datetime, ET for Entertime and AT for AdmissionTime
        if be_dt_et_yr!=0 and be_dt_et_mo == 0:
            # Case 1: Different in year, the same in month for DT and ET before the gap
            # Solution: Replace the DT year with the ET year in records before the gap
            case = 1
            DISPLAY_BEFORE(idx, show_case, case)

            new_yr = df.iloc[idx].Entertime.year
            idx2fix = df.index[:idx+1]
            if df.loc[idx2fix,'Datetime'].apply(fix_year, args=(new_yr,)).max() - df.Datetime.max() < 7 * sec_of_day:
                df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_year, args=(new_yr,))
            elif be_dt_et_dd > 28:
                for tmp in idx2fix:
                    new_mo = df.loc[tmp,'Entertime'].month-1
                    if new_mo == 0:
                        new_mo = 12
                    df.loc[tmp,'Datetime'] = df.loc[tmp,'Datetime'].replace(month=new_mo)
            elif be_dt_et_mo==0 and be_dt_et_dd==0 and np.abs(be_dt_et_hr)<2:
                for tmp in idx2fix:
                    if ((df.loc[tmp].Datetime.month == df.loc[tmp].Entertime.month) and 
                        (df.loc[tmp].Datetime.day == df.loc[tmp].Entertime.day) and 
                        (np.abs(df.loc[tmp].Datetime.hour - df.loc[tmp].Entertime.hour)<2)):
                        new_yr = df.loc[tmp,'Entertime'].year
                        df.loc[tmp,'Datetime'] = df.loc[tmp,'Datetime'].replace(year=new_yr)
                    elif (np.abs((df.loc[tmp].Datetime.month - df.loc[tmp].Entertime.month))==11 and 
                          (np.abs(df.loc[tmp].Datetime.day == df.loc[tmp].Entertime.day)%10 in [0,9])):
                        new_yr = df.loc[tmp,'Entertime'].year-1
                        df.loc[tmp,'Datetime'] = df.loc[tmp,'Datetime'].replace(year=new_yr)
            else:
                df.drop(df.index[:idx+1], inplace=True)
            
            DISPLAY_AFTER(idx, show_case, case)
        
        elif af_dt_et_yr!=0 and af_dt_et_mo == 0:
            # Case 2: Different in year, the same in month for DT and ET after the gap
            # Solution: Replace the DT year with the ET year in records after the gap
            case = 2
            DISPLAY_BEFORE(idx, show_case, case)
            if df.iloc[idx+1].Entertime.year!=df.iloc[idx].Entertime.year:
                df.drop(df.index[:idx+1], inplace=True)
            else:
                new_yr = df.iloc[idx+1].Entertime.year
                idx2fix = df.index[idx+1:]
                df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_year, args=(new_yr,))

            DISPLAY_AFTER(idx, show_case, case)
        
        elif be_dt_et_yr!=0 and df.iloc[idx].Entertime.year==df.iloc[idx+1].Datetime.year:
            # Case 3: Different DT year and ET year before the gap. And the ET year before the gap is the same 
            #         as the DT year after the gap, which means the ET is continuous when there is a gap in DT.
            # Solution: Delete records with DT before the AT.
            case = 3
            DISPLAY_BEFORE(idx, show_case, case)
            if be_dt_et_mo == -1 and be_dt_et_dd in [29,30]:
                new_yr = df.iloc[idx].Entertime.year
                df.loc[df.index[idx], 'Datetime'] = df.loc[df.index[idx], 'Datetime'].replace(year=new_yr)
            elif tbl!='labres':                
                df.drop(df.index[df.Datetime<AdmTime], inplace=True)
            
            DISPLAY_AFTER(idx, show_case, case)
        
        elif dt_gap_yr!=0 and dt_gap_mo == 0 and af_dt_et_yr==0:
            # Case 4: The gap is only caused by DT year difference between two consecutive records, and the DT 
            #         and ET year after the gap is consistent. 
            # Solution: Replace the DT year in records before the gap with the DT year after the gap.
            case = 4
            DISPLAY_BEFORE(idx, show_case, case)
            
            new_yr = df.iloc[idx+1].Datetime.year
            idx2fix = df.index[:idx+1]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_year, args=(new_yr,))                    
            
            DISPLAY_AFTER(idx, show_case, case)
        
        elif dt_gap_yr!=0 and dt_gap_mo == 0 and be_dt_et_yr==0:
            # Case 5: The gap is only caused by DT year difference between two consecutive records, and the DT 
            #         and ET year before the gap is consistent. 
            # Solution: Replace the DT year in records after the gap with the DT year before the gap.
            case = 5
            DISPLAY_BEFORE(idx, show_case, case)
                
            new_yr = df.iloc[idx].Datetime.year
            idx2fix = df.index[idx+1:]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_year, args=(new_yr,)) 
            
            DISPLAY_AFTER(idx, show_case, case)
        
        elif be_dt_et_yr!=0 or af_dt_et_yr!=0 or dt_gap_yr!=0:
            # ???
            case = 16
            DISPLAY_BEFORE(idx, show_case, case, where='center')
                
            if idx / len(df) > 0.95:
                df.drop(df.index[idx+1:], inplace=True)
            else:
                df.drop(df.index[(df.Datetime-gd.loc[pid,'AdmissionTime'])/sec_of_day<-1], inplace=True)
            
            DISPLAY_AFTER(idx, show_case, case, where='center')
        
        elif be_dt_et_mo!=0 and np.abs(be_dt_et_dd)<2:
            # All the previous cases have coverd case where there is a year difference, from now on, all cases
            # do not have year difference.
            # Case 6: When there is month difference in DT and ET and the absolute day difference is smaller 
            #         2 days before the gap, it is very likely there is a typo in the month before the gap.
            # Solution: Replace the DT month with the ET month in records before the gap.
            case = 6
            DISPLAY_BEFORE(idx, show_case, case)
            if idx / len(df) < 0.5:
                new_mo = df.iloc[idx].Entertime.month
                if tbl in ['monvals', 'observrec']:
                    if df.iloc[idx].Datetime.hour - df.iloc[idx].Entertime.hour > 22:
                        new_dd = df.iloc[idx].Datetime.day
                    else:
                        new_dd = df.iloc[idx].Entertime.day
                    fix_func = lambda x: x.replace(month=new_mo, day=new_dd)
                else:
                    fix_func = lambda x: x.replace(month=new_mo)
                idx2fix = df.index[:idx+1]
                df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_func)
            else:
                new_mo = df.iloc[idx+1].Entertime.month
                if tbl in ['monvals', 'observrec']:
                    if df.iloc[idx].Datetime.hour - df.iloc[idx].Entertime.hour > 22:
                        new_dd = df.iloc[idx].Datetime.day
                    else:
                        new_dd = df.iloc[idx].Entertime.day
                    fix_func = lambda x: x.replace(month=new_mo, day=new_dd)
                else:
                    fix_func = lambda x: x.replace(month=new_mo)
                idx2fix = df.index[idx+1:]
                df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_func)
                
            
            DISPLAY_AFTER(idx, show_case, case)
        
        elif be_dt_et_mo!=0 and dt_gap_dd==0:
            # Case 7: When there is month difference in DT and ET before the gap and no day difference between
            #         the records before and after the gap, it is very likely there is a typo in the month before
            #         the gap. 
            # Solution: Replace the DT month before the gap with the DT month after the gap.
            case = 7
            DISPLAY_BEFORE(idx, show_case, case)
                
            new_mo = df.iloc[idx+1].Datetime.month
            idx2fix = df.index[:idx+1]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_month, args=(new_mo,))
            
            DISPLAY_AFTER(idx, show_case, case)
        
        elif af_dt_et_mo!=0 and af_dt_et_dd==0:
            # Case 8: When there is month difference and no day difference in DT and ET, it is very likely there 
            #         is a typo in the month after the gap.
            # Solution: Replace the DT month with the ET month in records before the gap.
            case = 8
            DISPLAY_BEFORE(idx, show_case, case, where='post')
                
            new_mo = df.iloc[idx+1].Entertime.month
            idx2fix = df.index[idx+1:]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_month, args=(new_mo,))
            
            DISPLAY_AFTER(idx, show_case, case, where='post')
        
        elif np.abs(be_dt_et_mo)==11:
            # Case 9: When the absolute month difference is 11 but no year difference before the gap, it means 
            #         that the records should be one year more to be continuous.
            # Solution: Add 1 year to the DT year before the gap.
            case = 9
            DISPLAY_BEFORE(idx, show_case, case)
                
            for idx2fix in df.index[:idx+1]:
                if df.loc[idx2fix,'Datetime'].month==df.loc[idx2fix,'Entertime'].month:
                    new_yr = df.loc[idx2fix].Entertime.year
                elif (df.loc[idx2fix,'Datetime'].month-df.loc[idx2fix,'Entertime'].month==-11 and
                      df.loc[idx2fix,'Datetime'].year==df.loc[idx2fix,'Entertime'].year):
                    new_yr = df.loc[idx2fix].Entertime.year+1
                df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(year=new_yr)
                    
                df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(year=new_yr)
            
            DISPLAY_AFTER(idx, show_case, case)

        elif np.abs(af_dt_et_mo)==11:
            # Case 10: When the absolute month difference is 11 but no year difference after the gap, it means 
            #          that the records should be one year less to be continuous.
            # Solution: Subtract 1 year to the DT year before the gap.
            case = 10
            DISPLAY_BEFORE(idx, show_case, case, where='post')
            new_yr = df.iloc[idx+1].Entertime.year - 1
            fix_func = lambda x: x.replace(year=new_yr, day=x.day+1 if x.day<=30 else x.day)
            idx2fix = df.index[idx+1:]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_func)
            
            DISPLAY_AFTER(idx, show_case, case, where='post')

        elif ( np.abs(df.iloc[idx].Datetime.month-df.iloc[idx+1].Datetime.day)%10==0 and 
               np.abs(df.iloc[idx].Datetime.day-df.iloc[idx+1].Datetime.month) <=1 ):
            # Case 11: When the month and day before the gap are flipped.
            # Solution: Replace the month and day in records before the gap with the month and day after the gap.
            case = 11
            DISPLAY_BEFORE(idx, show_case, case, where='center')
                
            new_mo = df.iloc[idx+1].Datetime.month
            new_dd = df.iloc[idx+1].Datetime.day
            fix_func = lambda x: x.replace(month=new_mo, day=new_dd)
            idx2fix = df.index[:idx+1]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_func)
            
            DISPLAY_AFTER(idx, show_case, case, where='center')

        elif (len(df) - idx - 1)/len(df) < 0.05:
            # Case 12: When there are very few records after the long gap
            case = 12
            DISPLAY_BEFORE(idx, show_case, case, where='center')
            if len(np.where(diff_dt>10)[0]) > 1:
                index_long_gap = np.where(diff_dt>10)[0]
                idx2drop = []
                for k, idx in enumerate(index_long_gap):
                    if (df.iloc[idx+1].Datetime.month==df.iloc[idx+1].Entertime.month and
                        df.iloc[idx+1].Datetime.hour==df.iloc[idx+1].Entertime.hour and
                        df.iloc[idx+1].Datetime.day-df.iloc[idx+1].Entertime.day in [29,30]):
                        new_mo = df.iloc[idx+1].Entertime.month-1
                        new_dd = df.iloc[idx+1].Datetime.day + 1
                        for idx2fix in df.index[idx+1:index_long_gap[k+1]+1]:
                            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(month=new_mo,
                                                                                            day=new_dd)
                    else:
                        idx2drop.extend(df.index[idx+1:index_long_gap[k+1]+1 if k<len(index_long_gap)-1 else len(df)])
                if len(idx2drop) > 0:
                    df.drop(idx2drop, inplace=True)                
            else:
                df.drop(df.index[idx+1:], inplace=True)                
            DISPLAY_AFTER(idx, show_case, case, where='center')

        elif df.iloc[0].Datetime >= AdmTime:
            # Case 13: When the DT of the first record is later than AT. 
            # Solution: Remove the records after the gap.
            case = 13
            DISPLAY_BEFORE(idx, show_case, case, where='post')
            if tbl  == 'monvals':
                df.drop(df.index[idx+1:], inplace=True)
            else:
                if idx / len(df) < 0.1:
                    df.drop(df.index[:idx+1], inplace=True)
                else:
                    for k in np.arange(idx+1, len(df)):
                        tmp_diff_et_dt_yr = df.iloc[k].Datetime.year - df.iloc[k].Entertime.year
                        tmp_diff_et_dt_mo = df.iloc[k].Datetime.month - df.iloc[k].Entertime.month
                        tmp_diff_et_dt_dd = df.iloc[k].Datetime.day - df.iloc[k].Entertime.day
                        idx2fix = df.index[k]
                        if tmp_diff_et_dt_yr == 0 and tmp_diff_et_dt_dd == 0:
                            new_mo = df.iloc[k].Entertime.month
                            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(month=new_mo)
                        elif np.abs(tmp_diff_et_dt_mo) > 9 and np.abs(tmp_diff_et_dt_dd) > 27:
                            new_yr = df.iloc[k].Entertime.year
                            new_mo = df.iloc[k].Entertime.month - 1
                            new_dd = df.iloc[k].Datetime.day
                            old_month = df.iloc[k].Datetime.month
                            new_yr, new_mo, new_dd = date_correction(new_yr, new_mo, new_dd, old_month)
                            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(year=new_yr,
                                                                                            month=new_mo,
                                                                                            day=new_dd)
                DISPLAY_AFTER(idx, show_case, case, where='post')
            
        else:
            # Case 14: Some really weird cases.
            case = 14
            DISPLAY_BEFORE(idx, show_case, case, where='center')
                    
            str_mo = '{0:02}'.format(df.iloc[idx].Datetime.month)
            str_dd = '{0:02}'.format(df.iloc[idx].Datetime.day)
            new_mo = int(str_mo[1]+str_dd[1])
            new_dd = int(str_mo[0]+str_dd[0])
            if np.abs(df.iloc[idx+1].Datetime.month-new_mo) <=1 and np.abs(df.iloc[idx+1].Datetime.day-new_dd) <= 1:
                new_mo = df.iloc[idx+1].Datetime.month
                fix_func = lambda x: x.replace(month=new_mo, day=new_dd)
                idx2fix = df.index[:idx+1]
                df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_func)
            elif tbl!='labres':
                df.drop(df.index[df.Datetime<AdmTime], inplace=True)                
                DISPLAY_AFTER(idx, show_case, case, where='post')


    else:
        # Case 15: A case where there are two gaps.
        # Solution: Remove the records before AT
        case = 15
        DISPLAY_BEFORE(index_long_gap, show_case, case, where='center')
        if np.sum(index_long_gap / len(df) > 0.95) > 0:
            idx = index_long_gap[index_long_gap / len(df) > 0.95][0]
            df.drop(df.index[idx+1:], inplace=True)
        if len(index_long_gap[index_long_gap / len(df) <= 0.95]) == 1:
            idx = index_long_gap[index_long_gap / len(df) <= 0.95][0]
            if (np.abs(df.iloc[idx].Datetime.year-gd.loc[pid,'AdmissionTime'].year)==1 and 
                df.iloc[idx].Datetime.month==gd.loc[pid,'AdmissionTime'].month):
                new_yr = gd.loc[pid,'AdmissionTime'].year
                for idx2fix in df.index[:idx+1]:
                    df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(year=new_yr)
        elif tbl == 'labres':
            pass
        else:
            df.drop(df.index[df.Datetime<AdmTime], inplace=True)
    return df, case


def fix_df_with_long_gap(show_case=None):
    sec_of_hour = np.timedelta64(3600,'s')
    AdmTime = np.datetime64(gd.loc[pid].AdmissionTime.date())
    index_long_gap = np.where(np.logical_and(diff_dt>1, diff_dt<=31))[0]
    index_gap2drop = []

    for k, idx in enumerate(index_long_gap):
        if (np.abs(df.iloc[idx].Datetime-df.iloc[idx].Entertime)/sec_of_hour<=24 and 
            np.abs(df.iloc[idx+1].Datetime-df.iloc[idx+1].Entertime)/sec_of_hour<=24):
            # When the difference in DT and ET of the record before and after the gap is reasonably 
            # smaller than a day, it means that the gap is reasonable and not caused by a typo in DT.
            # These reasonable gaps are the not the cases that we would like to correct, therefore, they
            # are deleted from the list of gaps of interest.
            index_gap2drop.append(idx)
    if len(index_gap2drop) == len(index_long_gap):
        # Case 1: All gaps are reasonable gaps. But there might still be chunks of records before AT or after
        # the max DT of the monvals records, those chunks should be deleted.
        case = 1
        DISPLAY_BEFORE(index_long_gap, show_case, case, where='center')
        index_df2drop = []
        if tbl != 'monvals':
            if '1_hdf5_consent' in input_path:
                max_monvals_dt = pd.read_hdf(os.path.join(input_path, 'monvals.h5'), columns=['Datetime'], 
                                             where='PatientID=%d'%pid, mode='r').Datetime.max()
            else:
                max_monvals_dt = pd.read_hdf(output_path.replace(tbl, 'monvals'), columns=['Datetime'], 
                                             where='PatientID=%d'%pid, mode='r').Datetime.max()
        idx_not_reasonable = []
        for idx in index_long_gap:
            if diff_dt[idx] < 3:
                continue
            if df.iloc[idx].Datetime.date() < AdmTime:
                # If the chunk of records are observed before AT, it should be deleted
                if tbl!='labres':
                    index_df2drop.extend( df.index[:idx+1] )
                    idx_not_reasonable.append(idx)
            if tbl != 'monvals' and df.iloc[idx+1].Datetime.date() > max_monvals_dt:
                # If the chunk of records are observed after the max DT of monvals records, it should be deleted
                # as well
                index_df2drop.extend( df.index[idx+1:] )
                idx_not_reasonable.append(idx)
            if (len(df) - idx - 1) < 5 and (len(df) - idx - 1 ) / len(df) < 0.05:
                if tbl=='labres':
                    index_df2drop.extend( df.index[df.Datetime>(max_monvals_dt+sec_of_day)] )
                    idx_not_reasonable.append(idx)                
                else:
                    index_df2drop.extend( df.index[idx+1:] )
                    idx_not_reasonable.append(idx)                
                
        if len(index_df2drop)>0 and (len(index_df2drop) < 0.05*len(df) or diff_dt[idx]>3):
            # Delete records if there are records to delete
            index_df2drop = np.unique(index_df2drop)
            df.drop(index_df2drop, inplace=True)
            DISPLAY_AFTER(np.unique(idx_not_reasonable), show_case, case, where='center')


        return df, case
    
    if len(index_gap2drop) > 0:
        # If only some gaps are reasonable, delete thme from the list of gaps of interest and continue
        index_long_gap = np.sort(list(set(index_long_gap) - set(index_gap2drop)))

    if len(index_long_gap) > 1:
        # Case 2: When there is more than 1 gap of interest. We look at each chunk separated by the gaps, if the 
        # difference between DT and ET of each record within a chunk is smaller than 36 hours, we consider them
        # to be normal without typos (see case 2.1). When there is no day difference but month difference between 
        # DT and ET of each record within a chunk, then we assume there is a typo in the DT month of records which
        # will be corrected to the ET month of the corresponding record (see case 2.2). When the time span of ET
        # of records is smaller than 6 hours, but the time span of DT is larger than a day, it means that these 
        # records are history long before the admission and were entered within a short period; these records 
        # should also be deleted because there are no monvals observation in that historical period (see case 2.3).
        # When the current chunk is measured before AT, when the DT is after ET or when there is a month difference 
        # in the DT and ET, the chunk of records looks weird and not correctable, thus deletable (see case 2.4).
        # And for the remaining cases, they are usually reasonable chunks therefore no correction is needed. 
        case = 2
        DISPLAY_BEFORE(index_long_gap, show_case, case, where='center')
        index_long_gap = np.concatenate(([0], index_long_gap+1, [len(df)]))
        index_df2drop = []
        for i in range(len(index_long_gap)-1):
            tmp = df.iloc[index_long_gap[i]:index_long_gap[i+1]].copy()
            get_mo = lambda x: x.month
            get_dd = lambda x: x.day
            if diff_dt[index_long_gap[i]-1] > 3 and index_long_gap[i] / len(df) > 0.95:
                index_df2drop.extend(df.index[index_long_gap[i]:])
            elif np.abs(tmp.Datetime - tmp.Entertime).max() / np.timedelta64(1, 'h') <= 36:
                # case 2.1
                pass

            elif (np.sum(tmp.Datetime.apply(get_mo) - tmp.Entertime.apply(get_mo)==0) == 0 and 
                np.sum(np.abs(tmp.Datetime.apply(get_dd) - tmp.Entertime.apply(get_dd))>1) == 0):
                # case 2.2
                for k in range(len(tmp)):
                    new_yr = tmp.iloc[k].Entertime.year
                    new_mo = tmp.iloc[k].Entertime.month
                    df.loc[tmp.index[k],'Datetime'] = tmp.iloc[k].Datetime.replace(year=new_yr,
                                                                                   month=new_mo)
            elif ((tmp.iloc[-1].Entertime - tmp.iloc[0].Entertime) / sec_of_hour<=6 and 
                  (tmp.iloc[-1].Datetime - tmp.iloc[0].Datetime) / sec_of_hour > 24 and 
                  len(tmp)/len(df)<0.1 ):
                # case 2.3
                index_df2drop.extend(tmp.index)

            elif (tmp.Datetime.max() < AdmTime or 
                  np.sum((tmp.Datetime - tmp.Entertime)/sec_of_hour < 0) == 0 or
                  np.sum(tmp.Datetime.apply(get_mo) - tmp.Entertime.apply(get_mo)==0) == 0):
                # case 2.4
                if i > 0  and len(tmp) == 1:
                    pass
                else:
                    index_df2drop.extend(tmp.index)

        if tbl!='labres' and  len(index_df2drop) > 0:
            df.drop(index_df2drop, inplace=True)
            DISPLAY_AFTER(index_long_gap[1:-1], show_case, case, where='center')

        return df, case

    idx = index_long_gap[0]
    ### The remaining cases only have one gap of interest
    ### We first look at the record before the gap, then we look at the record after the gap
    dt_gap_mo = df.iloc[idx+1].Datetime.month - df.iloc[idx].Datetime.month
    dt_gap_dd = df.iloc[idx+1].Datetime.day - df.iloc[idx].Datetime.day

    be_dt_et_yr = df.iloc[idx].Datetime.year - df.iloc[idx].Entertime.year
    be_dt_et_mo = df.iloc[idx].Datetime.month - df.iloc[idx].Entertime.month
    be_dt_et_dd = df.iloc[idx].Datetime.day - df.iloc[idx].Entertime.day
    be_dt_et_hr = df.iloc[idx].Datetime.hour - df.iloc[idx].Entertime.hour

    af_dt_et_yr = df.iloc[idx+1].Datetime.year - df.iloc[idx+1].Entertime.year
    af_dt_et_mo = df.iloc[idx+1].Datetime.month - df.iloc[idx+1].Entertime.month
    af_dt_et_dd = df.iloc[idx+1].Datetime.day - df.iloc[idx+1].Entertime.day
    af_dt_et_hr = df.iloc[idx+1].Datetime.hour - df.iloc[idx+1].Entertime.hour
    
        
    if (((be_dt_et_yr == 0 and be_dt_et_mo != 0) or 
         (np.abs(be_dt_et_yr) == 1 and np.abs(be_dt_et_mo) == 11)) and 
        np.abs(be_dt_et_dd) < 2):
        # Case 3: When there is almost no difference in day between the DT and ET of the record before the gap,
        # but the month value is off, it means that the month of all records before the gap are wrong. So we 
        # correct the month of those records by subtracting the year and month difference of the record before
        # the gap from all records before the gap
        case = 3
        DISPLAY_BEFORE(idx, show_case, case, where='post')
        dt_gap_yr = df.iloc[idx+1].Datetime.year - df.iloc[idx].Datetime.year
        et_gap_yr = df.iloc[idx+1].Entertime.year - df.iloc[idx].Entertime.year
        

        dt_gap_mo = df.iloc[idx+1].Datetime.month - df.iloc[idx].Datetime.month
        et_gap_mo = df.iloc[idx+1].Entertime.month - df.iloc[idx].Entertime.month

        if (et_gap_yr != 0 and dt_gap_yr == 0) or (et_gap_mo != 0 and dt_gap_mo == 0) :
            dt_gap_dd = df.iloc[idx+1].Datetime.day - df.iloc[idx].Datetime.day
            dt_et_gap_dd = df.iloc[idx].Datetime.day - df.iloc[idx].Entertime.day

            dt_gap_hr = df.iloc[idx+1].Datetime.hour - df.iloc[idx].Datetime.hour
            dt_et_gap_hr = df.iloc[idx].Datetime.hour - df.iloc[idx].Entertime.hour
        
            if (dt_gap_dd != 0 and dt_et_gap_dd == 0) or (dt_gap_hr != 0 and dt_et_gap_hr == 0):
                tmp = df.iloc[:idx+1]
                if (tmp.Datetime.diff()/sec_of_hour).max() > 12:
                    df.drop(df.index[:idx+1], inplace=True)
                else:
                    for k in range(idx+1):
                        new_yr = df.iloc[k].Entertime.year
                        new_mo = df.iloc[k].Entertime.month
                        old_mo = df.iloc[k].Datetime.month
                        if df.iloc[k].Datetime.day==31 and df.iloc[k].Entertime.day==1:
                            df.loc[df.index[k], 'Datetime'] = df.loc[df.index[k], 'Datetime'].replace(year=new_yr,
                                                                                                      month=new_mo-1,
                                                                                                      day=28)
                        else:
                            df.loc[df.index[k], 'Datetime'] = df.loc[df.index[k], 'Datetime'].replace(year=new_yr,
                                                                                                      month=new_mo)
                        if np.abs(df.loc[df.index[k], 'Datetime']-df.loc[df.index[k], 'Entertime']) / sec_of_day > 1:
                            new_yr, new_mo, new_dd = date_correction(new_yr, new_mo-1, df.loc[df.index[k], 'Datetime'].day, old_mo)
                            df.loc[df.index[k], 'Datetime'] = df.loc[df.index[k], 'Datetime'].replace(month=new_mo,
                                                                                                      day=new_dd)
                        
            elif len(df) - idx - 1 < 5 and (len(df) - idx - 1) / len(df) < 0.05:
                df.drop(df.index[idx+1:], inplace=True)
                DISPLAY_AFTER(idx, show_case, case, where='post')
            elif idx < 5 and idx / len(df) <0.05:
                df.drop(df.index[:idx+1], inplace=True)
                DISPLAY_AFTER(idx, show_case, case, where='post')
        else:
            for k in range(idx+1):
                new_yr = df.iloc[k].Datetime.year - be_dt_et_yr
                new_mo = df.iloc[k].Datetime.month - be_dt_et_mo
                new_dd = df.iloc[k].Datetime.day
                old_mo = df.iloc[k].Datetime.month

                new_yr, new_mo, new_dd = date_correction(new_yr, new_mo, new_dd, old_mo)
                idx2fix = df.index[k]
                df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(year=new_yr, 
                                                                                month=new_mo, 
                                                                                day=new_dd)
            DISPLAY_AFTER(idx, show_case, case, where='post')

    elif (np.abs(be_dt_et_dd) >= 9 and 
          np.abs(be_dt_et_dd)%10 in [0,1,9] and
          np.abs(df.iloc[idx].Datetime - df.iloc[idx].Entertime) / sec_of_hour > 24 * 9):
        # Case 4: When the day between the DT and ET of the record before the gap is off by multiple of 10 days.
        case = 4
        DISPLAY_BEFORE(idx, show_case, case)
        if df.iloc[idx].Datetime.date() == AdmTime:
            pass
        # elif be_dt_et_mo == 0 and be_dt_et_hr == 0:
        #     tmp = df.iloc[:idx+1]
        #     if (tmp.Datetime.diff()/sec_of_hour).max()>12:
        #         df.drop(df.index[:idx+1], inplace=True)
        #     else:
        #         import ipdb
        #         ipdb.set_trace()
        #         for idx2fix in df.index[:idx+1]:
        #             new_dd = df.loc[idx2fix,'Entertime'].day
        #             df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(day=new_dd)
        else:
            new_diff_dd = -31 if be_dt_et_dd == -30 else int( np.round(be_dt_et_dd/10) * 10)
            if be_dt_et_mo==0:
                fix_func = lambda x: x-np.timedelta64(1,'D')*new_diff_dd
            else:
                fix_func = lambda x: x.replace(month=df.iloc[idx].Entertime.month)-np.timedelta64(1,'D')*new_diff_dd

            if idx/len(df) < 0.1:
                try:
                    df.loc[df.index[:idx+1],'Datetime'] = df.iloc[:idx+1].Datetime.apply(fix_func)
                except:
                    df.loc[df.index[:idx+1], 'Datetime'] = df.iloc[:idx+1].Datetime.apply(lambda x: x+sec_of_day*(30-new_diff_dd))

            if tbl != 'monvals':
                if '1_hdf5_consent' in input_path:
                    monvals_dt = pd.read_hdf(os.path.join(input_path, 'monvals.h5'), columns=['Datetime'], 
                                                 where='PatientID=%d'%pid, mode='r').Datetime
                else:
                    monvals_dt = pd.read_hdf(output_path.replace(tbl, 'monvals'), columns=['Datetime'], 
                                                 where='PatientID=%d'%pid, mode='r').Datetime
                if (monvals_dt.max()-df.Datetime.max()) / sec_of_day < -3:
                    df.drop(df.index[df.Datetime>monvals_dt.max()+sec_of_day], inplace=True)

            DISPLAY_AFTER(idx, show_case, case)
            
    elif (np.abs(be_dt_et_dd)==27 and 
          df.iloc[idx].Datetime.month == 2 and 
          df.iloc[idx].Entertime.month==2): 
        # Case 5: When the DT is supposed to be March 1st but misentered as Jan 1st when the ET is Feb 28th, which
        # means that the DT should be the next day of after the ET.
        case = 5
        DISPLAY_BEFORE(idx, show_case, case)

        fix_func = lambda x: x-np.timedelta64(1,'D')*(be_dt_et_dd-1)
        idx2fix = df.index[:idx+1]
        df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_func)
        
        DISPLAY_AFTER(idx, show_case, case)

        # When there is only one gap in the time series. 
    elif (((af_dt_et_yr == 0 and af_dt_et_mo != 0) or 
           (np.abs(af_dt_et_yr) == 1 and np.abs(af_dt_et_mo) == 11)) and 
          np.abs(af_dt_et_dd) < 2):
        # Case 6: When there is almost no day difference in DT and ET of the record after the gap, but the month
        # is off, so there might be a typo in the month. So correct the month of DT. 
        case = 6
        DISPLAY_BEFORE(idx, show_case, case, where='post')
        for k in np.arange(idx+1, len(df)):
            new_yr = df.iloc[k].Datetime.year - af_dt_et_yr
            new_mo = df.iloc[k].Datetime.month - af_dt_et_mo
            new_dd = df.iloc[k].Datetime.day
            old_mo = df.iloc[k].Datetime.month
            
            new_yr, new_mo, new_dd = date_correction(new_yr, new_mo, new_dd, old_mo)
            idx2fix = df.index[k]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(year=new_yr, 
                                                                            month=new_mo, 
                                                                            day=new_dd)
            
        DISPLAY_AFTER(idx, show_case, case, where='post')

    elif (np.abs(af_dt_et_dd)>=9 and 
          np.abs(af_dt_et_dd)%10 in [0,1,9] and
          np.abs(df.iloc[idx+1].Datetime - df.iloc[idx+1].Entertime) / sec_of_hour > 24 * 9):
        # Case 7: When the day between the DT and ET of the record after the gap is off by multiple of 10 days.
        case = 7
        DISPLAY_BEFORE(idx, show_case, case, where='center')
        if af_dt_et_dd%10 in [0,9] and af_dt_et_hr > 20:
            for idx2fix in df.index[idx+1:]:
                if df.loc[idx2fix,'Datetime'].day == df.loc[idx2fix,'Entertime'].day:
                    new_mo = df.loc[idx2fix,'Entertime'].month
                    df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(month=new_mo)
                else:
                    new_dd = df.loc[idx2fix,'Entertime'].day - 1
                    if new_dd == 0:
                        new_mo = df.loc[idx2fix,'Entertime'].month - 1
                        if new_mo == 0:
                            new_mo = 12
                            new_yr = df.loc[idx2fix,'Entertime'].year - 1
                            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(year=new_yr, 
                                                                                            month=new_mo)
                        else:
                            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(month=new_mo)

                    else:
                        df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(day=new_dd)
        elif af_dt_et_dd%10 == 1 and af_dt_et_hr < -20:
            for idx2fix in df.index[idx+1:]:
                new_dd = df.loc[idx2fix,'Entertime'].day + 1
                df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(day=new_dd)

        elif len(df) - idx - 1 < 5 and (len(df) - idx - 1) / len(df) < 0.05:
            df.drop(df.index[idx+1:], inplace=True)
            DISPLAY_AFTER(idx, show_case, case, where='center')
        elif idx/len(df)<0.05:
            df.drop(df.index[:idx+1], inplace=True)
        else:
            new_diff_dd = -31 if af_dt_et_dd == -30 else int( np.round(af_dt_et_dd/10) * 10)
            fix_func = lambda x: x-np.timedelta64(1,'D')*new_diff_dd
            df.loc[df.index[idx+1:],'Datetime'] = df.iloc[idx+1:].Datetime.apply(fix_func)

            is_reduce_day = np.logical_and(df.iloc[idx+1:].Datetime > df.iloc[idx+1:].Entertime, 
                                           df.iloc[idx+1:].Datetime.apply(lambda x: x.hour)==23)
            index_reduce_day = df.index[idx+1:][is_reduce_day]
            if len(index_reduce_day) > 0:
                fix_func = lambda x: x-np.timedelta64(1,'D')
                df.loc[index_reduce_day,'Datetime'] = df.loc[index_reduce_day,'Datetime'].apply(fix_func)
                DISPLAY_AFTER(idx, show_case, case, where='center')

    elif (np.abs(af_dt_et_dd)==27 and 
          df.iloc[idx+1].Datetime.month == 2 and 
          df.iloc[idx+1].Entertime.month==2): 
        # Case 8: when the DT of the record after the gap is supposed to be March 1st but is written as Feb 1st.
        case = 8
        DISPLAY_BEFORE(idx, show_case, case, where='post')
            
        fix_func = lambda x: x-np.timedelta64(1,'D')*(af_dt_et_dd-1)
        idx2fix = df.index[idx+1:]
        df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_func)
        
        DISPLAY_AFTER(idx, show_case, case, where='post')

    elif (df.iloc[0].Datetime >= AdmTime and
          tbl == 'pharmarec'):
        # Case 9: When the DT of the record is later than AT and the table is pharmarec, if the status is 780, 
        # just correct the day of the DT to the day of ET; but if the status is not 780, then compute the rate 
        # of the drug and if the rate makes the std of rate smaller than no change, otherwise replace the last 
        # rate with the median rate and recompute the last time point.
        case = 9
        if af_dt_et_mo==0 and af_dt_et_hr==0:
            for idx2fix in df.index[idx+1:]:
                new_dd =  df.loc[idx2fix,'Entertime'].day
                df.loc[idx2fix,'Datetime'] =  df.loc[idx2fix,'Datetime'].replace(day=new_dd)
        elif diff_dt[idx] > 3:
            df.drop(df.index[idx+1:], inplace=True)
        DISPLAY_BEFORE(idx, show_case, case, where='center')
        pass
    
    elif df.iloc[0].Datetime >= AdmTime:
        # Case 10: if the DT of the first record is later than AT and the table is not pharmarec, then if 
        # the DT of the last record with the DT of the last record in monvals, then delete all records after the 
        # gap.
        case = 10
        DISPLAY_BEFORE(idx, show_case, case, where='center')
        if tbl == 'monvals':
            if (idx+1) / len(df) > 0.95:
                df.drop(df.index[idx+1:], inplace=True)
                DISPLAY_AFTER(idx, show_case, case, where='center')
        else:
            if '1_hdf5_consent' in input_path:
                monvals_dt = pd.read_hdf(os.path.join(input_path, 'monvals.h5'), columns=['Datetime'], 
                                         where='PatientID=%d'%pid, mode='r').Datetime
            else:
                monvals_dt = pd.read_hdf(output_path.replace(tbl, 'monvals'), columns=['Datetime'], 
                                         where='PatientID=%d'%pid, mode='r').Datetime
            if len(monvals_dt) > 0:
                diff_monvals_dt = np.diff(np.sort(monvals_dt)) / np.timedelta64(1, 'h')
                if (np.argmax(diff_monvals_dt)+1) / len(diff_monvals_dt) > 0.95 and (idx+1) / len(df) > 0.95:
                    df.drop(df.index[idx+1:], inplace=True)
                    DISPLAY_AFTER(idx, show_case, case, where='center')
                elif (df.loc[df.index[idx+1], 'Datetime'].day == gd.loc[pid,'AdmissionTime'].day and 
                      np.abs(df.loc[df.index[idx+1], 'Datetime'].month - gd.loc[pid,'AdmissionTime'].month)==1):
                    new_mo = gd.loc[pid,'AdmissionTime'].month
                    for idx2fix in df.index[idx+1:]:
                        df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(month=new_mo)                    
                else:
                    max_monvals_dt = monvals_dt.max()
                    if (df.iloc[idx+1].Datetime - max_monvals_dt) / sec_of_day > 3:
                        df.drop(df.index[idx+1:], inplace=True)
                        DISPLAY_AFTER(idx, show_case, case, where='center')

    elif df.iloc[idx].Datetime < AdmTime:
        # Case 11: When the DT before the gap is earlier than AT. If the percentage of records is less than 10% of
        # the time series, then delete all records before the gap, otherwise if the DT before the gap is earlier
        # than the min DT from monvals or the DT after the gap is later than max DT from monvals, then delete the 
        # records before or after the gap.
        case = 11
        DISPLAY_BEFORE(idx, show_case, case, where='center')
        if idx / len(df) < 0.1:
            if be_dt_et_yr==0 and be_dt_et_mo==0 and be_dt_et_dd==-1 and np.abs(be_dt_et_hr)<3:
                for idx2fix in df.index[:idx+1]:
                    new_dd = df.loc[idx2fix,'Entertime'].day
                    df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(day=new_dd)
            elif idx == 0 or diff_dt[idx] > 3:
                df.drop(df.index[:idx+1], inplace=True)
            else:
                pass
            diff_dt_tmp = df.Datetime.diff().values[1:] / sec_of_day        
            if diff_dt_tmp.max() > 1:
                idx = np.argmax(diff_dt_tmp)
                if idx/len(df) > 0.95:
                    df.drop(df.index[idx+1:], inplace=True)
            # DISPLAY_AFTER(idx, show_case, case, where='center')
        elif tbl!='monvals':
            if '1_hdf5_consent' in input_path:
                monvals_dt = pd.read_hdf(os.path.join(input_path, 'monvals.h5'), columns=['Datetime'], 
                                             where='PatientID=%d'%pid, mode='r').Datetime
            else:
                monvals_dt = pd.read_hdf(output_path.replace(tbl, 'monvals'), columns=['Datetime'], 
                                             where='PatientID=%d'%pid, mode='r').Datetime
            min_monvals_dt = monvals_dt.min().date()
            max_monvals_dt = monvals_dt.max().date()
            index_df2drop = []
            
            if df.iloc[idx].Datetime.date() < min_monvals_dt:
                if tbl!='labres':
                    index_df2drop.extend(df.index[:idx+1])
            if df.iloc[idx+1].Datetime.date() > max_monvals_dt:
                index_df2drop.extend(df.index[idx+1:])
            if len(index_df2drop) > 0:
                df.drop(index_df2drop, inplace=True)
                DISPLAY_AFTER(idx, show_case, case, where='center')

    else:
        # Case 12: The rest of the cases. If the DT of the record after the gap is later than the max DT 
        # of records in monvals then delete all records after the gap.
        case = 12
        DISPLAY_BEFORE(idx, show_case, case, where='post')
        
        if tbl!='monvals':
            if '1_hdf5_consent' in input_path:
                max_monvals_dt = pd.read_hdf(os.path.join(input_path, 'monvals.h5'), columns=['Datetime'], 
                                             where='PatientID=%d'%pid, mode='r').Datetime.max().date()
            else:
                max_monvals_dt = pd.read_hdf(output_path.replace(tbl, 'monvals'), columns=['Datetime'], 
                                             where='PatientID=%d'%pid, mode='r').Datetime.max().date()
            if (np.abs(df.iloc[idx+1].Datetime.month-gd.loc[pid,'AdmissionTime'].month)==1 and 
                df.iloc[idx+1].Datetime.day==gd.loc[pid,'AdmissionTime'].day):
                new_mo = gd.loc[pid,'AdmissionTime'].month
                for idx2fix in df.index[idx+1:]:
                    df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(month=new_mo)
            elif df.iloc[idx+1].Datetime.date() > max_monvals_dt:
                df.drop(df.index[idx+1:], inplace=True)
                    
        DISPLAY_AFTER(idx, show_case, case, where='post')

    return df, case


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-tbl')
    parser.add_argument('-input_path')
    parser.add_argument('-output_path')
    parser.add_argument('-chunking_info_file')
    parser.add_argument('-batch_id', type=int)
    parser.add_argument('--id_path', default=None)
    parser.add_argument('--restart_pid', default=66204, type=int)
    parser.add_argument('--write_to_disk', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    tbl = args.tbl
    chunking_info_file = args.chunking_info_file
    input_path = args.input_path
    output_path = args.output_path
    batch_id = args.batch_id
    id_path = args.id_path
    write_to_disk = args.write_to_disk
    debug = args.debug

    if tbl in ['monvals', 'comprvals', 'dervals']: 
        fields = ['PatientID', 'VariableID', 'Datetime', 
                  'Entertime', 'Value', 'Status']

    elif tbl == 'labres':
        fields = ['PatientID', 'VariableID', 'ResultID', 'Datetime', 
                  'Entertime', 'Value', 'Status', ]

    elif tbl == 'observrec':
        fields = ['PatientID', 'Datetime', 'Entertime', 'VariableID', 
                  'Value', 'Status']

    elif tbl == 'pharmarec':
        fields = ['PatientID', 'PharmaID', 'InfusionID', 'Route', 
                  'Datetime', 'Entertime', 'CumulDose', 'GivenDose', 
                  'Rate', 'Status']

    if not write_to_disk:
        print('TEST MODE: WILL NOT WRITE THE RESULTS TO DISK.')

    chunking_info = pd.read_csv(chunking_info_file)
    chunking_info.rename(columns={'ChunkfileIndex': 'BatchID'}, inplace=True)
    if 'PatientID' in chunking_info.columns:
        pid_list = chunking_info.PatientID[chunking_info.BatchID==batch_id].values
    else:
        pid_list = chunking_info.index[chunking_info.BatchID==batch_id].values

    if debug:
        pid_list = np.array([int(x) for x in open('pids2fix_%s.csv'%tbl, 'r').readlines()])
        if args.restart_pid is not None:
            pid_list = pid_list[np.where(pid_list==args.restart_pid)[0][0]:]

    if '1_hdf5_consent' in input_path:
        gd = pd.read_hdf(os.path.join(input_path, 'generaldata.h5'), mode='r')
        gc.collect()
    else:
        tbl_path = os.path.join(input_path, 'p_generaldata')
        gd_file = [f for f in os.listdir(tbl_path) if 'crc' not in f and 'parquet' in f][0]
        gd = pd.read_parquet(os.path.join(tbl_path, gd_file), engine='pyarrow')
        gd.loc[:,'admissiontime'] = pd.to_datetime(gd.admissiontime).dt.floor(freq='s')
        gd.rename(columns={'admissiontime': 'AdmissionTime',
                           'patientid': 'PatientID'}, inplace=True)
        if id_path is None:
            raise Exception('Please provide the path to the pid-file mapping files.')
        else:
            pid_file_mapping = pd.read_hdf(os.path.join(id_path, 'PID_File_Mapping_%s.h5'%tbl))
            pid_file_mapping.set_index('PatientID', inplace=True)

    gd.set_index('PatientID', inplace=True)
    gd = gd[gd.index.isin(pid_list)]

    # Data paths
    if debug:
        debug_output_path = copy(output_path)
    output_path = os.path.join(output_path, tbl)
    if write_to_disk:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    output_path = os.path.join(output_path, 
                               '%s_%d_%d--%d.h5'%(tbl, 
                                                  batch_id, 
                                                  np.min(pid_list), 
                                                  np.max(pid_list)))

    sec_of_day = np.timedelta64(24*3600, 's')
    new_idx = 0
    df_batch = None
    for nn, pid in enumerate(pid_list):
        if debug:
            batch_id = chunking_info.loc[chunking_info.index[chunking_info.PatientID==pid][0],'BatchID']
            if 'PatientID' in chunking_info.columns:
                debug_pid_list = chunking_info.PatientID[chunking_info.BatchID==batch_id].values
            else:
                debug_pid_list = chunking_info.index[chunking_info.BatchID==batch_id].values
            output_path = os.path.join(debug_output_path, tbl)
            if write_to_disk:
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
            output_path = os.path.join(output_path, 
                                       '%s_%d_%d--%d.h5'%(tbl, 
                                                          batch_id, 
                                                          np.min(debug_pid_list), 
                                                          np.max(debug_pid_list)))
            debug_monvals = pd.read_hdf(output_path.replace(tbl, 'monvals'), columns=['Datetime'], 
                                        where='PatientID=%d'%pid, mode='r').Datetime
        if '1_hdf5_consent' in input_path:
            df = pd.read_hdf(os.path.join(input_path, '%s.h5'%tbl), 
                             where='PatientID=%d'%pid, mode='r') 
        else:
            if df_batch is None or pid not in df_batch.PatientID.unique():
                try:
                    df_batch = pd.read_parquet(pid_file_mapping.loc[pid].FileName.replace('_preview',''), 
                                               engine='pyarrow') 
                except:
                    continue
                df_batch.rename(columns={'datetime': 'Datetime', 'entertime': 'Entertime', 
                                         'sampletime': 'Datetime', 'patientid': 'PatientID',  
                                         'variableid': 'VariableID', 'valueid': 'ValueID',
                                         'pharmaid': 'PharmaID', 'infusionid': 'InfusionID', 
                                         'resultid': 'ResultID', 'cumuldose': 'CumulDose', 
                                         'rate': 'Rate', 'route': 'Route', 'value': 'Value',  
                                         'givendose': 'GivenDose', 'status': 'Status'}, inplace=True)
                df_batch.drop([col for col in df_batch.columns if col not in fields], 
                        axis=1, inplace=True)
                gc.collect()
                for col in ['Datetime', 'Entertime']:
                    df_batch.loc[:,col] = df_batch[col].dt.floor(freq='s')
                df = df_batch[df_batch.PatientID==pid].copy()
            else:
                df = df_batch[df_batch.PatientID==pid].copy()

        if len(df) == 0:
            print('Patient', pid, 'has no data.')
            continue

        # Rename some columns so that the columns names are consistent across all tables
        if tbl == 'pharmarec':
            df.rename(columns={'PharmaID': 'VariableID', 
                               'GivenDose': 'Value'}, inplace=True)

        # Remove duplicates with the exact same values and sort by datetime
        key_cols = ['Datetime', 'PatientID', 'VariableID', 'Value', 'Status']
        if tbl == 'pharmarec':
            key_cols.extend(['InfusionID', 'Route', 'CumulDose', 'Rate'])
        elif tbl == 'labres':
            key_cols.extend(['ResultID'])
        df.drop_duplicates(key_cols, inplace=True)
        df.sort_values(['Datetime', 'Entertime'], inplace=True)
        gc.collect()


        # The time difference between adjecent records in days
        diff_dt = df.Datetime.diff().values[1:] / sec_of_day        

        if len(diff_dt) == 0:
            pass
        else:
            # If there exists gap longer than 31 days
            try:
                assert(diff_dt.max() <= 31)
            except AssertionError:
                old_max_gap = diff_dt.max()
                old_df_size = len(df)
                if debug:
                    print('Patient', pid, 'longest gap before fixing:', diff_dt.max(), 'days.')
                    print(gd.loc[pid].AdmissionTime)
                    print(len(df))
                    print(df.iloc[max(0, np.where(diff_dt>31)[0][0]-10):min(np.where(diff_dt>31)[0][0]+10, len(df))] )
                    import ipdb
                    ipdb.set_trace()
                df, case = fix_df_with_very_long_gap()
                df.sort_values(['Datetime', 'Entertime'], inplace=True)
                if len(df) <= 1:
                    print('small', old_df_size, len(df), pid)
                else:
                    diff_dt = df.Datetime.diff().values[1:] / sec_of_day
                    new_max_gap = diff_dt.max()
                    new_df_size = len(df)
                    if (old_df_size-new_df_size)/float(old_df_size)>0.2 or new_max_gap > 31:
                        print('Patient', pid, 'longest gap after fixing:', new_max_gap, 'days. delete %2.2f%%'%((old_df_size - new_df_size)/float(old_df_size)*100))
                    if debug:
                        print(len(df))
                        print(df.iloc[np.max(np.argmax(diff_dt)-5, 0):np.argmax(diff_dt)+5])
                        print('Patient', pid, 'longest gap after fixing:', new_max_gap, 'days.')
                        ipdb.set_trace()

            try:
                assert(diff_dt.max() <= 1)
            except AssertionError:
                old_max_gap = diff_dt.max()
                old_df_size = len(df)
                if debug:
                    print('Patient', pid, 'longest gap before fixing:', diff_dt.max(), 'days.')
                    print(gd.loc[pid].AdmissionTime)
                    print(len(df))
                    print(df.iloc[max(0, np.where(diff_dt>1)[0][0]-10):min(np.where(diff_dt>1)[0][0]+10, len(df))] )
                    import ipdb
                    ipdb.set_trace()
                df, case = fix_df_with_long_gap()
                df.sort_values(['Datetime', 'Entertime'], inplace=True)
                if len(df) <= 1:
                    print('small', old_df_size, len(df), pid)
                else:
                    diff_dt = df.Datetime.diff().values[1:] / sec_of_day
                    new_max_gap = diff_dt.max()
                    new_df_size = len(df)
                    if  (old_df_size-new_df_size)/float(old_df_size)>0.2 or new_max_gap > 3:
                        print('Patient', pid, 'longest gap after fixing:', new_max_gap, 'days. delete %2.2f%%'%((old_df_size - new_df_size)/float(old_df_size)*100))
                    if debug:
                        print(len(df))
                        print(df.iloc[np.max(np.argmax(diff_dt)-5, 0):np.argmax(diff_dt)+5])
                        print('Patient', pid, 'longest gap after fixing:', new_max_gap, 'days.')
                        ipdb.set_trace()

            # try:
            #     assert(diff_dt.max() <= 1)
            # except AssertionError:
            #     print('Patient', pid, 'longest gap after fixing:', diff_dt.max(), 'days (', diff_dt.max()*24, 'hours).')
            

        # Reverse the renaming for observrec table
        if tbl == 'pharmarec':
            df.rename(columns={'Value': 'GivenDose', 
                               'VariableID': 'PharmaID'}, inplace=True)

        df.set_index(np.arange(new_idx, new_idx+len(df)), drop=True, inplace=True)
        new_idx += len(df)
        if (df.VariableID==15001546).sum() > 0:
            idx_v15001546 = df.index[df.VariableID==15001546]
            dict_fio2 = {1: 60, 2:100}
            df.loc[idx_v15001546, 'Value'] = df.loc[idx_v15001546, 'Value'].apply(lambda x: dict_fio2[x] if x in [1,2] else x)
        if write_to_disk:
            df.to_hdf(output_path, 'data', append=True, data_columns=True, 
                      complevel=5, complib='blosc:lz4', format='table')
            if tbl=='monvals':
                output_path_svo2 = output_path.replace('monvals', 'monvals_svo2')
                if not os.path.exists(os.path.split(output_path_svo2)[0]):
                    os.mkdir(os.path.split(output_path_svo2)[0])
                df_svo2 = df[df.VariableID==4200]
                if len(df_svo2) > 0:
                    df_svo2.to_hdf(output_path_svo2, 'data', append=True, 
                                   data_columns=True, format='table',
                                   complevel=5, complib='blosc:lz4')

        sys.stdout.write('# patients processed: %3d / %3d\r'%(nn+1, len(pid_list)))
        sys.stdout.flush()
        gc.collect()
print()
