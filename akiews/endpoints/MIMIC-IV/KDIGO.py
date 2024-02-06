import pandas as pd
import numpy as np
import datetime as dt
import math as m

from paths import *


def Baseline(eGFR, age, gender, ethnicity=0):
    '''
    eGFR: 75
    age: of patient
    gender: 1 Female, 0 other
    ethnicity: 1 African American, 0 other
    return baseline serum creatinine (1.0-1.6 mg/dL)
    '''

    if gender == 'F':
        gender = 1
    else:
        gender = 0
    if ethnicity == 'BLACK/AFRICAN AMERICAN':
        ethnicity = 1
    else:
        ethnicity = 0

    if type(age) == str:
        if len(age) <= 2:
            age = int(age)
        elif age[0] == '>':
            age = 90
        else:
            print('Unkown age', age)
            age = 90

    # Table 9 from KDIGO guidelines (used by all studies using RIFLE)
    if age < 25:
        if gender and ethnicity:
            return 1.2
        elif gender:
            return 1.0
        elif ethnicity:
            return 1.5
        else:
            return 1.3
    elif age < 30:
        if gender and ethnicity:
            return 1.1
        elif gender:
            return 1.0
        elif ethnicity:
            return 1.5
        else:
            return 1.2
    elif age < 40:
        if gender and ethnicity:
            return 1.1
        elif gender:
            return 0.9
        elif ethnicity:
            return 1.4
        else:
            return 1.2
    elif age < 55:
        if gender and ethnicity:
            return 1.0
        elif gender:
            return 0.9
        elif ethnicity:
            return 1.3
        else:
            return 1.1
    elif age < 66:
        if gender and ethnicity:
            return 1.0
        elif gender:
            return 0.8
        elif ethnicity:
            return 1.3
        else:
            return 1.1
    else:
        if gender and ethnicity:
            return 0.9
        elif gender:
            return 0.8
        elif ethnicity:
            return 1.2
        else:
            return 1.0

    # Original formula from 2009
    # creatinine_b = m.pow(eGFR / (175 * (int(age)**(-0.203)) * (0.742**(gender)) * 1.21**(ethnicity)), -1 / 1.154)
    # if creatinine_b > 1.6:
    #     creatinine_b = 1.6
    # if creatinine_b < 1.0:
    #     creatinine_b = 1.0

    # return creatinine_b


def first_valid_value(x):
    if x.first_valid_index() is None:
        return None
    else:
        return x[x.first_valid_index()]


def get_creatinine(cr_p, ethn, ag, gen, c_base='min'):
    try:
        creatinine_b = Baseline(75, ag, gen, ethn)
    except:
        creatinine_b = 1
    creatinine = cr_p.dropna(how='any')
    creatinine = creatinine.sort_values(by=[time_key])

    creatinine[creatinine_key] = creatinine[creatinine_key].apply(pd.to_numeric, errors='coerce')
    creatinine[creatinine_key] = [i / 88.4 if i > 5 else i for i in creatinine[creatinine_key].values]  # convert mmol/L to mg/dl
    creatinine = creatinine.dropna(how='any')

    if 'min' in c_base:
        creatinine_b = np.nanmin([creatinine_b, creatinine[creatinine_key].min()])
    elif 'first' in c_base:
        if len(cr_p) > 0:
            creatinine_b = first_valid_value(creatinine[creatinine_key])

    creatinine = creatinine.reset_index(drop=True).set_index(time_key)
    creatinine = creatinine.resample(gridsize_str).mean().interpolate(method='linear', limit=int(ffill_horizon / pd.Timedelta(gridsize_str)))
    creatinine = creatinine.reset_index()

    return creatinine, creatinine_b


def get_median_from_reduced(urine, window_start, window_stop, key):
    # print('Median', window_start, window_stop)
    df_single = urine[urine['measurement'] >= 1]
    # print(df_single)
    subs_df = df_single[np.logical_and(window_start < pd.to_datetime(df_single[time_key].values), pd.to_datetime(df_single[time_key].values) <= window_stop)][key]
    # print(subs_df)
    if len(subs_df) > 0:
        return subs_df.mean()
    return np.nan


def get_urine(ur_p, normalize=True, fill=True, halflife=0):
    urine = ur_p[[urine_key, time_key, weight_key]]
    urine[urine_key] = urine[urine_key].replace({0: np.nan})
    urine = urine.dropna(how='any')
    urine = urine.sort_values(by=[time_key])
    urine['measurement'] = 1


    # print(weight)
    if normalize and len(urine) > 0:
        urine[weight_key] = urine[weight_key].fillna(method='bfill')
        urine = urine.dropna(subset=[urine_key])
        urine[urine_key] = urine[urine_key] / urine[weight_key]
        urine = urine.drop(columns=[weight_key])
    urine = urine.sort_values(by=[time_key])

    urine[urine_key] = urine[urine_key].apply(pd.to_numeric, errors='coerce')
    # print(ur_time)
    if fill and len(urine) > 0:
        urine = urine.reset_index(drop=True).set_index(time_key)
        urine = urine.resample(gridsize_str).sum()
        urine[urine_key] = urine[urine_key].replace({0: np.nan})
        urine[urine_key] = urine[urine_key].fillna(method='bfill')
        if int(halflife) > 0:  # suggested by http://ceur-ws.org/Vol-1213/paper7.pdf
            # print(urine)
            urine[urine_key] = urine[urine_key].ewm(halflife=int(halflife)).mean()
            # print(urine)
        elif int(halflife) < 0:
            urine[urine_key] = urine[urine_key].rolling(window=abs(int(halflife)), win_type='gaussian', center=True).mean()
        urine = urine.reset_index()
    urine = urine.dropna(how='any')

    return urine


def get_u_mean_from_dataframe(urine, window_start, window_stop):
    if window_stop <= urine[time_key].max():
        subs_urine = urine[(urine[time_key] > window_start) & (urine[time_key] <= window_stop)][urine_key]
        if not subs_urine.isnull().values.any():
            if len(subs_urine)>0:
                return subs_urine.mean(skipna=False)
    return np.nan


def get_last_value_from_dataframe(df, time, dt, key):
    subs_cr = df[np.logical_and(time - dt < pd.to_datetime(df[time_key].values), pd.to_datetime(df[time_key].values) <= time)][key]
    if len(subs_cr) > 0:
        return subs_cr.min()
    return np.nan


def get_dialysis(rrt_p):
    if len(rrt_p) > 0:
        return rrt_p[time_key].min()
    else:
        return pd.NaT


def KDIGO(pid, cr_p, ur_p, ethn, ag, gen, in_dt, out_dt, rrt_p, halflife=0, c_base='min_i', gap=6):

    urine = get_urine(ur_p, normalize=True, fill=False, halflife=halflife)
    # print(urine)
    creatinine, creatinine_b = get_creatinine(cr_p, ethn, ag, gen, c_base=c_base)
    dialysis = get_dialysis(rrt_p)
    Events = pd.date_range(in_dt.floor('H'), out_dt.floor('H'), freq=gridsize_str)
    # print(Events)

    min_gap = gap * 60 * reference_time

    Events = pd.DataFrame(index=Events, columns=ep_types)
    Events[patient_key] = pid

    # print(Events)
    for time, row in Events.iterrows():
        u_mean_24 = get_u_mean_from_dataframe(urine, time, time + 24 * 60 * reference_time)
        u_mean_12 = get_u_mean_from_dataframe(urine, time, time + 12 * 60 * reference_time)
        u_mean_6 = get_u_mean_from_dataframe(urine, time, time + 6 * 60 * reference_time)
        u_median_24 = get_median_from_reduced(urine, time, time + 24 * 60 * reference_time, urine_key)
        u_median_12 = get_median_from_reduced(urine, time, time + 12 * 60 * reference_time, urine_key)
        u_median_6 = get_median_from_reduced(urine, time, time + 6 * 60 * reference_time, urine_key)
        next_c_value = get_last_value_from_dataframe(creatinine, time, gridsize, creatinine_key)
        creatinine_48_value = get_last_value_from_dataframe(creatinine, time, ffill_horizon, creatinine_key)

        if time > dialysis:
            Events.loc[time, '3.d'] = 1
        else:
            Events.loc[time, '3.d'] = 0

        if u_mean_12 == 0.:
            Events.loc[time:time + 12 * 60 * reference_time, '3.au'] = 1
        elif not np.isnan(u_mean_12) and not Events.loc[time, '3.au'] == 1:
            Events.loc[time:time + 12 * 60 * reference_time, '3.au'] = 0

        if u_mean_24 < 0.3 and u_median_24 < 0.3:
            Events.loc[time:time + 24 * 60 * reference_time, '3.r.u'] = 1
            Events.loc[time:time + 24 * 60 * reference_time, '3.u'] = 1
        elif not np.isnan(u_mean_24) and not Events.loc[time, '3.r.u'] == 1:
            Events.loc[time:time + 24 * 60 * reference_time, '3.r.u'] = 0
            Events.loc[time:time + 24 * 60 * reference_time, '3.u'] = 0

        if u_mean_12 < 0.5 and u_median_12 < 0.5:
            Events.loc[time:time + 12 * 60 * reference_time, '2.r.u'] = 1
            Events.loc[time:time + 12 * 60 * reference_time, '2.u'] = 1
        elif not np.isnan(u_mean_12) and not Events.loc[time, '2.r.u'] == 1:
            Events.loc[time:time + 12 * 60 * reference_time, '2.r.u'] = 0
            Events.loc[time:time + 12 * 60 * reference_time, '2.u'] = 0

        if u_mean_6 < 0.5 and u_median_6 < 0.5:
            Events.loc[time:time + 6 * 60 * reference_time, '1.r.u'] = 1
            Events.loc[time:time + 6 * 60 * reference_time, '1.u'] = 1
        elif not np.isnan(u_mean_6) and not Events.loc[time, '1.r.u'] == 1:
            Events.loc[time:time + 6 * 60 * reference_time, '1.r.u'] = 0
            Events.loc[time:time + 6 * 60 * reference_time, '1.u'] = 0

        if next_c_value - creatinine_48_value >= 4:
            Events.loc[time, '3.i'] = 1
        elif not np.isnan(next_c_value) or '_i' in c_base:
            Events.loc[time, '3.i'] = 0

        if next_c_value / creatinine_b > 3:
            Events.loc[time, '3.b'] = 1
        elif not np.isnan(next_c_value) or '_i' in c_base:
            Events.loc[time, '3.b'] = 0

        if next_c_value / creatinine_b > 2.0:
            Events.loc[time, '2.b'] = 1
        elif not np.isnan(next_c_value) or '_i' in c_base:
            Events.loc[time, '2.b'] = 0

        if next_c_value / creatinine_b > 1.5:
            Events.loc[time, '1.b'] = 1
        elif not np.isnan(next_c_value) or '_i' in c_base:
            Events.loc[time, '1.b'] = 0

        if next_c_value - creatinine_48_value >= 0.3:
            Events.loc[time, '1.i'] = 1
        elif not np.isnan(next_c_value) or '_i' in c_base:
            Events.loc[time, '1.i'] = 0

    # print(Events)

    # print('Postprocessing')
    # postprocessing urine events
    # first: merge consecutive regions
    for column_urine in ['3.u', '2.u', '1.u']:
        prev_ep_time = pd.NaT
        start_region = False
        start_time = pd.NaT
        Events_with_gaps = Events.copy()
        for index, row in Events_with_gaps.iterrows():
            if row[column_urine] != 1:
                start_region = True
            if row[column_urine] == 1:
                if start_region:
                    if index - prev_ep_time < min_gap:
                        print(column_urine,'Merged')
                        Events.loc[prev_ep_time:index, column_urine] = 1
                        print(Events)
                else:
                    prev_ep_time = index
    # print(Events)
    # second: tructuate ep regions by threhold
    for column_urine, threshold in zip(['3.u', '2.u', '1.u'], [0.3, 0.5, 0.5]):
        for k, v in Events.groupby((Events[column_urine].shift() != Events[column_urine]).cumsum()):
            if v[column_urine].values[0] == 1:
                # print(v)
                for idx, row in v.iterrows():
                    val = get_last_value_from_dataframe(urine, idx, gridsize, urine_key)
                    # print(val)
                    if val >= threshold:
                        break
                    elif not pd.isnull(val):
                        # print(column_urine, " Inserted zero in beginning")
                        Events.loc[idx, column_urine] = 0

                for idx, row in v[::-1].iterrows():
                    val = get_last_value_from_dataframe(urine, idx, gridsize, urine_key)
                    #print(val)
                    if val >= threshold:
                        break
                    elif not pd.isnull(val):
                        # print("Inserted zero at end")
                        Events.loc[idx, column_urine] = 0
    # if np.any(Events['1.u'].values>0):
    #     print(Events)
    return Events
