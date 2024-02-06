import pandas as pd
import numpy as np
import datetime as dt
import math as m

from imputation import imputed_weight, imputed_age
from paths import *


def creatinine_before_ICU(pid, time, dt):
    last_measurements = creatinine_tests_pre_ICU[creatinine_tests_pre_ICU['study_id'] == pid].dropna()
    last_measurements = last_measurements[(last_measurements['lab_date'] < time) & (last_measurements['lab_date'] > time - dt)]
    last_measurements['lab_nval'] = [i / 88.4 if i > 5 else i for i in last_measurements['lab_nval'].values]  # convert mmol/L to mg/dl
    return last_measurements


def Baseline(p_dfv, pid, eGFR, age, gender,weight, ethnicity=0):
    '''
    eGFR: 75
    age: of patient
    gender: 1 Female, 0 other
    ethnicity: 1 African American, 0 other
    return baseline serum creatinine (1.0-1.6 mg/dL)
    '''
    last_measurements = creatinine_before_ICU(pid, p_dfv[time_hirid_key].min(), pd.Timedelta('5 days'))
    if len(last_measurements) > 0:
        creatinine_b = last_measurements['lab_nval'].mean()
        return creatinine_b
    else:
        print('Not enough last measurements for ', pid)
        if not age:
            age = imputed_age(gender, ethnicity)

        if gender == 'F':
            gender = 1
        else:
            gender = 0


        if not weight:
            weight = imputed_weight(age, gender)

        if weight > 80:
            reduction = 0.2
        else:
            reduction = 0

        if ethnicity == 'African American':
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
                return 1.2-reduction
            elif gender:
                return 1.0-reduction
            elif ethnicity:
                return 1.5-reduction
            else:
                return 1.3-reduction
        elif age < 30:
            if gender and ethnicity:
                return 1.1-reduction
            elif gender:
                return 1.0-reduction
            elif ethnicity:
                return 1.5-reduction
            else:
                return 1.2-reduction
        elif age < 40:
            if gender and ethnicity:
                return 1.1-reduction
            elif gender:
                return 0.9-reduction
            elif ethnicity:
                return 1.4-reduction
            else:
                return 1.2
        elif age < 55:
            if gender and ethnicity:
                return 1.0-reduction
            elif gender:
                return 0.9-reduction
            elif ethnicity:
                return 1.3-reduction
            else:
                return 1.1-reduction
        elif age < 66:
            if gender and ethnicity:
                return 1.0-reduction
            elif gender:
                return 0.8-reduction
            elif ethnicity:
                return 1.3-reduction
            else:
                return 1.1-reduction
        else:
            if gender and ethnicity:
                return 0.9-reduction
            elif gender:
                return 0.8-reduction
            elif ethnicity:
                return 1.2-reduction
            else:
                return 1.0-reduction

    # Original formula from 2009
    # creatinine_b = m.pow(eGFR / (175 * (int(age)**(-0.203)) * (0.742**(gender)) * 1.21**(ethnicity)), -1 / 1.154)
    # if creatinine_b > 1.6:
    #     creatinine_b = 1.6
    # if creatinine_b < 1.0:
    #     creatinine_b = 1.0

    # return creatinine_b


def get_dialysis(p_dfv):
    if len(p_dfv[ckd_hirid_key].dropna()) == 0:  # no CKD patient
        if np.all([int(i) < 2 for i in p_dfv[ckd_hirid_key].dropna().values]):
            min_t = pd.NaT
            for dialysis_hirid_key in dialysis_hirid_keys:
                if len(p_dfv[dialysis_hirid_key].dropna()) > 0:
                    t_this = pd.to_datetime(p_dfv[[time_hirid_key, dialysis_hirid_key]].dropna()[time_hirid_key].values[0])  # time
                    if not t_this > min_t:
                        min_t = t_this
            return min_t
    return pd.NaT


def get_hr(p_dfv):
    hr = p_dfv[[hr_hirid_key, time_hirid_key]]
    hr = hr.sort_values(by=[time_hirid_key])
    hr = hr.reset_index(drop=True).set_index(time_hirid_key)
    hr = hr.resample('5 min').mean().interpolate(method='linear', limit=12)  # 1h = 12 * 5 min
    # hr = hr.dropna()
    hr = hr.reset_index()
    return hr


def first_valid_value(x):
    if x.first_valid_index() is None:
        return None
    else:
        return x[x.first_valid_index()]


def get_creatinine(p_dfv, pid, dataset='HiRID', c_base='min'):
    creatinine_b = Baseline(p_dfv, pid, 75, p_dfv[age_hirid_key].notnull().values[0], p_dfv[gender_hirid_key].notnull().values[0],p_dfv[weight_hirid_key].notnull().values[0])

    creatinine = p_dfv[[creatinine_hirid_key, time_hirid_key]]
    creatinine = creatinine.dropna(how='any')
    # print(creatinine)
    creatinine = creatinine.sort_values(by=[time_hirid_key])

    creatinine[creatinine_hirid_key] = creatinine[creatinine_hirid_key].apply(pd.to_numeric, errors='coerce')
    creatinine[creatinine_hirid_key] = [i / 88.4 if i > 5 else i for i in creatinine[creatinine_hirid_key].values]  # convert mmol/L to mg/dl
    creatinine = creatinine.dropna(how='any')

    if 'min' in c_base:
        creatinine_b = np.nanmin([creatinine_b, creatinine[creatinine_hirid_key].min()])
    elif 'first' in c_base:
        if len(creatinine) > 0:
            creatinine_b = first_valid_value(creatinine[creatinine_hirid_key])

    creatinine = creatinine.reset_index(drop=True).set_index(time_hirid_key)
    creatinine = creatinine.resample(gridsize_str).mean().interpolate(method='linear', limit=int(ffill_horizon / pd.Timedelta(gridsize_str)))
    creatinine = creatinine.reset_index()

    return creatinine, creatinine_b


def get_urine(p_dfv, normalize=True, fill=True, grid=pd.DataFrame(columns=[time_hirid_key])):
    urine = p_dfv[[urine_hirid_key, time_hirid_key]].dropna().reset_index(drop=True)
    urine[urine_hirid_key] = urine[urine_hirid_key].apply(pd.to_numeric, errors='coerce')
    urine = urine.sort_values(by=[time_hirid_key])
    urine['measurement'] = 1

    if len(grid) > 0:
        ur = []
        for idx, row in urine.iterrows():
            if idx == 0:
                ur.append(np.nan)
            else:
                if np.any(np.isnan(grid[np.logical_and(t < pd.to_datetime(grid[time_hirid_key].values), pd.to_datetime(grid[time_hirid_key].values) <= row[time_hirid_key])][hr_hirid_key].values)):
                    t = row[time_hirid_key]
                    ur.append(np.nan)
                elif len(grid[np.logical_and(t < pd.to_datetime(grid[time_hirid_key].values), pd.to_datetime(grid[time_hirid_key].values) <= row[time_hirid_key])][hr_hirid_key].values) == 0:
                    t = row[time_hirid_key]
                    ur.append(np.nan)
                else:
                    ur.append(row[urine_hirid_key])
            t = row[time_hirid_key]
        urine[urine_hirid_key] = ur
        # print(urine)

    if normalize:
        weight = first_valid_value(p_dfv[weight_hirid_key])
        if type(weight) == list:
            if len(weight) > 0:
                weight = int(weight[0])
            else:
                weight = False
        else:
            try:
                weight = int(weight)
            except:
                weight = False
        if weight == False or weight == 0:
            try:
                weight = imputed_weight(p_dfv[age_hirid_key].notnull().values[0], p_dfv[gender_hirid_key].notnull().values[0])
            except:
                weight = 80
        urine[urine_hirid_key] = urine[urine_hirid_key].apply(lambda x: float(x) / weight)

    if fill:
        urine = urine.reset_index(drop=True).set_index(time_hirid_key)
        urine = urine.resample(gridsize_str).mean()
        urine[urine_hirid_key] = urine[urine_hirid_key].fillna(method='bfill')
        urine = urine.reset_index()

    return urine


def get_grid(p_dfv):
    cp = p_dfv[p_dfv[time_hirid_key] >= p_dfv.dropna(subset=[hr_hirid_key], how='any')[time_hirid_key].min()]
    cp = cp[cp[time_hirid_key] <= cp.dropna(subset=[hr_hirid_key], how='any')[time_hirid_key].max()]
    cp = cp.reset_index(drop=True).set_index(time_hirid_key)
    cp = cp.resample(gridsize_str).mean().fillna(method='bfill')
    cp = cp.reset_index()
    return cp


def put_p_on_grid(p_dfv, pid, c_base='min_i'):
    grid = get_grid(p_dfv)
    urine = get_urine(p_dfv, normalize=True, grid=grid)
    # print(urine)
    creatinine, creatinine_b = get_creatinine(p_dfv, pid, c_base=c_base)
    # print(creatinine)
    dialysis = get_dialysis(p_dfv)
    # print(dialysis)
    grid = get_grid(p_dfv)
    # print(grid)
    return urine, creatinine, creatinine_b, dialysis, grid


def get_u_mean_from_dataframe(urine, window_start, window_stop):
    if window_stop <= urine[time_hirid_key].max():
        subs_urine = urine[(urine[time_hirid_key] > window_start) & (urine[time_hirid_key] <= window_stop)][urine_hirid_key]
        if not subs_urine.isnull().values.any():
            return subs_urine.mean(skipna=False)
    return np.nan


def get_last_value_from_dataframe(pid, df, time, dt, key, time_key):
    subs_cr = df[np.logical_and(time - dt < pd.to_datetime(df[time_key].values), pd.to_datetime(df[time_key].values) <= time)][key]
    if len(subs_cr) > 0:
        return subs_cr.values[-1]
    else:
        if key == creatinine_hirid_key:
            last_measurements = creatinine_before_ICU(pid, time, pd.Timedelta('5 days'))
            if len(last_measurements) > 0:
                return last_measurements['lab_nval'].values[-1]
        return np.nan


def get_min_value_from_dataframe(pid, df, time, dt, key, time_key):
    subs_cr = df[np.logical_and(time - dt < pd.to_datetime(df[time_key].values), pd.to_datetime(df[time_key].values) <= time)][key]
    subs_cr_before = creatinine_before_ICU(pid, time, dt)
    min_creatinine = np.nan
    if len(subs_cr_before) > 0:
        min_creatinine = subs_cr_before['lab_nval'].min()
    if len(subs_cr) > 0:
        min_creatinine = np.nanmin([subs_cr.min(), min_creatinine])
    return min_creatinine


def get_median_from_reduced(urine, window_start, window_stop, key):
    # print('Median', window_start, window_stop)
    df_single = urine[urine['measurement'] == 1]
    # print(df_single)
    subs_df = df_single[np.logical_and(window_start < pd.to_datetime(df_single[time_hirid_key].values), pd.to_datetime(df_single[time_hirid_key].values) <= window_stop)][key]
    # print(subs_df)
    if len(subs_df) > 0:
        return subs_df.mean()
    return np.nan
