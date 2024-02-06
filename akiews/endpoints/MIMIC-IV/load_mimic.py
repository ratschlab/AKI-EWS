import pandas as pd
import matplotlib.pyplot as plt
from paths import *


itemid_creatinine = [50912, 52541, 52019]
itemid_creatinine_chartevents = [220615]

itemid_urine = [226557, 226558, 226559, 226561, 226563, 226564, 226565, 227489, 226566, 226560, 226567, 226627]  # , 226560]  # ask agagin void?


def extract_urine():
    iter_csv = pd.read_csv(base_csv+'/icu/outputevents.csv', iterator=True, chunksize=100000, usecols=['itemid','value','hadm_id','valueuom','charttime'])
    urine = pd.concat([chunk[chunk['itemid'].isin(itemid_urine)] for chunk in iter_csv])
    urine = urine.dropna(subset=['hadm_id','value'], how='any')
    urine['charttime'] = pd.to_datetime(urine['charttime'])
    print(urine)

    iter_csv = pd.read_csv(base_csv+'/icu/inputevents.csv', iterator=True, chunksize=100000, usecols=['hadm_id','starttime','patientweight'])
    weight = pd.concat([chunk.dropna(how='any') for chunk in iter_csv])
    weight['charttime'] = pd.to_datetime(weight['starttime'])
    weight = weight.drop(columns=['starttime'])
    print(weight)
    urine = pd.merge(left=urine, right=weight, on=['hadm_id', 'charttime'])


    static = pd.read_csv('static.csv', parse_dates=['intime', 'outtime'])
    urine = pd.merge(left=urine,right=static[['hadm_id','stay_id']],on=['hadm_id'])
    print(urine)
    # urine = urine.drop_duplicates(subset=['hadm_id', 'charttime','value'])
    urine = urine.reset_index(drop=True)

    return urine

def analyze_creatinine():
    iter_csv = pd.read_csv(base_csv+'/hosp/labevents.csv', iterator=True, chunksize=100000, usecols=['itemid','hadm_id','value','charttime'])
    creatinine = pd.concat([chunk[chunk['itemid'].isin(itemid_creatinine)] for chunk in iter_csv])
    creatinine = creatinine.dropna(subset=['hadm_id','value'], how='any')
    creatinine['charttime'] = pd.to_datetime(creatinine['charttime'])
    static = pd.read_csv('static.csv', parse_dates=['intime', 'outtime'])
    creatinine = pd.merge(left=creatinine,right=static,on=['hadm_id'])
    creatinine = creatinine.drop_duplicates(subset=["hadm_id", "charttime", "value"])

    creatinie_inside = creatinine[np.logical_and(creatinine['charttime']<creatinine['outtime'],creatinine['charttime']>creatinine['intime'])]

    print('all of labevents')
    print(len(creatinine))
    print('inside ICU stay')
    print(len(creatinie_inside))


    iter_csv = pd.read_csv(base_csv+'/icu/chartevents.csv', iterator=True, chunksize=100000, usecols=['itemid','hadm_id','value','charttime'])
    creatinine_ce = pd.concat([chunk[chunk['itemid'].isin(itemid_creatinine_chartevents)] for chunk in iter_csv])
    creatinine_ce = creatinine_ce.dropna(subset=['hadm_id','value'], how='any')
    creatinine_ce['charttime'] = pd.to_datetime(creatinine_ce['charttime'])
    creatinine_ce = pd.merge(left=creatinine_ce,right=static,on=['hadm_id'])
    creatinine_ce = creatinine_ce.drop_duplicates(subset=["hadm_id", "charttime", "value"])


    creatinie_inside_ce = creatinine_ce[np.logical_and(creatinine_ce['charttime']<creatinine_ce['outtime'],creatinine_ce['charttime']>creatinine_ce['intime'])]

    print('all of chartevents')
    print(len(creatinine_ce))
    print('all of chartevents inside')
    print(len(creatinine_inside_ce))

    creatinine_inside = creatinine_inside.append(creatinie_inside_ce)
    creatinine_inside_first = creatinine_inside.drop_duplicates(subset=["hadm_id", "charttime", "value"],keep='first')
    creatinine_inside_last = creatinine_inside.drop_duplicates(subset=["hadm_id", "charttime", "value"],keep='last')

    print('total inside')
    print(creatinine_inside)
    print('unique chartevents')
    print(len(creatinine_inside_first[creatinine_inside_first['itemid']==220615]))
    print('unique labevents')
    print(len(creatinine_inside_last[creatinine_inside_last['itemid'].isin(itemid_creatinine)]))





def extract_creatinine():
    iter_csv = pd.read_csv(base_csv+'/hosp/labevents.csv', iterator=True, chunksize=100000, usecols=['itemid','hadm_id','value','charttime'])
    creatinine = pd.concat([chunk[chunk['itemid'].isin(itemid_creatinine)] for chunk in iter_csv])
    creatinine = creatinine.dropna(subset=['hadm_id','value'], how='any')
    creatinine['charttime'] = pd.to_datetime(creatinine['charttime'])


    iter_csv = pd.read_csv(base_csv+'/icu/chartevents.csv', iterator=True, chunksize=100000, usecols=['itemid','hadm_id','value','charttime'])
    creatinine_ce = pd.concat([chunk[chunk['itemid'].isin(itemid_creatinine_chartevents)] for chunk in iter_csv])
    creatinine_ce = creatinine_ce.dropna(subset=['hadm_id','value'], how='any')
    creatinine_ce['charttime'] = pd.to_datetime(creatinine_ce['charttime'])

    creatinine = creatinine.append(creatinine_ce)
    # creatinine = creatinine.drop_duplicates(subset=["hadm_id", "charttime", "value"])

    static = pd.read_csv('static.csv', parse_dates=['intime', 'outtime'])
    creatinine = pd.merge(left=creatinine,right=static[['hadm_id','stay_id']],on=['hadm_id'])
    creatinine = creatinine.reset_index(drop=True)

    creatinine = creatinine.sort_values(by=['hadm_id','charttime'])

    return creatinine


def extract_global():

    subject_hadm = pd.read_csv(base_csv+'/core/admissions.csv', usecols=['subject_id','hadm_id','ethnicity'])
    subject_rest = pd.read_csv(base_csv+'/core/patients.csv', usecols=['subject_id','anchor_age','gender'])
    static = pd.merge(left=subject_hadm, right=subject_rest, on=['subject_id'])
    static = static.drop(columns=['subject_id'])


    times = pd.read_csv(base_csv+'/icu/icustays.csv', usecols=['hadm_id','stay_id','intime','outtime'])
    times = times.dropna(how='any')
    times['intime'] = pd.to_datetime(times['intime'])
    times['outtime'] = pd.to_datetime(times['outtime'])
    static = pd.merge(left=static, right=times, on=['hadm_id'])

    static = static.reset_index(drop=True)

    return static


def get_all_p_icu():
    return static[patient_key].values

def get_creatinine_pid(pid):
    return all_creatinine[all_creatinine[patient_key] == pid]


def get_static_pid(pid):
    return static[static[patient_key] == pid][['ethnicity', 'anchor_age', 'gender', 'intime', 'outtime']].values


def get_urine_pid(pid):
    return all_urine[all_urine[patient_key] == pid]


def get_RRT_pid(pid):
    return pd.DataFrame(columns=[patient_key, 'charrtime'])

# static = extract_global()
# static.to_csv('static.csv', index=False)
static = pd.read_csv('static.csv', parse_dates=['intime', 'outtime'])
# print(static)

# all_creatinine = extract_creatinine()
# all_creatinine.to_csv('all_creatinine.csv', index=False)
all_creatinine = pd.read_csv('all_creatinine.csv', parse_dates=['charttime'])

# analyze_creatinine()

# all_urine = extract_urine()
# all_urine.to_csv('all_urine.csv', index=False)
all_urine = pd.read_csv('all_urine.csv', parse_dates=['charttime'])
# print(all_urine)


# for pid in get_all_p():
#     get_creatinine_pid(pid)
