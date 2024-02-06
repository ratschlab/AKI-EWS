import pandas as pd
import matplotlib.pyplot as plt
from paths import *


itemid_creatinine = [50912, 52541, 52019]
itemid_creatinine_chartevents = [220615]

itemid_urine = [226557, 226558, 226559, 226561, 226563, 226564, 226565, 227489, 226566, 226560, 226567, 226627]  # , 226560]  # ask agagin void?



def analyze_creatinine():
    iter_csv = pd.read_csv(base_csv+'/hosp/labevents.csv', iterator=True, chunksize=100000, usecols=['itemid','hadm_id','value','charttime'])
    creatinine = pd.concat([chunk[chunk['itemid'].isin(itemid_creatinine)] for chunk in iter_csv])
    creatinine = creatinine.dropna(subset=['hadm_id','value'], how='any')
    creatinine['charttime'] = pd.to_datetime(creatinine['charttime'])
    static = pd.read_csv('static.csv', parse_dates=['intime', 'outtime'])
    creatinine = pd.merge(left=creatinine,right=static,on=['hadm_id'])
    creatinine = creatinine.drop_duplicates(subset=["hadm_id", "charttime", "value"])

    creatinine_inside = creatinine[np.logical_and(creatinine['charttime']<creatinine['outtime'],creatinine['charttime']>creatinine['intime'])]

    print('all of labevents')
    print(len(creatinine))
    print('inside ICU stay')
    print(len(creatinine_inside))


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
    print(len(creatinie_inside_ce))

    creatinine_inside = creatinine_inside.append(creatinie_inside_ce)
    creatinine_inside_first = creatinine_inside.drop_duplicates(subset=["hadm_id", "charttime", "value"],keep='first')
    creatinine_inside_last = creatinine_inside.drop_duplicates(subset=["hadm_id", "charttime", "value"],keep='last')

    print('total inside')
    print(len(creatinine_inside))
    print('unqiue')
    print(len(creatinine_inside_last))
    print('unique chartevents')
    print(len(creatinine_inside_last[creatinine_inside_last['itemid']==220615]))
    print('unique labevents')
    print(len(creatinine_inside_first[creatinine_inside_first['itemid'].isin(itemid_creatinine)]))


analyze_creatinine()

