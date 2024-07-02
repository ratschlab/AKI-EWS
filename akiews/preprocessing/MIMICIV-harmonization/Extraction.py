

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_pharm(events, itemids):
    variable = pd.DataFrame()
    for item in itemids:
        variable = pd.concat([variable, events[events['medication'].str.contains(item, case=False, na=False)]])
    variable = variable.sort_values('hadm_id')
    return variable

def get_events(events, itemids):
    vm = events[events['itemid'].isin(itemids)]
    unit = vm.totalamountuom.unique()
    return vm, unit

def get_end_time(row, acting_period_dict):
    medication = row['medication'].lower()
    for key, value in acting_period_dict.items():
        if key.lower() in medication:
            duration = pd.Timedelta(value, 'm')
            return row['charttime_start'] + duration
    return None

def get_end_time_for_inputevents(row, acting_period_dict):
    medication = row['itemid']
    
    for key, value in acting_period_dict.items():
        if key == medication:
            duration = pd.Timedelta(value, 'm') 
            return row['starttime'] + duration
    return None


# patient stays

admission = pd.read_csv('core/admissions.csv.gz', compression='gzip')
admission = admission.set_index('hadm_id')

icustays = pd.read_csv('icu/icustays.csv.gz', compression='gzip')
icustays = icustays.set_index('hadm_id')
icustays = icustays[['subject_id', 'stay_id', 'intime', 'outtime', 'los']]

icustays['Surgical status'] = admission[admission['admission_type'] == 'SURGICAL SAME DAY ADMISSION']['admission_type']
icustays['Emergency status'] = admission[admission['admission_type'].isin(
    ['URGENT', 'DIRECT EMER.', 'EW EMER.'])]['admission_type']

icustays['Emergency status'] = icustays['Emergency status'].notnull().astype('int')
icustays['Surgical status'] = icustays['Surgical status'].notnull().astype('int')

icustays = icustays[
    ["subject_id", "stay_id", "intime", "Surgical status", "Emergency status"]
]
icustays.rename(
    columns={
        "Emergency status": "Emergency",
        "Surgical status": "Surgical",
        "intime": "starttime",
    },
    inplace=True,
)
icustays.to_parquet('Variable Extractions/ICU_stays.parquet')


#  inputevents

inputevents = pd.read_csv('icu/inputevents.csv.gz', compression='gzip')
pharmacy = pd.read_csv('hosp/pharmacy.csv.gz', compression='gzip')


## IN, vm31, unit: ml/hour
IN = inputevents[inputevents["rateuom"].str.contains("ml", case=False, na=False)]

IN.loc[IN[IN["rateuom"] == "mL/kg/hour"].index, "rate"] = (
    IN.loc[IN[IN["rateuom"] == "mL/kg/hour"].index, "rate"]
    * IN.loc[IN[IN["rateuom"] == "mL/kg/hour"].index, "patientweight"]
)

IN.loc[IN[IN["rateuom"] == "mL/kg/hour"].index, "rateuom"] = "mL/hour"

IN.loc[IN[IN["rateuom"] == "mL/min"].index, "rate"] = (
    IN.loc[IN[IN["rateuom"] == "mL/min"].index, "rate"] * 60
)

IN.loc[IN[IN["rateuom"] == "mL/min"].index, "rateuom"] = "mL/hour"

IN = IN[["stay_id", "starttime", "rate"]]
IN["starttime"] = pd.to_datetime(IN["starttime"]).dt.floor("H")
IN = IN.groupby(["stay_id", "starttime"]).sum()

IN.to_parquet("Variable Extractions/vm31.parquet")


## loop diuretics, pm 69, unit: mg/hour

pm69, units = get_events(inputevents, [221794, 229639, 228340])
pm69 = pm69.loc[pm69.ordercategorydescription == 'Drug Push']

acting_period_pm69 = {
    221794: 360,
    229639: 360,
    228340: 360,
}

pm69_compact = pm69[['hadm_id', 'stay_id', 'starttime', 'itemid', 'amount']]
pm69_compact['starttime'] = pd.to_datetime(pm69_compact['starttime']).dt.round('H')
pm69_compact = pm69_compact.dropna()
pm69_compact["endtime"] = pm69_compact.apply(
    lambda row: get_end_time_for_inputevents(row, acting_period_pm69), axis=1
)
pm69_compact.to_parquet('Variable Extractions/pm69.parquet')


## enteral feeding, pm93, binary

pm93, units = get_events(
    inputevents,
    [
        221206,
        229014,
        227975,
        227972,
        227973,
        227974,
        228131,
        228132,
        228133,
        228134,
        228135,
    ],
)

pm93['charttime_start'] = pd.to_datetime(pm93['starttime']).dt.round('H')
pm93_compact = pm93[['stay_id',  'charttime_start', 'rate']]
pm93_compact = pm93_compact.dropna()
pm93_compact['charttime_end'] = pm93_compact['charttime_start'] + pd.Timedelta(720, 'm')

pm93_compact.to_parquet('Variable Extractions/pm93.parquet')


## packed red blood cells, pm35, unit: ml/hour

a = [225168, 220996]
pm35 = inputevents[inputevents.itemid.isin(a)]
pm35.loc[
    pm35[pm35["rateuom"] == "mL/min"].index, "rate"
] = (
    pm35.loc[
        pm35[pm35["rateuom"] == "mL/min"].index,
        "rate",
    ]
    * 60
)
pm35 = pm35[
    ["stay_id", "starttime", "amount", "rate"]
]
pm35["starttime"] = pd.to_datetime(
    pm35["starttime"]
).dt.floor("H")
pm35 = pm35.groupby(["stay_id", "starttime"]).sum()
pm35.to_parquet(
    "Variable Extractions/pm35.parquet"
)


## Kalium, pm101, binary
pm101, units = get_events(inputevents,  [225166])
pm101['charttime_start'] = pd.to_datetime(pm101['starttime']).dt.round('H')
pm101_compact = pm101[['stay_id',  'charttime_start']]
pm101_compact = pm101_compact.dropna()
pm101_compact['charttime_end'] = pm101_compact['charttime_start'] + pd.Timedelta(1440, 'm')
pm101_compact.to_parquet('Variable Extractions/pm101.parquet')


## admin of abx, binary
a = pd.read_csv('mimiciv_abx_itemid.csv', header=None)[0].tolist()
pm73 = inputevents[inputevents.itemid.isin(a)]
pm73 = pm73[['stay_id', 'starttime', 'amount']]
pm73['starttime'] = pd.to_datetime(pm73['starttime']).dt.floor("H")
pm73 = pm73.groupby(['stay_id', 'starttime']).sum()
pm73['amount'] = (pm73['amount'] > 0).astype(int)
pm73['charttime_start'] = pm73.index.get_level_values(1)
pm73['charttime_end'] = pm73['charttime_start'] + pd.Timedelta(1440, 'm')
pm73.to_parquet('Variable Extractions/pm73.parquet')


## opiate, pm86, unit: mg/min
pm86, units = get_events(inputevents,  [221744, 225942, 221833, 225154, 227520])
pm86 = pm86.loc[pm86.ordercategorydescription == 'Drug Push']
pm86.loc[pm86.amountuom == 'mg', 'amount'] = pm86.loc[pm86.amountuom == 'mg', 'amount']* 1000

acting_period_pm86 = {
    221744:0,
    225942:0,
    221833:240,
    225154:240,
    227520:1440
}

pm86_compact = pm86[['hadm_id', 'stay_id', 'starttime', 'itemid', 'amount']]
pm86_compact['starttime'] = pd.to_datetime(pm86_compact['starttime']).dt.round('H')
pm86_compact = pm86_compact.dropna()
pm86_compact["endtime"] = pm86_compact.apply(
    lambda row: get_end_time_for_inputevents(row, acting_period_pm86), axis=1
)
pm86_compact.to_parquet('Variable Extractions/pm86.parquet')


## Mg, pm104, binary

pm104, units = get_events(inputevents,  [222011])
pm104['charttime_start'] = pd.to_datetime(pm104['starttime']).dt.round('H')
pm104_compact = pm104[['stay_id',  'charttime_start']]
pm104_compact = pm104_compact.dropna()
pm104_compact['charttime_end'] = pm104_compact['charttime_start'] + pd.Timedelta(1440, 'm')
pm104.to_parquet('Variable Extractions/pm104.parquet')


## dobutamine, pm41, unit: mg/min

a = [221653]
pm41 = inputevents[inputevents.itemid.isin(a)]
pm41['rate'] = pm41['rate'] * pm41['patientweight']
pm41['rate'] = pm41['rate']/1000
pm41 = pm41[['stay_id', 'starttime', 'amount', 'rate']]
pm41['starttime'] = pd.to_datetime(pm41['starttime']).dt.floor("H")
pm41 = pm41.groupby(['stay_id', 'starttime']).sum()
pm41.to_parquet('Variable Extractions/pm41.parquet')


## heparin, pm95, unit: ml/min

a = [225152, 225975, 229597, 230044]
pm95 = inputevents[inputevents.itemid.isin(a)]
pm95 = pm95[pm95["rateuom"].notnull()]
pm95.loc[pm95[pm95["rateuom"] == "units/kg/hour"].index, "rate"] = (
    pm95.loc[pm95[pm95["rateuom"] == "units/kg/hour"].index, "rate"]
    * pm95.loc[pm95[pm95["rateuom"] == "units/kg/hour"].index, "patientweight"]
)
pm95["rate"] = pm95["rate"] / 60
pm95 = pm95[["stay_id", "starttime", "amount", "rate"]]
pm95["starttime"] = pd.to_datetime(pm95["starttime"]).dt.floor("H")
pm95 = pm95.groupby(["stay_id", "starttime"]).sum()
pm95.to_parquet("Variable Extractions/pm95.parquet")

## Anti delirant medi, pm 109, Binary
pm109 = get_pharm(
    pharmacy,
    [
        "Haldol",
        "Remeron",
        "Seroquel",
        "Dipiperon",
        "Haloperidol",
        "Clonidine",
        "Nozinan",
        "Quetiapine",
        "Risperidone"
    ],
)
acting_period_pm109 = {
    "Haldol": 240,
    "Remeron": 720,
    "Seroquel": 720,
    "Dipiperon": 720,
    "Haloperidol": 240,
    "Clonidine":720,
    "Nozinan": 720,
    "Quetiapine": 720,
    "Risperidone": 720,
}
pm109['charttime_start'] = pd.to_datetime(pm109['starttime']).dt.round('H')
pm109_compact = pm109[['hadm_id', 'medication', 'charttime_start']]
pm109_compact = pm109_compact.dropna()
pm109_compact["charttime_end"] = pm109_compact.apply(
    lambda row: get_end_time(row, acting_period_pm109), axis=1
)
pm109_compact.to_parquet('Variable Extractions/pm109.parquet')


# procedure events

procedureevents = pd.read_csv('icu/procedureevents.csv.gz', compression='gzip')


## peritoneal dialysis vm275

vm275 = procedureevents.loc[procedureevents.itemid == 225805]
vm275[["hadm_id", "stay_id", "starttime", "endtime", "value", "valueuom"]].to_parquet(
    "Variable Extractions/vm275.parquet"
)


# output events

outputevents = pd.read_csv('icu/outputevents.csv.gz', compression='gzip')

## Urine, vm276, unit:ml

a = [
    226557,
    226558,
    226559,
    226560,
    226561,
    226563,
    226564,
    226565,
    226566,
    226567,
    226584,
    227510,
]
vm276 = outputevents[outputevents.itemid.isin(a)]
vm276["charttime"] = pd.to_datetime(vm276["charttime"]).dt.floor("H")
vm276 = vm276[["stay_id", "charttime", "value"]]
vm276.rename(columns={'charttime':'starttime'},inplace=True)
vm276 = vm276.groupby(["stay_id", "starttime"]).sum()
vm276.to_parquet("Variable Extractions/vm276.parquet")


# Chartevents

chartevents = pd.read_csv('icu/chartevents.csv.gz', compression='gzip')


## Heart rate, vm1, unit:bpm

a = [220045]
hr = chartevents[chartevents.itemid.isin(a)]
hr = hr[['stay_id', 'charttime', 'value']]
hr['value'] = hr['value'].astype(float)
hr['charttime'] = pd.to_datetime(hr['charttime']).dt.floor("H")
hr = hr.groupby(['stay_id', 'charttime']).max()
hr.to_parquet('Variable Extractions/vm1.parquet')

## weight, vm131, unit:kg
a = [224639]
weight = chartevents[chartevents.itemid.isin(a)]
weight = weight[['stay_id', 'charttime', 'value']]
weight['value'] = weight['value'].astype(float)
weight['charttime'] = pd.to_datetime(weight['charttime']).dt.floor("H")
weight = weight.groupby(['stay_id', 'charttime']).max()
weight.to_parquet('Variable Extractions/vm131.parquet')

## creatinine, vm156, unit:umol/kg
a = [220615]
creatinine = chartevents[chartevents.itemid.isin(a)]
creatinine = creatinine[['stay_id', 'charttime', 'value']]
creatinine['value'] = creatinine['value'].astype(float) * 88.42
creatinine['charttime'] = pd.to_datetime(creatinine['charttime']).dt.floor("H")
creatinine = creatinine.groupby(['stay_id', 'charttime']).max()
creatinine.to_parquet('Variable Extractions/vm156.parquet')

## Magnesium, vm154, mg/dL

a = [220635]
magnesium = chartevents[chartevents.itemid.isin(a)]
magnesium = magnesium[['stay_id', 'charttime', 'value']]
magnesium['charttime'] = pd.to_datetime(magnesium['charttime']).dt.floor("H")
magnesium = magnesium.groupby(['stay_id', 'charttime']).max()
magnesium.to_parquet('Variable Extractions/vm154.parquet')

## Bilirubine/total, vm162, unit:mg/dL
a = [220635]
bilirubine = chartevents[chartevents.itemid.isin(a)]
bilirubine = bilirubine[['stay_id', 'charttime', 'value']]
bilirubine['charttime'] = pd.to_datetime(bilirubine['charttime']).dt.floor("H")
bilirubine = bilirubine.groupby(['stay_id', 'charttime']).max()
bilirubine['value'] = bilirubine['value'].astype(float)
bilirubine.to_parquet('Variable Extractions/vm162.parquet')

## CRP, vm176, unit: mg/L
a = [227444]
CRP = chartevents[chartevents.itemid.isin(a)]
CRP = CRP[['stay_id', 'charttime', 'value']]
CRP['charttime'] = pd.to_datetime(CRP['charttime']).dt.floor("H")
CRP = CRP.groupby(['stay_id', 'charttime']).max()
CRP['value'] = CRP['value'].astype(float)
CRP.to_parquet('Variable Extractions/vm176.parquet')


## EtCO2, vm21, unit:
a = [228640]
EtCO2 = chartevents[chartevents.itemid.isin(a)]
EtCO2 = EtCO2[['stay_id', 'charttime', 'value']]
EtCO2['charttime'] = pd.to_datetime(EtCO2['charttime']).dt.floor("H")
EtCO2 = EtCO2.groupby(['stay_id', 'charttime']).max()
EtCO2['value'] = EtCO2['value'].astype(float)
EtCO2.to_parquet('Variable Extractions/vm21.parquet')

## RRset/m, vm65
a = [224688]
RRset = chartevents[chartevents.itemid.isin(a)]
RRset = RRset[['stay_id', 'charttime', 'value']]
RRset['charttime'] = pd.to_datetime(RRset['charttime']).dt.floor("H")
RRset = RRset.groupby(['stay_id', 'charttime']).max()
RRset['value'] = RRset['value'].astype(float)
RRset.to_parquet('Variable Extractions/vm65.parquet')

