
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def expand_row_nonbinary(r):
    hours = pd.date_range(r['charttime_start'], r['charttime_end'], freq='H', closed='left')
    return pd.DataFrame({'stay_id': r['stay_id'], 'charttime': hours, 'rate': r['rate']})

def expand_row_binary(r):
    hours = pd.date_range(r['starttime'], r['endtime'], freq='H', closed='left')
    return pd.DataFrame({'stay_id': r['stay_id'], 'AbsDatetime': hours, 'value': 1})

def expand_row_binary_v2(r):
    hours = pd.date_range(r['charttime_start'], r['charttime_end'], freq='H', closed='left')
    return pd.DataFrame({'hadm_id': r['hadm_id'], 'AbsDatetime': hours, 'value': 1})

# patient stays

icustays = pd.read_parquet('Variable Extractions/ICU_stays.parquet')

## vm31
vm31 = pd.read_parquet("Variable Extractions/vm31.parquet")
vm31.rename(columns={'rate':'vm31'}, inplace=True)
merged = vm31

# procedure events
## vm275
vm275 = pd.read_parquet("Variable Extractions/vm275.parquet")
vm275 = vm275.groupby(['stay_id', 'starttime'] ).max()[['value']]
vm275.rename(columns={'value':'vm275'},inplace=True)
merged = pd.concat([merged, vm275])

# Output events
for variables in ['vm1', 'vm276', 'vm131', 'vm156', 'vm154', 'vm162', 'vm176', 'vm21', 'vm65']:
    vm = pd.read_parquet('Variable Extractions/{}.parquet'.format(variables))
    vm.index = vm.index.set_names(['stay_id', 'starttime'])
    vm.rename(columns={'value':variables},inplace=True)
    merged = pd.concat([merged, vm])


# Inputevents
## pm69
pm69_compact = pd.read_parquet("Variable Extractions/pm69.parquet")
pm69_compact["amount"] = pm69_compact["amount"] / 1000
pm69_compact["charttime"] = pm69_compact.apply(
    lambda row: pd.date_range(
        row["starttime"], row["endtime"], freq="H", closed="left"
    ),
    axis=1,
)
pm69_expanded = pm69_compact.explode("charttime")
pm69_expanded.drop(columns=["hadm_id", "itemid", "starttime", "endtime"], inplace=True)
pm69_expanded.rename(['charttime':'starttime'], inplace=True)
pm69_expanded.set_index(['stay_id', 'starttime'],inplace=True)
pm69_expanded.rename(columns={"amount": "pm69"}, inplace=True)
merged = merged.merge(pm69_expanded, left_index=True, right_index=True, how='left')

## pm93
pm93_compact = pd.read_parquet(
   'Variable Extractions/pm93.parquet'
)

pm93_expanded = (
    pm93_compact.groupby("stay_id")
    .apply(lambda x: pd.concat([expand_row_nonbinary(row) for _, row in x.iterrows()]))
    .reset_index(drop=True)
)
pm93_expanded = pm93_expanded.set_index(["stay_id", "charttime"])["rate"]
pm93_expanded = pm93_expanded.astype(bool).astype(int)
pm93_expanded = pd.DataFrame(pm93_expanded)
pm93_expanded = pm93_expanded.rename(columns={"rate": "pm93"})
pm93_expanded.rename(['AbsDatetime':'starttime'], inplace=True)
pm93_expanded = pm93_expanded.rename_axis(["stay_id", "starttime"])
pm93_expanded = pm93_expanded[~pm93_expanded.index.duplicated(keep='first')]
merged = merged.merge(pm93_expanded, left_index=True, right_index=True, how='left')

## pm73
pm73_expanded = pd.read_parquet("Variable Extractions/pm73.parquet")
pm73_expanded.rename(columns={'charttime_start': 'AbsDatetime', 'amount': 'pm73'}, inplace=True)
pm73_expanded = pm73_expanded[['pm73']]
merged = pd.concat([merged, pm73_expanded])

## pm101
pm101_compact = pd.read_parquet("Variable Extractions/pm101.parquet")
pm101_compact.rename(columns={'charttime_start':'starttime', 'charttime_end':'endtime'}, inplace=True)
pm101_compact = pm101_compact.iloc[:1000]
pm101_compact = (
    pm101_compact.groupby("stay_id")
    .apply(lambda x: pd.concat([expand_row_binary(row) for _, row in x.iterrows()]))
    .reset_index(drop=True)
)
pm101_expanded = pd.DataFrame(pm101_compact)
pm101_expanded = pm101_expanded.rename(columns={"value": "pm101"})
pm101_expanded = pm101_expanded[~pm101_expanded.index.duplicated(keep='first')]
pm101_expanded.rename(['AbsDatetime':'starttime'], inplace=True)
pm101_expanded.set_index(['stay_id', 'starttime'],inplace=True)
merged = merged.merge(pm101_expanded, left_index=True, right_index=True, how='left')

## pm109
pm109_compact = pd.read_parquet('Variable Extractions/pm109.parquet')
pm109_compact = pm109_compact.drop('medication', 1)
pm109_compact = pm109_compact.iloc[:1000]

pm109_compact = (
    pm109_compact.groupby("hadm_id")
    .apply(lambda x: pd.concat([expand_row_binary_v2(row) for _, row in x.iterrows()]))
    .reset_index(drop=True)
)
pm109_compact = pm109_compact.loc[pm109_compact.hadm_id.isin(icustays.index)]
pm109_compact = pm109_compact.drop_duplicates()
pm109_compact = pm109_compact.set_index("hadm_id")
result = pm109_compact.reset_index().merge(
    icustays.reset_index(), on="hadm_id", how="left"
)
pm109_expanded.rename(['AbsDatetime':'starttime'], inplace=True)
pm109_expanded = (
    result.drop("hadm_id", 1)
    .set_index(["stay_id", "starttime"])
    .rename(columns={"value": "pm109"})
)[['pm109']]
merged = merged.merge(pm109_expanded, left_index=True, right_index=True, how='left')

## pm73
pm73 = pd.read_parquet("Variable Extractions/pm73.parquet")
pm73 = pm73[['amount']].rename(columns={'amount':'pm73'})
merged = pd.concat([merged, pm73])

# ## pm104
pm104_compact = pd.read_parquet('Variable Extractions/pm104.parquet')
pm104_compact = pm104_compact.iloc[:1000]
pm104_expanded = (
    pm104_compact.groupby("stay_id")
    .apply(lambda x: pd.concat([expand_row_binary(row) for _, row in x.iterrows()]))
    .reset_index(drop=True)
)
pm104_expanded.rename(['AbsDatetime':'starttime'], inplace=True)
pm104_expanded.set_index(['stay_id', 'starttime'],inplace=True)
merged = merged.merge(pm104_expanded, left_index=True, right_index=True, how='left')

## pm86
pm86_compact = pd.read_parquet('Variable Extractions/pm86.parquet').iloc[:1000]
pm86_expanded = (
    pm86_compact.groupby("stay_id")
    .apply(lambda x: pd.concat([expand_row_binary(row) for _, row in x.iterrows()]))
    .reset_index(drop=True)
)
pm86_expanded.rename(['AbsDatetime':'starttime'], inplace=True)
pm86_expanded.set_index(['stay_id', 'starttime'],inplace=True)
merged = merged.merge(pm86_expanded, left_index=True, right_index=True, how='left')

## pm41
pm41 = pd.read_parquet("Variable Extractions/pm41.parquet")
pm41 = pm41[['rate']]
pm41.rename(columns={'rate':'pm41'},inplace=True)
merged = pd.concat([merged, pm41])

## pm35
pm35 = pd.read_parquet("Variable Extractions/pm35.parquet")
pm35 = pm35[['rate']]
pm35.rename(columns={'rate':'pm35'},inplace=True)
merged = pd.concat([merged, pm35])

## static variables
icustays.set_index('stay_id', inplace=True)
merged = merged.join(icustays[['Surgical', 'Emergency']], on='stay_id', how='left')
merged.index = merged.index.set_names(['stay_id', 'AbsDatetime'])
merged.to_parquet('Variable Extractions/MIMIC_merged.parquet')

