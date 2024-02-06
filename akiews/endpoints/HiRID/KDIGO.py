import pandas as pd
import numpy as np

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

from AKI import get_u_mean_from_dataframe, get_last_value_from_dataframe, put_p_on_grid, get_min_value_from_dataframe, get_median_from_reduced
from paths import *


def KDIGO(p_dfv, pid, halflife=8, c_base='min_i', min_gap=12 * 60 * pd.Timedelta('1 min'), printing=False):
    urine, creatinine, creatinine_b, dialysis, Events = put_p_on_grid(p_dfv, pid, c_base=c_base)

    Events = pd.DataFrame(index=Events[time_hirid_key], columns=ep_types)
    Events['PatientID'] = pid
    if printing:
        print(pid, halflife, c_base, min_gap)
        print(urine)
        print(creatinine)
        print(dialysis)
    for time, row in Events.iterrows():
        u_mean_24 = get_u_mean_from_dataframe(urine, time, time + 24 * 60 * reference_time)
        u_mean_12 = get_u_mean_from_dataframe(urine, time, time + 12 * 60 * reference_time)
        u_mean_6 = get_u_mean_from_dataframe(urine, time, time + 6 * 60 * reference_time)
        u_median_24 = get_median_from_reduced(urine, time, time + 24 * 60 * reference_time, urine_hirid_key)
        u_median_12 = get_median_from_reduced(urine, time, time + 12 * 60 * reference_time, urine_hirid_key)
        u_median_6 = get_median_from_reduced(urine, time, time + 6 * 60 * reference_time, urine_hirid_key)
        next_c_value = get_last_value_from_dataframe(pid, creatinine, time, gridsize, creatinine_hirid_key, time_hirid_key)
        creatinine_48_value = get_min_value_from_dataframe(pid, creatinine, time, ffill_horizon, creatinine_hirid_key, time_hirid_key)

        if time > dialysis:
            Events.loc[time, '3.d'] = 1
        else:
            Events.loc[time, '3.d'] = 0

        if u_mean_12 == 0. and u_median_12 == 0.:
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

    if printing:
        print(pid, 'regions')
        print(Events)

    # print('Postprocessing')
    # postprocessing urine events
    # first: merge consecutive regions
    # for column_urine in ['3.u', '2.u', '1.u']:
    #     prev_ep_time = pd.NaT
    #     start_region = False
    #     start_time = pd.NaT
    #     for index, row in Events.iterrows():
    #         if row[column_urine] != 1:
    #             start_region = True
    #         if row[column_urine] == 1:
    #             if start_region:
    #                 if index - prev_ep_time < min_gap:
    #                     # print('Merged')
    #                     Events.loc[prev_ep_time:index, column_urine] = 1
    #             else:
    #                 prev_ep_time = index
    # print(Events)
    # second: tructuate ep regions by threhold
    for column_urine, threshold in zip(['3.u', '2.u', '1.u'], [0.3, 0.5, 0.5]):
        for k, v in Events.groupby((Events[column_urine].shift() != Events[column_urine]).cumsum()):
            if v[column_urine].values[0] == 1:
                # print(v)
                for idx, row in v.iterrows():
                    val = get_last_value_from_dataframe(pid, urine, idx, gridsize, urine_hirid_key, time_hirid_key)
                    # print(val)
                    if val >= threshold:
                        break
                    elif not pd.isnull(val):
                        # print("Inserted zero in beginning")
                        Events.loc[idx, column_urine] = 0

                for idx, row in v[::-1].iterrows():
                    val = get_last_value_from_dataframe(pid, urine, idx, gridsize, urine_hirid_key, time_hirid_key)
                    # print(val)
                    if val >= threshold:
                        break
                    elif not pd.isnull(val):
                        # print("Inserted zero at end")
                        Events.loc[idx, column_urine] = 0
                # print(Events.loc[v[time_hirid_key].min():v[time_hirid_key].max(), :])
    if np.any(Events['1.u'].values>0)and np.all(Events['1.i'].values==0)and np.all(Events['1.b'].values==0):
        print(pid, 'events')
        print(Events.loc[Events['1.u'].values>0,:])
    return Events
