from glob import glob
import pandas as pd
import numpy as np
import os


base_csv = '/cluster/work/grlab/clinical/hirid2/research/misc_derived/mimic-iv-1.0/'
base_ep = '/cluster/work/grlab/clinical/hirid2/research/3b_endpoints_renal/MIMIC-IV/'

dialysis_key = 'vm202'  # or vm200 for CKD patients
weight_key = 'patientweight'  # multiple recordings per patient possible
time_key = 'charttime'
creatinine_key = 'value'
patient_key = 'stay_id'
urine_key = 'value'
reference_time = pd.Timedelta(np.timedelta64(1, 'm'))
ffill_horizon = 48 * 60 * reference_time
gridsize = 60 * pd.Timedelta(np.timedelta64(1, 'm'))
gridsize_str = '1 h'

ep_types = np.asarray(['1.r.u', '1.u', '1.i', '1.b', '2.r.u', '2.u', '2.b', '3.au', '3.r.u', '3.u', '3.b', '3.i', '3.d', 'endpoint_status'])

one_columns = ['1.u', '1.i', '1.b']
two_columns = ['2.u', '2.b']
three_columns = ['3.au', '3.u', '3.b', '3.i', '3.d']


one_columns_creatinine = ['1.i', '1.b']
two_columns_creatinine = ['2.b']
three_columns_creatinine = [ '3.b', '3.i']

one_columns_urine = ['1.u']
two_columns_urine = ['2.u']
three_columns_urine = ['3.au', '3.u','3.d']


