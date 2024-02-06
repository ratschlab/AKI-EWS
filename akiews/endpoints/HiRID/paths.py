from glob import glob
import pandas as pd
import numpy as np

# For HiRid
# MODIFY THIS FOR DIFFERENT SOURCE FILE
endpoints_dir = '/cluster/work/grlab/clinical/hirid2/research/5b_imputed_renal/v8/reduced/point_est/'


# base_h5_hirid = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/3_merged/v6b/reduced/'
base_h5_hirid = '/cluster/work/grlab/clinical/hirid2/research/3_merged/v8/reduced/'
base_ep = '/cluster/work/grlab/clinical/hirid2/research/3b_endpoints_renal/220922/'
static_h5_hirid = './static.h5'
dialysis_h5_hirid = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/dialyses/dialyses_dynamic.h5'

cr_post = pd.read_csv('creatinin_post14.csv')
cr_pre = pd.read_csv('creatinin_pre14.csv')
creatinine_tests_pre_ICU = pd.concat([cr_pre, cr_post]).dropna().astype({'study_id': 'int32', 'lab_nval': 'int32'})
creatinine_tests_pre_ICU['lab_date'] = pd.to_datetime(creatinine_tests_pre_ICU['lab_date'])


creatinine_hirid_key = 'vm156'
urine_hirid_key = 'vm24'
hr_hirid_key = 'vm1'
age_hirid_key = 'Age'
exit_hirid_key = 'Discharge'
gender_hirid_key = 'Sex'
dialysis_hirid_keys = ['vm72', 'vm202', 'vm274', 'vm275', 'vm285', 'vm286', 'vm287']  # or vm200 for CKD patients
ckd_hirid_key = 'vm201'
dialysis_clinical_var = ['v10002319', 'v10002508', 'v10002509', 'v10002550']
weight_hirid_key = 'vm131'  # multiple recordings per patient possible
time_hirid_key = 'Datetime'  # using (pd.Timstamp(x)- pd.Timestamp(t) ) /pd.Timedelta(np.timedelta64(1,'m'))
time_endpoint_key = 'AbsDatetime'
admission_hirid_key = 'AdmissionTime'
reference_time = pd.Timedelta(np.timedelta64(1, 'm'))
ffill_horizon = 48 * 60 * reference_time
# gridsize = 6 * 60 * pd.Timedelta(np.timedelta64(1, 'm'))
# gridsize_str = '6 h'
# gridsize = 60 * pd.Timedelta(np.timedelta64(1, 'm'))
# gridsize_str = '1 h'
gridsize = 5 * pd.Timedelta(np.timedelta64(1, 'm'))
gridsize_str = '5 min'


ep_drugs = 'pm69'

# have the static df as global
static_hirid = pd.read_hdf(static_h5_hirid)[['PatientID', exit_hirid_key, age_hirid_key, gender_hirid_key, admission_hirid_key]]  # Column PatientID -> get pid from there?
all_pid_hirid = static_hirid['PatientID'].unique()


ep_types = np.asarray(['1.r.u', '1.u', '1.i', '1.b', '2.r.u', '2.u', '2.b', '3.au', '3.r.u', '3.u', '3.b', '3.i', '3.d', 'endpoint_status'])

one_columns = ['1.u', '1.i', '1.b','masked_urine']
two_columns = ['2.u', '2.b']
three_columns = ['3.au', '3.u', '3.b', '3.i', '3.d']


one_columns_creatinine = ['1.i', '1.b']
two_columns_creatinine = ['2.b']
three_columns_creatinine = [ '3.b', '3.i']

one_columns_urine = ['1.u','masked_urine']
two_columns_urine = ['2.u']
three_columns_urine = ['3.au', '3.u','3.d']

