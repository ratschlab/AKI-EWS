import pandas as pd
from glob import glob
import numpy as np


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


catheter_table = pd.DataFrame(columns=['endtime', 'name', 'patientid', 'starttime'])
dir_patcareseqs = '/cluster/work/grlab/clinical/hirid2/pg_db_export/p_patcareseqs/'
for file in glob(dir_patcareseqs + "*.parquet"):
    table = pd.read_parquet(file, columns=['endtime', 'name', 'patientid', 'starttime'], engine='pyarrow')
    table = table[table['name'].str.contains('blasenkatheter|cystofix|conduit|nephrostoma|foley', case=False) & np.logical_not(table['name'].str.contains('wunde|ex'))]
    catheter_table = catheter_table.append(table).reset_index(drop=True)


def get_catheter_times(pid):
    cath_pid = catheter_table[catheter_table['patientid'] == pid]
    return cath_pid


# for pid in np.unique(catheter_table['patientid']):
#     print(get_catheter_times(pid))
