''' 
Utility functions related to static data of the patients
'''

import ipdb

import numpy as np


def lookup_admission_category(pid, df_patient_full):
    ''' Returns the admission category for the static UMCDB data-frame'''
    df_patient=df_patient_full[df_patient_full["admissionid"]==pid]
    if not df_patient.shape[0]==1:
        assert(False)
    adm_cat=list(df_patient.admissionyeargroup)[0]
    return adm_cat


def lookup_admission_time(pid, df_patient_full):
    ''' Looks up a proxy to admission time for a PID'''
    df_patient=df_patient_full[df_patient_full["PatientID"]==pid]

    if not df_patient.shape[0]==1:
        return None
        
    adm_time=np.array(df_patient["AdmissionTime"])[0]
    return adm_time
