''' Label functions'''

import sys
import os
import os.path
import ipdb
import datetime
import timeit
import random
import gc
import psutil
import csv
import glob

import pandas as pd
import numpy as np
import scipy as sp

import mlhc_data_manager.util.filesystem as mlhc_fs
import mlhc_data_manager.util.array as mlhc_array
import mlhc_data_manager.util.io as mlhc_io

IMPUTE_GRID_PERIOD_SECS=300.0

def nan_exists_transition(event1_arr,event2_arr,event3_arr,
                          maybe1_arr,maybe2_arr,maybe3_arr,
                          not1_arr,not2_arr,not3_arr):
    ''' Given two NP arrays, returns element-wise existence of a 0->1 (event activation), for special
        case of first element of the vector, having some 1 will mean a transition.'''
    assert(event1_arr.size==event2_arr.size==event3_arr.size==maybe1_arr.size==maybe2_arr.size==maybe3_arr.size)
    assert(event1_arr.size>0)
    output_arr=np.zeros_like(event1_arr)

    # Define starting out in any events as being a deterioration from no observation.
    if maybe1_arr[0]==1.0 or maybe2_arr[0]==1.0 or maybe3_arr[0]==1.0:
        output_arr[0]=np.nan

    elif event1_arr[0]==1.0 or event2_arr[0]==1.0 or event3_arr[0]==1.0:
        output_arr[0]=1.0
    
    for idx in np.arange(1,event1_arr.size):

        # Transition into any of the events from a lower severity level. From the mutual exclusivity this condition can 
        # be simplified by checking if no down-wards transition took place.
        if maybe1_arr[idx-1]==1.0 or maybe1_arr[idx]==1.0 or maybe2_arr[idx-1]==1.0 or maybe2_arr[idx]==1.0 or maybe3_arr[idx-1]==1.0 or maybe3_arr[idx]==1.0:
            output_arr[idx]=np.nan

        elif event1_arr[idx-1]==0.0 and event1_arr[idx]==1.0 and event2_arr[idx-1]==0.0 and event3_arr[idx-1]==0.0 or \
             event2_arr[idx-1]==0.0 and event2_arr[idx]==1.0 and event3_arr[idx-1]==0.0 or \
             event3_arr[idx-1]==0.0 and event3_arr[idx]==1.0:
            output_arr[idx]=1.0

    return output_arr


def drug_stop(drug_arr, lhours=None, rhours=None):
    ''' Drug stop, transition from 1 to 0 in the binary drug array'''
    out_arr=np.zeros_like(drug_arr)
    for jdx in range(drug_arr.size):
        if drug_arr[jdx]==0 or np.isnan(drug_arr[jdx]):
            out_arr[jdx]=np.nan
            continue
        if drug_arr[jdx]==1:
            fut_win=drug_arr[min(drug_arr.size,jdx+lhours*12):min(drug_arr.size,jdx+rhours*12)]
            if (fut_win==0).any():
                out_arr[jdx]=1
    return out_arr

def patient_instability(event1_arr,event2_arr,event3_arr,maybe1_arr,maybe2_arr,maybe3_arr):
    '''Calculates patient instability, can use vectorized operations, not sure how those deal with NANs, so let's
       just be explicit...'''
    assert(event1_arr.size==event2_arr.size==event3_arr.size==maybe1_arr.size==maybe2_arr.size==maybe3_arr.size)
    assert(event1_arr.size>0)
    output_arr=np.zeros_like(event1_arr)

    for idx in np.arange(event1_arr.size):

        if maybe1_arr[idx]==1.0 or maybe2_arr[idx]==1.0 or maybe3_arr[idx]==1.0:
            output_arr[idx]=np.nan

        if event1_arr[idx]==1.0 or event2_arr[idx]==1.0 or event3_arr[idx]==1.0:
            output_arr[idx]=1.0

    return output_arr



def any_positive_transition(event_arr, lhours, rhours, grid_step_seconds):
    ''' Returns transformed arr that is positive at a location if in a future horizon some event is true in 
        input array (type can be co-erced to boolean)
    '''
    assert(rhours>=lhours)
    gridstep_per_hours=int(3600/grid_step_seconds)
    out_arr=np.zeros_like(event_arr)
    sz=event_arr.size

    for idx in range(event_arr.size):
        event_val=event_arr[idx]

        if np.isnan(event_val):
            out_arr[idx]=np.nan
            continue

            future_arr=event_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
            
            if future_arr.size==0:
                continue

            elif np.isnan(future_arr).all():
                out_arr[idx]=np.nan

            if event_val==0.0 and (future_arr==1.0).any():
                out_arr[idx]=1.0

    return out_arr


def time_to_worse_state_binned(endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    ''' Given that the patient transitions to a worse state in the array, creates a multi-class classification problem by binning the time to the
        earliest occurence of worse state in the future horizon'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    mins_per_gridstep=int(grid_step_secs/60)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size

    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]
        
        if np.isnan(e_val):
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        if future_arr.size==0:
            out_arr[idx]=-1.0
            continue

        elif np.isnan(future_arr).all():
            out_arr[idx]=np.nan
            continue

        if e_val==0.0:
            if (future_arr==1.0).any() or (future_arr==2.0).any() or (future_arr==3.0).any():
                min_idxs=[]
                
                if (future_arr==1.0).any():
                    min_idxs.append(np.where(future_arr==1.0)[0][0])

                if (future_arr==2.0).any():
                    min_idxs.append(np.where(future_arr==2.0)[0][0])

                if (future_arr==3.0).any():
                    min_idxs.append(np.where(future_arr==3.0)[0][0])
                
                time_to_det=mins_per_gridstep*np.min(min_idxs)
                quant_time_to_det=time_to_det-time_to_det%30
                out_arr[idx]=quant_time_to_det

        elif e_val==1.0: 
            if (future_arr==2.0).any() or (future_arr==3.0).any() :
                min_idxs=[]

                if (future_arr==2.0).any():
                    min_idxs.append(np.where(future_arr==2.0)[0][0])

                if (future_arr==3.0).any():
                    min_idxs.append(np.where(future_arr==3.0)[0][0])

                time_to_det=mins_per_gridstep*np.min(min_idxs)
                quant_time_to_det=time_to_det-time_to_det%30
                out_arr[idx]=quant_time_to_det

        elif e_val==2.0: 
            if (future_arr==3.0).any():
                out_arr[idx]=1.0
                time_to_det=mins_per_gridstep*np.where(future_arr==3.0)[0][0]
                quant_time_to_det=time_to_det-time_to_det%30
                out_arr[idx]=quant_time_to_det

        elif e_val==3.0:
            out_arr[idx]=np.nan

    return out_arr


def time_to_worse_state(endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    ''' Given that the patient transitions to a worse state in the array, creates a regression problem by binning the time to the 
        earliest occurence of worse state in the future horizon, assign a large sentinel value to negative instances.'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    mins_per_gridstep=int(grid_step_secs/60)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size

    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]
        
        if np.isnan(e_val):
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        if future_arr.size==0:
            out_arr[idx]=-1.0
            continue

        elif np.isnan(future_arr).all():
            out_arr[idx]=np.nan
            continue

        if e_val==0.0:
            if (future_arr==1.0).any() or (future_arr==2.0).any() or (future_arr==3.0).any():
                min_idxs=[]
                
                if (future_arr==1.0).any():
                    min_idxs.append(np.where(future_arr==1.0)[0][0])

                if (future_arr==2.0).any():
                    min_idxs.append(np.where(future_arr==2.0)[0][0])

                if (future_arr==3.0).any():
                    min_idxs.append(np.where(future_arr==3.0)[0][0])
                
                time_to_det=mins_per_gridstep*np.min(min_idxs)
                out_arr[idx]=time_to_det

        elif e_val==1.0: 
            if (future_arr==2.0).any() or (future_arr==3.0).any() :
                min_idxs=[]

                if (future_arr==2.0).any():
                    min_idxs.append(np.where(future_arr==2.0)[0][0])

                if (future_arr==3.0).any():
                    min_idxs.append(np.where(future_arr==3.0)[0][0])

                time_to_det=mins_per_gridstep*np.min(min_idxs)
                out_arr[idx]=time_to_det

        elif e_val==2.0: 
            if (future_arr==3.0).any():
                time_to_det=mins_per_gridstep*np.where(future_arr==3.0)[0][0]
                out_arr[idx]=time_to_det

        elif e_val==3.0:
            out_arr[idx]=np.nan

    return out_arr


def exists_stable_to_event1_transition(event1_arr,event2_arr,event3_arr):
    assert(event1_arr.size==event2_arr.size==event3_arr.size)
    assert(event1_arr.size>0)
    output_arr=np.zeros_like(event1_arr)

    if event1_arr[0]==1.0 and event2_arr[0]==0.0 or event3_arr[0]==0.0:
        output_arr[0]=1.0

    for idx in np.arange(1,event1_arr.size):
        if event1_arr[idx-1]==0.0 and event1_arr[idx]==1.0 and event2_arr[idx-1]==0.0 and event2_arr[idx]==0.0 \
           and event3_arr[idx-1]==0.0 and event3_arr[idx]==0.0:
            output_arr[idx]=1.0
    
    return output_arr




def shifted_exists_future_interval(label_in_arr,forward_lbound,forward_rbound,invert_label=False):
    '''Given input label array, returns existence in a time interval ahead
       The step size on the input grid is 5 minutes.'''
    pos_label=0.0 if invert_label else 1.0
    gridstep_per_hours=int(3600/IMPUTE_GRID_PERIOD_SECS)
    output_arr=np.zeros_like(label_in_arr)

    for idx in np.arange(label_in_arr.size):

        full_sz=label_in_arr.size

        if forward_lbound==0:
            lwindow_idx=idx+1
        else:
            lwindow_idx=idx+int(forward_lbound*gridstep_per_hours)
        rwindow_idx=idx+int(forward_rbound*gridstep_per_hours)

        if lwindow_idx < full_sz:
            output_arr[idx]=1.0 if (label_in_arr[lwindow_idx:min(full_sz,rwindow_idx)]==pos_label).any() else 0.0
        else:
            output_arr[idx]=np.nan
    
    return output_arr




def time_to_event(label_in_arr,forward_rbound,invert_label=False):
    '''Given input label array computes the time to the first event in the interval ahead, up to maximum
       time into the future'''
    pos_label=0.0 if invert_label else 1.0
    gridstep_per_hours=12
    output_arr=np.zeros_like(label_in_arr)

    for idx in np.arange(label_in_arr.size):
        full_sz=label_in_arr.size
        lwindow_idx=idx+1
        rwindow_idx=idx+forward_rbound*gridstep_per_hours
        
        if lwindow_idx < full_sz:
            tent_event_arr=label_in_arr[lwindow_idx:min(full_sz,rwindow_idx)]
            event_idxs=np.argwhere(tent_event_arr==pos_label)
            if event_idxs.size==0:
                output_label=-1.0
            else:
                output_label=(event_idxs.min()+1)*5.0
            output_arr[idx]=output_label
        else:
            output_arr[idx]=np.nan
    
    return output_arr






def future_deterioration(event1_arr, event2_arr, event3_arr, maybe1_arr, maybe2_arr, maybe3_arr, 
                         pn1_arr, pn2_arr, pn3_arr, l_hours, r_hours, grid_step_secs):
    ''' Computes existence of guarded deterioration in a future window, which means that the patient did 
        not deteriorate during the lead time [0,l] and then deteriorated in the window [l,r], i.e. has a 
        worse state in this window'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros_like(event1_arr)
    sz=event1_arr.size
    
    for idx in range(event1_arr.size):
        e1_val=event1_arr[idx]
        e2_val=event2_arr[idx]
        e3_val=event3_arr[idx]
        m1_val=maybe1_arr[idx]
        m2_val=maybe2_arr[idx]
        m3_val=maybe3_arr[idx]

        # We cannot determine in which state we started off with...
        if np.isnan(e1_val) or np.isnan(e2_val) or np.isnan(e3_val) or m1_val==1.0 or m2_val==1.0 or m3_val==1.0:
            out_arr[idx]=np.nan
            continue

        lead1_arr=event1_arr[idx: min(sz, idx+gridstep_per_hours*l_hours)]
        lead2_arr=event2_arr[idx: min(sz, idx+gridstep_per_hours*l_hours)]
        lead3_arr=event3_arr[idx: min(sz, idx+gridstep_per_hours*l_hours)]

        future1_arr=event1_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]
        future2_arr=event2_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]
        future3_arr=event3_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]

        # No future to consider, => no deterioration
        if future1_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future1_arr).all() or np.isnan(future2_arr).all() or np.isnan(future3_arr).all() or \
             np.isnan(lead1_arr).all() or np.isnan(lead2_arr).all() or np.isnan(lead3_arr).all():
            out_arr[idx]=np.nan
            continue

        # State 0: Stability
        if e1_val==0.0 and e2_val==0.0 and e3_val==0.0:
            if ((future1_arr==1.0).any() or (future2_arr==1.0).any() or (future3_arr==1.0).any()) \
               and not (lead1_arr==1.0).any() and not (lead2_arr==1.0).any() and not (lead3_arr==1.0).any():
                out_arr[idx]=1.0

        # State 1: Low severity patient state
        elif e1_val==1.0: 
            if ((future2_arr==1.0).any() or (future3_arr==1.0).any()) \
               and not (lead2_arr==1.0).any() and not (lead3_arr==1.0).any():
                out_arr[idx]=1.0
            
        # State 2: Intermediate severity patient state
        elif e2_val==1.0: 
            if (future3_arr==1.0).any() and not (lead3_arr==1.0).any():
                out_arr[idx]=1.0


    return out_arr



def future_worse_state( endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    ''' Computes existence of deterioration in future window directly from the endpoint series, this does not actually 
        check for transitions but existence of the event in a future, so the actual deterioration could have happened
        before the future window'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        # We cannot determine in which state we started off with...
        if np.isnan(e_val):
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future_arr).all():
            out_arr[idx]=np.nan
            continue

        # State 0: Stability
        if e_val==0.0:
            if (future_arr==1.0).any() or (future_arr==2.0).any() or (future_arr==3.0).any():
                out_arr[idx]=1.0

        # State 1: Low severity patient state
        elif e_val==1.0: 
            if (future_arr==2.0).any() or (future_arr==3.0).any() :
                out_arr[idx]=1.0
            
        # State 2: Intermediate severity patient state
        elif e_val==2.0: 
            if (future_arr==3.0).any():
                out_arr[idx]=1.0

        # State 3: No deterioration from this level is possible, so we will not use these segments
        elif e_val==3.0:
            out_arr[idx]=np.nan

    return out_arr


def future_worse_state_soft( endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    ''' Computes existence of deterioration in future window directly from the endpoint series, this does not actually 
        check for transitions but existence of the event in a future, so the actual deterioration could have happened
        before the future window'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        # We cannot determine in which state we started off with...
        if e_val=="unknown" or "maybe" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        # State 0: Stability
        if e_val=="event 0":
            if (future_arr=="event 1").any() or (future_arr=="event 2").any() or (future_arr=="event 3").any() \
               or (future_arr=="maybe 1").any() or (future_arr=="maybe 2").any() or (future_arr=="maybe 3").any() \
               or (future_arr=="probably not 1").any() or (future_arr=="probably not 2").any() or (future_arr=="probably not 3").any():
                out_arr[idx]=1.0

        # State 0.5 Intermediate state
        elif e_val in ["probably not 1", "probably not 2", "probably not 3"]:
            if (future_arr=="event 1").any() or (future_arr=="event 2").any() or (future_arr=="event 3").any():
                out_arr[idx]=1.0

        # State 1: Low severity patient state
        elif e_val=="event 1": 
            if (future_arr=="event 2").any() or (future_arr=="event 3").any():
                out_arr[idx]=1.0
            
        # State 2: Intermediate severity patient state
        elif e_val=="event 2": 
            if (future_arr=="event 3").any():
                out_arr[idx]=1.0

        # State 3: No deterioration from this level is possible, so we will not use these segments
        elif e_val=="event 3":
            out_arr[idx]=np.nan

    return out_arr

def future_worse_state_from_0_TRAIN(geq1_arr,drug_arr,l_hours,r_hours,grid_step_secs):
    ''' Main label function, predict deterioration from state 0 (stability), exclude uncertain 
        positive labels where until the onset of the failure, a kidney-harming drug is newly 
        introduced, or uncertain negative labels, where a kidney-saving action is taken during
        the horizon.'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(geq1_arr.size)
    sz=geq1_arr.size
    
    for idx in range(sz):
        e_val=geq1_arr[idx]
        d_val=drug_arr[idx]
        if np.isnan(e_val) or e_val==1:
            out_arr[idx]=np.nan
            continue
        future_arr=geq1_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        future_drug_arr=drug_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue
        
        # Future has only invalid states
        elif np.sum(np.isfinite(future_arr))==0 or np.sum(np.isfinite(future_drug_arr))==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ0 to GEQ1 state on KDIGO or transition potentially masked by drug stop.
        if e_val==0.0:
            if (future_arr==1.0).any():
                onset_idx=np.where(future_arr==1)[0].min()
                if d_val==0 and (future_drug_arr[:onset_idx]==1).any():
                    out_arr[idx]=np.nan
                else:
                    out_arr[idx]=1.0
                    
            else:
                # Uncertain negative label, (stopping of a kidney harming drug)
                if d_val==1 and (future_drug_arr==0).any():
                    out_arr[idx]=np.nan
                
    return out_arr


def future_worse_state_from_0_EVAL(geq1_arr,l_hours,r_hours,grid_step_secs):
    ''' Main label function, predict deterioration from state 0 (stability)'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(geq1_arr.size)
    sz=geq1_arr.size
    
    for idx in range(sz):
        e_val=geq1_arr[idx]
        if np.isnan(e_val) or e_val==1:
            out_arr[idx]=np.nan
            continue
        future_arr=geq1_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue
        
        # Future has only invalid states
        elif np.sum(np.isfinite(future_arr))==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ0 to GEQ1 state on KDIGO.
        if e_val==0.0:
            if (future_arr==1.0).any():
                out_arr[idx]=1.0
                
    return out_arr


def future_worse_state_from_0_FIXED(geq1_arr,l_hours,r_hours,grid_step_secs,fixed_hour, onwards=False):
    ''' Main label function, predict deterioration from state 0 (stability), only define the label 
    at particular time-points after the beginning of the stay. The special argument
    <augmented> enables label augmentation so the check for the correct part of the stay is soft.
    The parameter <onwards> will define a final bin'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(geq1_arr.size)
    sz=geq1_arr.size
    
    for idx in range(sz):

        # Onwards use the model for all time-steps including and after
        if onwards:
            if idx<gridstep_per_hours*fixed_hour:
                out_arr[idx]=np.nan
                continue

        # Use the model only in that particular hour
        else:
            if idx<gridstep_per_hours*fixed_hour or idx>=gridstep_per_hours*(fixed_hour+1):
                out_arr[idx]=np.nan
                continue
        
        e_val=geq1_arr[idx]

        # Patient is not stable or current state unknown
        if np.isnan(e_val) or e_val==1:
            out_arr[idx]=np.nan
            continue
        
        future_arr=geq1_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only invalid states
        elif np.isfinite(future_arr).sum()==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ0 to GEQ1 state on KDIGO.
        if e_val==0.0:
            if (future_arr==1.0).any():
                out_arr[idx]=1.0
                
    return out_arr


def future_worse_state_urine_from_0_FIXED(urine_geq1_arr, l_hours, r_hours, grid_step_secs, fixed_hour):
    ''' Main label function, predict deterioration from state 0 (stability), only define the label 
    at particular time-points after the beginning of the stay. For urine tasks.'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(urine_geq1_arr.size)
    sz=urine_geq1_arr.size
    
    for idx in range(sz):

        # Not a valid point to apply the model
        if not idx==gridstep_per_hours*fixed_hour:
            out_arr[idx]=np.nan
            continue
        
        e_val=urine_geq1_arr[idx]

        # Patient is not stable or current state unknown
        if np.isnan(e_val) or e_val==1:
            out_arr[idx]=np.nan
            continue
        
        future_arr=urine_geq1_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only invalid states
        elif np.isfinite(future_arr).sum()==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ0 to GEQ1 state on KDIGO urine.
        if e_val==0.0:
            if (future_arr==1.0).any():
                out_arr[idx]=1.0
                
    return out_arr


def future_worse_state_urine_from_0(urine_geq1_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, predict deterioration from state 0 (stability), only define the label 
    at particular time-points after the beginning of the stay. For urine tasks.'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(urine_geq1_arr.size)
    sz=urine_geq1_arr.size
    
    for idx in range(sz):

        e_val=urine_geq1_arr[idx]

        # Patient is not stable or current state unknown
        if np.isnan(e_val) or e_val==1:
            out_arr[idx]=np.nan
            continue
        
        future_arr=urine_geq1_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only invalid states
        elif np.isfinite(future_arr).sum()==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ0 to GEQ1 state on KDIGO urine.
        if e_val==0.0:
            if (future_arr==1.0).any():
                out_arr[idx]=1.0
                
    return out_arr

def future_worse_state_urine_from_1(urine_geq1_arr, urine_geq2_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, predict deterioration from state 0,1 (stability), only define the label 
    at particular time-points after the beginning of the stay. For urine tasks.'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(urine_geq2_arr.size)
    sz=urine_geq2_arr.size
    
    for idx in range(sz):

        r_val=urine_geq1_arr[idx]
        e_val=urine_geq2_arr[idx]

        # Patient is not stable or current state unknown
        if np.isnan(e_val) or np.isnan(r_val) or e_val==1 or r_val==0:
            out_arr[idx]=np.nan
            continue
        
        future_arr=urine_geq2_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only invalid states
        elif np.isfinite(future_arr).sum()==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ1 to GEQ2 state on KDIGO urine.
        if e_val==0.0 and r_val==1 and (future_arr==1.0).any():
            out_arr[idx]=1.0
                
    return out_arr


def future_worse_state_urine_from_2(urine_geq2_arr, urine_geq3_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, predict deterioration from state 0,1 (stability), only define the label 
    at particular time-points after the beginning of the stay. For urine tasks.'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(urine_geq3_arr.size)
    sz=urine_geq3_arr.size
    
    for idx in range(sz):

        r_val=urine_geq2_arr[idx]
        e_val=urine_geq3_arr[idx]

        # Patient is not stable or current state unknown
        if np.isnan(e_val) or np.isnan(r_val) or e_val==1 or r_val==0:
            out_arr[idx]=np.nan
            continue
        
        future_arr=urine_geq3_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only invalid states
        elif np.isfinite(future_arr).sum()==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ1 to GEQ2 state on KDIGO urine.
        if e_val==0.0 and r_val==1.0 and (future_arr==1.0).any():
            out_arr[idx]=1.0
                
    return out_arr



def future_worse_state_creat_from_0_FIXED(creat_geq1_arr, l_hours, r_hours, grid_step_secs, fixed_hour):
    ''' Main label function, predict deterioration from state 0 (stability), only define the label 
    at particular time-points after the beginning of the stay. For creatinine tasks.'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(creat_geq1_arr.size)
    sz=creat_geq1_arr.size
    
    for idx in range(sz):

        # Not a valid point to apply the model
        if not idx==gridstep_per_hours*fixed_hour:
            out_arr[idx]=np.nan
            continue
        
        e_val=creat_geq1_arr[idx]

        # Patient is not stable or current state unknown
        if np.isnan(e_val) or e_val==1:
            out_arr[idx]=np.nan
            continue
        
        future_arr=creat_geq1_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only invalid states
        elif np.isfinite(future_arr).sum()==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ0 to GEQ1 state on KDIGO creatinine.
        if e_val==0.0:
            if (future_arr==1.0).any():
                out_arr[idx]=1.0
                
    return out_arr


def future_worse_state_creat_from_0(creat_geq1_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, predict deterioration from state 0 (stability), only define the label 
    at particular time-points after the beginning of the stay. For creatinine tasks.'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(creat_geq1_arr.size)
    sz=creat_geq1_arr.size
    
    for idx in range(sz):
        
        e_val=creat_geq1_arr[idx]

        # Patient is not stable or current state unknown
        if np.isnan(e_val) or e_val==1:
            out_arr[idx]=np.nan
            continue
        
        future_arr=creat_geq1_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only invalid states
        elif np.isfinite(future_arr).sum()==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ0 to GEQ1 state on KDIGO creatinine.
        if e_val==0.0:
            if (future_arr==1.0).any():
                out_arr[idx]=1.0
                
    return out_arr


def future_worse_state_creat_from_1(creat_geq1_arr, creat_geq2_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, predict deterioration from state 0,1 (stability), only define the label 
    at particular time-points after the beginning of the stay. For creatinine tasks.'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(creat_geq2_arr.size)
    sz=creat_geq2_arr.size
    
    for idx in range(sz):

        r_val=creat_geq1_arr[idx]
        e_val=creat_geq2_arr[idx]

        # Patient is not stable or current state unknown
        if np.isnan(e_val) or np.isnan(r_val) or e_val==1 or r_val==0:
            out_arr[idx]=np.nan
            continue
        
        future_arr=creat_geq2_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only invalid states
        elif np.isfinite(future_arr).sum()==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ1 to GEQ2 state on KDIGO creatinine.
        if e_val==0.0 and r_val==1.0 and (future_arr==1.0).any():
            out_arr[idx]=1.0
                
    return out_arr


def future_worse_state_creat_from_2(creat_geq2_arr, creat_geq3_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, predict deterioration from state 0,1 (stability), only define the label 
    at particular time-points after the beginning of the stay. For creatinine tasks.'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(creat_geq3_arr.size)
    sz=creat_geq3_arr.size
    
    for idx in range(sz):

        r_val=creat_geq2_arr[idx]
        e_val=creat_geq3_arr[idx]

        # Patient is not stable or current state unknown
        if np.isnan(e_val) or np.isnan(r_val) or e_val==1 or r_val==0:
            out_arr[idx]=np.nan
            continue
        
        future_arr=creat_geq3_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only invalid states
        elif np.isfinite(future_arr).sum()==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ1 to GEQ2 state on KDIGO creatinine.
        if e_val==0.0 and r_val==1.0 and (future_arr==1.0).any():
            out_arr[idx]=1.0
                
    return out_arr




def future_worse_state_from_0_BEFORE(geq1_arr, l_hours, r_hours, grid_step_secs, before_hour):
    ''' Main label function, predict deterioration from state 0 (stability), only define the label at particular
        time-points before a certain hour into the stay'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(geq1_arr.size)
    sz=geq1_arr.size
    
    for idx in range(sz):

        # Not a valid point to apply the model
        if idx>gridstep_per_hours*before_hour:
            out_arr[idx]=np.nan
            continue        
        
        e_val=geq1_arr[idx]

        # Currently patient is not stable or unknown
        if np.isnan(e_val) or e_val==1:
            out_arr[idx]=np.nan
            continue
        
        future_arr=geq1_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue
        
        elif np.isfinite(future_arr).sum()==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ0 to GEQ1 state on KDIGO.
        if e_val==0.0:
            if (future_arr==1.0).any():
                out_arr[idx]=1.0
                
    return out_arr



def future_worse_state_from_0_aux(ep_status_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, using aux variant, to level 1'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros_like(ep_status_arr)
    sz=ep_status_arr.size
    
    for idx in range(sz):
        e_val=ep_status_arr[idx]
        if np.isnan(e_val) or e_val>=1.0:
            out_arr[idx]=np.nan
            continue
        future_arr=ep_status_arr[min(sz, 1+idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue
        
        # Future has only invalid states
        elif np.sum(np.isfinite(future_arr))==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from 0 to GEQ1 state in KDIGO level
        if e_val==0.0:
            if (future_arr>=1.0).any():
                out_arr[idx]=1.0
                
    return out_arr


def future_worse_state_from_0_1_aux(ep_status_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, using aux variant, to level 2'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros_like(ep_status_arr)
    sz=ep_status_arr.size
    
    for idx in range(sz):
        e_val=ep_status_arr[idx]
        if np.isnan(e_val) or e_val>=2.0:
            out_arr[idx]=np.nan
            continue
        future_arr=ep_status_arr[min(sz, 1+idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue
        
        # Future has only invalid states
        elif np.sum(np.isfinite(future_arr))==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ1 to GEQ2 state in KDIGO level
        if e_val<=1.0:
            if (future_arr>=2.0).any():
                out_arr[idx]=1.0
                
    return out_arr


def future_worse_state_from_0_1_2_aux(ep_status_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, using aux variant. to level 3'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros_like(ep_status_arr)
    sz=ep_status_arr.size
    
    for idx in range(sz):
        e_val=ep_status_arr[idx]
        if np.isnan(e_val) or e_val>=3.0:
            out_arr[idx]=np.nan
            continue
        future_arr=ep_status_arr[min(sz, 1+idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue
        
        # Future has only invalid states
        elif np.sum(np.isfinite(future_arr))==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ2 to GEQ3 state in KDIGO level
        if e_val<=2.0:
            if (future_arr>=3.0).any():
                out_arr[idx]=1.0
                
    return out_arr


def future_worse_state_from_1_EVAL(geq1_arr, geq2_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, predict deterioration from state 1'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(geq2_arr.size)
    sz=geq2_arr.size
    
    for idx in range(sz):
        r_val=geq1_arr[idx]
        e_val=geq2_arr[idx]
        if np.isnan(e_val) or np.isnan(r_val) or e_val==1 or r_val==0:
            out_arr[idx]=np.nan
            continue
        
        future_arr=geq2_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue
        
        elif np.sum(np.isfinite(future_arr))==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from GEQ1 to GEQ2 state on KDIGO.
        if r_val==1 and e_val==0 and (future_arr==1.0).any():
            out_arr[idx]=1.0
                
    return out_arr

def future_worse_state_from_1_TRAIN(geq1_arr, geq2_arr, drug_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, predict deterioration from state 1, taking into account kidney-harming
        drugs to mark label uncertainty.'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(geq2_arr.size)
    sz=geq2_arr.size
    
    for idx in range(sz):
        r_val=geq1_arr[idx]
        e_val=geq2_arr[idx]
        d_val=drug_arr[idx]
        
        if np.isnan(e_val) or np.isnan(r_val) or e_val==1 or r_val==0:
            out_arr[idx]=np.nan
            continue
        
        future_arr=geq2_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        future_drug_arr=drug_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue
        
        elif np.sum(np.isfinite(future_arr))==0 or np.sum(np.isfinite(future_drug_arr))==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from GEQ1 to GEQ2 state on KDIGO (taking into account label uncertainty from drugs)
        if r_val==1 and e_val==0:

            if (future_arr==1.0).any():
                onset_idx=np.where(future_arr==1)[0].min()
                if d_val==0 and (future_drug_arr[:onset_idx]==1).any():
                    out_arr[idx]=np.nan
                else:
                    out_arr[idx]=1.0

            else:
                if d_val==1 and (future_drug_arr==0).any():
                    out_arr[idx]=np.nan
                
    return out_arr

def future_worse_state_from_2_EVAL(geq2_arr, geq3_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, predict deterioration from state 2'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(geq3_arr.size)
    sz=geq3_arr.size
    
    for idx in range(sz):
        r_val=geq2_arr[idx]
        e_val=geq3_arr[idx]
        
        if np.isnan(e_val) or np.isnan(r_val) or e_val==1 or r_val==0:
            out_arr[idx]=np.nan
            continue
        
        future_arr=geq3_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue
        
        elif np.sum(np.isfinite(future_arr))==0:
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ3 to GEQ3 state on KDIGO.
        if r_val==1 and e_val==0 and (future_arr==1.0).any():
            out_arr[idx]=1.0
                
    return out_arr


def future_worse_state_from_2_TRAIN(geq2_arr, geq3_arr, drug_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, predict deterioration from state 2, taking into account kidney harming
        drugs'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(geq3_arr.size)
    sz=geq3_arr.size
    
    for idx in range(sz):
        r_val=geq2_arr[idx]
        e_val=geq3_arr[idx]
        d_val=drug_arr[idx]
        
        if np.isnan(e_val) or np.isnan(r_val) or e_val==1 or r_val==0:
            out_arr[idx]=np.nan
            continue
        
        future_arr=geq3_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        future_drug_arr=drug_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        
        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue
        
        elif np.sum(np.isfinite(future_arr))==0 or np.sum(np.isfinite(future_drug_arr)==0):
            out_arr[idx]=np.nan
            continue
        
        # Transition from LEQ3 to GEQ3 state on KDIGO.
        if r_val==1 and e_val==0:

            if (future_arr==1.0).any():
                onset_idx=np.where(future_arr==1)[0].min()
                if d_val==0 and (future_drug_arr[:onset_idx]==1).any():
                    out_arr[idx]=np.nan
                else:
                    out_arr[idx]=1.0

            else:
                if d_val==1 and (future_drug_arr==0).any():
                    out_arr[idx]=np.nan
                
    return out_arr


def future_worse_state_from_0_1_unk( endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    ''' Main label function, predict deterioration from state 0 (stability) or state 1'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]
        if np.isnan(e_val) or e_val>=2:
            out_arr[idx]=np.nan
            continue
        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
        if future_arr.size==0:
            continue
        elif np.isnan(future_arr).all():
            out_arr[idx]=np.nan
            continue
        if e_val in [0.0,-1.0]:
            if (future_arr==1.0).any() or (future_arr==2.0).any() or (future_arr==3.0).any():
                out_arr[idx]=1.0
        elif e_val==1.0:
            if (future_arr==2.0).any() or (future_arr==3.0).any():
                out_arr[idx]=1.0

    return out_arr


def future_worse_state_soft_from_0( endpoint_status_arr, l_hours, r_hours, grid_step_secs):

    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        if e_val in ["unknown","event 1", "event 2", "event 3"] or "maybe" in e_val or "probably not" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        # State 0: Stability
        if e_val=="event 0":
            if (future_arr=="event 1").any() or (future_arr=="event 2").any() or (future_arr=="event 3").any() \
               or (future_arr=="maybe 1").any() or (future_arr=="maybe 2").any() or (future_arr=="maybe 3").any() \
               or (future_arr=="probably not 1").any() or (future_arr=="probably not 2").any() or (future_arr=="probably not 3").any():
                out_arr[idx]=1.0

    return out_arr



def future_worse_state_from_pn(endpoint_status_arr, l_hours, r_hours, grid_step_secs):

    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        if e_val in ["unknown","event 0","event 1", "event 2", "event 3"] or "maybe" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        # Probably not pre-state
        if e_val in ["probably not 1", "probably not 2", "probably not 3"]:
            if (future_arr=="event 1").any() or (future_arr=="event 2").any() or (future_arr=="event 3").any():
                out_arr[idx]=1.0

    return out_arr



def future_worse_state_from_1_or_2(endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        if e_val in ["unknown","event 0","event 3"] or "maybe" in e_val or "probably not" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        # Event 1 state
        if e_val=="event 1":
            if (future_arr=="event 2").any() or (future_arr=="event 3").any():
                out_arr[idx]=1.0

        if e_val=="event 2":
            if (future_arr=="event 3").any():
                out_arr[idx]=1.0

    return out_arr



def exists_stability_to_any(event1_arr, event2_arr, event3_arr, maybe1_arr, maybe2_arr, maybe3_arr, l_hours, r_hours):
    ''' Stability to any event in {1,2,3}'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/IMPUTE_GRID_PERIOD_SECS)
    out_arr=np.zeros_like(event1_arr)
    sz=event1_arr.size
    
    for idx in range(event1_arr.size):
        e1_val=event1_arr[idx]
        e2_val=event2_arr[idx]
        e3_val=event3_arr[idx]
        m1_val=maybe1_arr[idx]
        m2_val=maybe2_arr[idx]
        m3_val=maybe3_arr[idx]

        # We cannot determine in which state we started off with, or patient is currently not stable
        if np.isnan(e1_val) or np.isnan(e2_val) or np.isnan(e3_val) or m1_val==1.0 or m2_val==1.0 or m3_val==1.0 or e1_val==1.0 or e2_val==1.0 or e3_val==1.0:
            out_arr[idx]=np.nan
            continue

        future1_arr=event1_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]
        future2_arr=event2_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]
        future3_arr=event3_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]

        # No future to consider, => no deterioration
        if future1_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future1_arr).all() or np.isnan(future2_arr).all() or np.isnan(future3_arr).all():
            out_arr[idx]=np.nan
            continue

        if (future1_arr==1.0).any() or (future2_arr==1.0).any() or (future3_arr==1.0).any():
            out_arr[idx]=1.0

    return out_arr
    


def exists_stability_to_1(event1_arr, event2_arr, event3_arr, maybe1_arr, maybe2_arr, maybe3_arr, l_hours, r_hours):
    ''' Stability to any event in {1}'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/IMPUTE_GRID_PERIOD_SECS)
    out_arr=np.zeros_like(event1_arr)
    sz=event1_arr.size
    
    for idx in range(event1_arr.size):
        e1_val=event1_arr[idx]
        e2_val=event2_arr[idx]
        e3_val=event3_arr[idx]
        m1_val=maybe1_arr[idx]
        m2_val=maybe2_arr[idx]
        m3_val=maybe3_arr[idx]

        # We cannot determine in which state we started off with or patient is currently not stable
        if np.isnan(e1_val) or np.isnan(e2_val) or np.isnan(e3_val) or m1_val==1.0 or m2_val==1.0 or m3_val==1.0 or e1_val==1.0 or e2_val==1.0 or e3_val==1.0:
            out_arr[idx]=np.nan
            continue

        future1_arr=event1_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]

        # No future to consider, => no deterioration
        if future1_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future1_arr).all():
            out_arr[idx]=np.nan
            continue

        if (future1_arr==1.0).any():
            out_arr[idx]=1.0

    return out_arr







