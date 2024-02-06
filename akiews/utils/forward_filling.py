'''
Imputation routines for different variable types in the ICU Bern data-set, we distinguish between the
cases 'event' (endpoint), measured variable and pharma variable which have to be treated differently.
'''

import os
import os.path
import sys
import ipdb

import numpy as np


def value_empty(size, default_val, dtype=None):
    ''' Returns a vector filled with elements of a specific value'''

    if dtype is not None:
        tmp_arr=np.empty(size, dtype=dtype)
    else:
        tmp_arr=np.empty(size)

    tmp_arr[:]=default_val
    return tmp_arr



def empty_nan(sz):
    ''' Returns an empty NAN vector of specified size'''
    arr=np.empty(sz)
    arr[:]=np.nan
    return arr



def impute_forward_fill_binary(raw_ts_1,raw_values_1,raw_ts_2,raw_values_2,timegrid_pred):
    ''' Fill in a time series vector to a given time grid given two separate time-series whose time-stamps are to be 
        interleaved. The logic is still that the last observed observation will be filled in to a new location on 
        the time-grid'''
    raw_ts=np.concatenate((raw_ts_1,raw_ts_2))
    raw_values=np.concatenate((raw_values_1,raw_values_2))
    sort_idx=np.argsort(raw_ts)
    return impute_forward_fill(raw_ts[sort_idx],raw_values[sort_idx],timegrid_pred)


def impute_forward_fill_event(raw_ts, raw_values, timegrid_pred):
    ''' Fill in a time series vector by forward filling from a vector with event begin-end
        encoding (like the endpoints)'''
    pred_values=np.zeros_like(timegrid_pred)
    input_ts=0
    grid_period=bern_meta.impute_grid_period_secs

    # Fill in each point of the time-grid.
    for idx,ts in np.ndenumerate(timegrid_pred):
        
        while input_ts<raw_ts.size and raw_ts[input_ts]<=ts:
            input_ts+=1
        # POSTCONDITION: input_ts=raw_ts.size OR raw_ts[input_ts]>ts : Past the time grid point
        
        # No value has been observed before the current time-grid point. This just means that no event takes
        # place.
        if input_ts==0:
            pred_values[idx[0]]=0.0
            continue

        last_event_switch=raw_values[input_ts-1]

        ## Special case that the switch falls into the first part of the interval, declare this
        #  as the event switch instead of the last before interval left bound.
        if input_ts<raw_ts.size and raw_ts[input_ts]-ts<=0.5*grid_period:
            last_event_switch=raw_values[input_ts]

        ## Check the type of the last event switch, either we are inside or outside the event
        if last_event_switch==1.0:
            pred_values[idx[0]]=1.0
        elif last_event_switch==-1.0:
            pred_values[idx[0]]=0.0
        else:
            print("ERROR: Invalid event encoding, exiting...")
            sys.exit(1)

    return pred_values





def impute_forward_fill_pharma(raw_ts,raw_values,timegrid_pred, use_inj_table, inj_ts=None, inj_vals=None):
    '''Fill in the pharma vector by recomputing the doses to the time grid. For the special case of an injection
       it will be attributed to the next minute for simplicity.'''
    grid_period=bern_meta.impute_grid_period_secs
    pred_values=np.zeros_like(timegrid_pred)
    input_ts=0

    ## Advance beyond duplicate time-stamps at the beginning of series

    # Fill in each point of the time-grid.
    for idx,ts in np.ndenumerate(timegrid_pred):

        if idx==0:
            prev_ts=0
            while raw_ts[prev_ts]<ts:
                prev_ts+=1
            prev_ts-=1
        else:
            prev_ts=input_ts-1
            
        while input_ts<raw_ts.size and raw_ts[input_ts]<=ts+grid_period:
            input_ts+=1
        # POSTCONDITION: input_ts=raw_ts.size OR raw_ts[input_ts]>ts : Past the time grid point
        
        # No dose has been given up to here, or we are at the end-of-time and we assume that 
        # the given dose is 0.0
        if input_ts==0 or input_ts>=raw_ts.size:
            pred_values[idx[0]]=0.0
            continue

        # MIDCONDITION: <last_pharma_ts> and <after_pharma_ts> are well defined
        after_pharma_ts=raw_ts[input_ts]
        assert(input_ts>prev_ts)
        
        if not prev_ts==-1:
            last_pharma_ts=raw_ts[prev_ts]
            assert(after_pharma_ts>last_pharma_ts)            

        ## Simple case, prev_ts is before the grid interval, input_ts after the grid interval
        if input_ts-prev_ts==1:
            abs_given_dose=raw_values[input_ts]

            # Special case, decompose sum from injection table
            if use_inj_table:
                check_ts=raw_ts[input_ts]
                check_arr=inj_ts==check_ts
                if check_arr.any():
                    inj_value=inj_vals[check_arr][0]

                    ## If this does not hold, it could not have been summed into.
                    if abs_given_dose>=inj_value:
                        abs_given_dose-=inj_value
            
            time_diff=after_pharma_ts-last_pharma_ts
            assert(time_diff>=grid_period)
            eqv_dose=grid_period/time_diff*abs_given_dose

        ## Some time-stamp intervals are interior to the grid period
        else:
            abs_given_dose_initial=raw_values[prev_ts+1]
            initial_ts=raw_ts[prev_ts+1]

            ## There has been a previous time-stamp before the grid
            if not prev_ts==-1:
                assert(initial_ts>last_pharma_ts)
                assert(initial_ts>=ts)
                time_diff_initial=initial_ts-last_pharma_ts
                time_diff_in_grid=initial_ts-ts
                assert(time_diff_in_grid<=grid_period)
                eqv_dose=time_diff_in_grid/time_diff_initial*abs_given_dose_initial
            else:
                eqv_dose=abs_given_dose_initial

            ## Interior time stamps that should be completely attributed to the grid period, prevent
            #  duplicates from being added twice.
            for interior_ts in np.arange(prev_ts+2,input_ts):
                if raw_ts[interior_ts]>raw_ts[interior_ts-1]:
                    eqv_dose+=raw_values[interior_ts]

            abs_given_dose_final=raw_values[input_ts]

            ## Special case, decompose sum on injection table
            if use_inj_table:
                check_ts=raw_ts[input_ts]
                check_arr=inj_ts==check_ts
                if check_arr.any():
                    inj_value=inj_vals[check_arr][0]

                    ## If this does not hold, then the injection was not originally summed into the value.
                    if abs_given_dose_final>=inj_value:
                        abs_given_dose_final-=inj_value
                    else:
                        print("WARNING: Injection dose anomaly")
                        print("Full dose: {}".format(abs_given_dose_final))
                        print("Injection value: {}".format(inj_value))
            
            final_ts=raw_ts[input_ts-1]
            assert(after_pharma_ts>final_ts)
            assert(after_pharma_ts>ts+grid_period)
            assert(ts+grid_period>=final_ts)
            time_diff_final=after_pharma_ts-final_ts
            time_diff_in_grid=ts+grid_period-final_ts
            eqv_dose+=time_diff_in_grid/time_diff_final*abs_given_dose_final

        pred_values[idx[0]]=eqv_dose

    return pred_values


def impute_forward_fill_complex(raw_ts, raw_values, timegrid_pred, global_mean, grid_period, fill_interval_secs=None,
                                rolling_mean_secs=None, return_mean_secs=None, var_lower=None, var_upper=None, 
                                var_type=None, var_encoding=None, variable_id=None, weight_imputed_col=None, static_height=None,
                                personal_bmi=None):
    ''' Adaptive version of forward filling, which has specific forward filling thresholds as a function of the 
        median sampling rate of the variable, if it cannot forward fill it will return to the rolling mean of the last
        observation over a time which also is a function of the median sampling rate. 

        ARGUMENTS:

        grid_period: The period of the time grid that we are going to create.
  
        fill_interval_secs: The variable-specific interval in seconds up to which it is fine to forward 
                            fill the last observation of a variable

        rolling_mean_secs: The size of the rolling mean window that we use to decide which level to return 
                           to over the extrapolation span

        return_mean_secs: The number of seconds over which we will return to the rolling mean.

        var_lower: A lower threshold of physiological value, discard observations outside of the range

        var_higher: An upper threshold of physiological value, discard observations outside of the range
    
        var_type \in {non_pharma, pharma}: Fundamental variable type

        var_encoding \in {continuous, binary, categorical, ordinal}: Variable data type/encoding

        variable_id: Allow to pass the variable ID to treat some special cases of imputation

        weight_imputed_col: Allow to pass the imputed weight column so we can perform dynamic imputation using formulae
 
        static_height: Pass the static height of the patient so we can perform imputation using formulae

        personal_bmi: BMI of patient
    
    '''

    # Pharma variables of all types and categorical/binary variables are forward filled indefinitely
    if var_type=="pharma" or var_encoding in ["categorical", "binary"] or var_type=="weight":
        leave_nan_threshold_secs=np.inf
    else:
        leave_nan_threshold_secs=fill_interval_secs

    pred_values=np.zeros_like(timegrid_pred)
    cum_count_ms=np.zeros_like(timegrid_pred)
    time_to_last_ms=value_empty(timegrid_pred.size,-1.0)
    input_ts=0
    backward_window=[]
    slope_active=False
    slope_cum_secs=0
    
    # Fill in each point of the time-grid.
    for idx,ts in np.ndenumerate(timegrid_pred):
        
        while input_ts<raw_ts.size and raw_ts[input_ts]<=ts+grid_period:
            backward_window.append((raw_ts[input_ts],raw_values[input_ts]))
            input_ts+=1
            
        # POSTCONDITION: input_ts=raw_ts.size OR raw_ts[input_ts]>ts : Past the time grid point
        
        # No value has been observed before the current time-grid point. We have to fill in using the global mean
        if input_ts==0:

            # Fill in the normal value using dynamic formulae imputation (see the excel sheet for formula spec)
            if variable_id in ["v1000","vm13"]:
                current_weight=weight_imputed_col[idx[0]]
                if not np.isfinite(static_height):
                    static_height=np.sqrt(current_weight/personal_bmi)
                global_mean=3.5*0.007184*current_weight**0.425*static_height**0.725
            elif variable_id in ["v10020000","v30005010","v30005110","vm24","vm31","vm32"]:
                current_weight=weight_imputed_col[idx[0]]
                global_mean=current_weight
                
            pred_values[idx[0]]=global_mean
            continue

        ext_offset=ts-raw_ts[input_ts-1]

        # Filling using slope estimation and backward windows, for observations start are before the time-grid use forward filling
        if raw_ts[input_ts-1]>=0 and ext_offset>leave_nan_threshold_secs:

            ## Slope and aim point has to be recomputed
            if not slope_active:
                # Determine the aiming point for the extrapolation
                assert(len(backward_window)>0)
                rolling_mean_window=[]
                for search_idx in range(len(backward_window)-1, -1, -1):
                    history_ts,history_val = backward_window[search_idx]
                    if ts-history_ts>rolling_mean_secs:                        break
                    rolling_mean_window.append(history_val)

                # If the variable encoding is ordinal, we want to aim towards a valid value, so we clip the aim value
                # to the integer grid
                if var_encoding=="ordinal":
                    aim_val=int(round(np.median(np.array(rolling_mean_window))))
                else:
                    aim_val=np.median(np.array(rolling_mean_window))

                extrap_slope=(aim_val-raw_values[input_ts-1])/return_mean_secs
                delta_y=extrap_slope*grid_period
                gen_val=pred_values[idx[0]-1]+delta_y

                # If ordinal value, clip the generated value to the integer grid
                if var_encoding=="ordinal":
                    gen_val=int(round(gen_val))

                pred_values[idx[0]]=gen_val
                slope_cum_secs=grid_period
                slope_active=True

            # Continue to fill with the slope estimate and just advance on the X-axis.
            else:
                delta_y=extrap_slope*grid_period
                if slope_cum_secs<return_mean_secs:
                    gen_val=pred_values[idx[0]-1]+delta_y
                    pred_values[idx[0]]=gen_val
                else:
                    gen_val=pred_values[idx[0]-1]
                    pred_values[idx[0]]=gen_val

                slope_cum_secs+=grid_period

        # Classical forward filling
        else:
            slope_active=False
            
            # Handle the special case where the same variable was observed at the exact time stamp in two tables
            if input_ts>1 and raw_ts[input_ts-1]==raw_ts[input_ts-2]:
                gen_val=np.mean(raw_values[input_ts-2:input_ts])
                pred_values[idx[0]]=gen_val
            else:
                gen_val=raw_values[input_ts-1]
                pred_values[idx[0]]=gen_val

        cum_count_ms[idx[0]]=input_ts
        time_to_last_ms[idx[0]]=ext_offset

    return (pred_values,cum_count_ms, time_to_last_ms)

''' NO IMPUTATION SCHEMA ================================================================================================================'''

def no_impute(raw_ts, raw_values, timegrid_pred, global_mean, grid_period, fill_interval_secs=None,
              rolling_mean_secs=None, return_mean_secs=None, var_lower=None, var_upper=None, 
              var_type=None, var_encoding=None, variable_id=None, weight_imputed_col=None, static_height=None,
              personal_bmi=None):
    ''' Adaptive version of forward filling, which has specific forward filling thresholds as a function of the 
        median sampling rate of the variable, if it cannot forward fill it will return to the rolling mean of the last
        observation over a time which also is a function of the median sampling rate. This code does not impute but only fills into the
        directly adjacent grid interval.
    '''

    leave_nan_threshold_secs=360.0 # Fill up to 6 minutes forward, essentially means no imputation

    pred_values=np.zeros_like(timegrid_pred)
    cum_count_ms=np.zeros_like(timegrid_pred)
    time_to_last_ms=value_empty(timegrid_pred.size,-1.0)
    input_ts=0
    
    # Fill in each point of the time-grid.
    for idx,ts in np.ndenumerate(timegrid_pred):
        
        while input_ts<raw_ts.size and raw_ts[input_ts]<=ts+grid_period:
            input_ts+=1
            
        # POSTCONDITION: input_ts=raw_ts.size OR raw_ts[input_ts]>ts : Past the time grid point
        
        # No value has been observed before the current time-grid point. Do not impute
        if input_ts==0:
            pred_values[idx[0]]=0.0 if var_type=="pharma" else np.nan
            continue

        ext_offset=ts-raw_ts[input_ts-1]

        # Do not impute
        if ext_offset>leave_nan_threshold_secs:
            pred_values[idx[0]]=0.0 if var_type=="pharma" else np.nan
        else:
            
            # Handle the special case where the same variable was observed at the exact time stamp in two tables
            if input_ts>1 and raw_ts[input_ts-1]==raw_ts[input_ts-2]:
                gen_val=np.mean(raw_values[input_ts-2:input_ts])
                pred_values[idx[0]]=gen_val
            else:
                gen_val=raw_values[input_ts-1]
                pred_values[idx[0]]=gen_val

        cum_count_ms[idx[0]]=input_ts
        time_to_last_ms[idx[0]]=ext_offset

    return (pred_values,cum_count_ms, time_to_last_ms)

''' ONLY FORWARD FILLING imputation schema ========================================================================================================='''

def impute_forward_fill_simple(raw_ts, raw_values, timegrid_pred, global_mean, grid_period, fill_interval_secs=np.inf, 
                               var_type=None, variable_id=None, weight_imputed_col=None, static_height=None,
                               personal_bmi=None, custom_formula=None):
    ''' 
    Simple forward filling algorithm used in the respiratory failure project
    '''

    pred_values=np.zeros_like(timegrid_pred)
    cum_count_ms=np.zeros_like(timegrid_pred)
    time_to_last_ms=value_empty(timegrid_pred.size,-1.0)
    input_ts=0
    cum_real_meas=0
    last_real_ms=None

    for idx,ts in np.ndenumerate(timegrid_pred):
        
        while input_ts<raw_ts.size and raw_ts[input_ts]<=ts:

            # For pharma rate variables do not count 0 rate as a measurement
            if var_type=="Ordinal" and "pm" in variable_id:
                if not raw_values[input_ts]==0.0:
                    cum_real_meas+=1
                    last_real_meas=input_ts
            else:
                cum_real_meas+=1
                last_real_ms=input_ts

            input_ts+=1
            
        # POSTCONDITION: input_ts=raw_ts.size OR raw_ts[input_ts]>ts : Past the time grid point

        # Compute dynamic fill values for custom imputation
        if variable_id in ["vm13"] and custom_formula:
            current_weight=weight_imputed_col[idx[0]]
            if not np.isfinite(static_height):
                static_height=np.sqrt(current_weight/personal_bmi)
            global_mean=3.5*0.007184*current_weight**0.425*static_height**0.725
        elif variable_id in ["vm24","vm31","vm32"] and custom_formula:
            current_weight=weight_imputed_col[idx[0]]
            global_mean=current_weight
        
        # No value has been observed before the current time-grid point. We have to fill in using the global mean
        if input_ts==0:
            pred_values[idx[0]]=global_mean
            continue

        ext_offset=ts-raw_ts[input_ts-1]

        if last_real_ms is None:
            real_offset=-1.0
        else:
            real_offset=ts-raw_ts[last_real_ms]

        assert(ext_offset>=0)

        # Fill with normal value after forward filling horizon
        if ext_offset>fill_interval_secs:
            pred_values[idx[0]]=global_mean
        else:

            # Handle the special case where the same variable was observed at the exact time stamp in two tables
            if input_ts>1 and raw_ts[input_ts-1]==raw_ts[input_ts-2]:
                gen_val=np.mean(raw_values[input_ts-2:input_ts])
                pred_values[idx[0]]=gen_val
            else:
                gen_val=raw_values[input_ts-1]
                pred_values[idx[0]]=gen_val

        cum_count_ms[idx[0]]=cum_real_meas
        time_to_last_ms[idx[0]]=real_offset

    return (pred_values,cum_count_ms, time_to_last_ms)

''' =========================================================================================================================='''
