
import os
import os.path
import ipdb
import argparse
import sys
import timeit

import gin
import numpy as np
import pandas as pd

from ml_input_util import gen_features_df

import akiews.utils.io as mlhc_io
import akiews.utils.memory as mlhc_memory


def is_df_sorted(df, colname):
    return (np.array(df[colname].diff().dropna(),dtype=np.float64) >=0).all()


def execute(configs):
    ''' Parallel wrapper for feature generation, processing one batch of the data as a time, 
        corresponding to one output file'''
    split_key=configs["split_key"]
    batch_idx=configs["batch_idx"]

    print("Job SPLIT: {}, BATCH: {}".format(split_key, batch_idx),flush=True)
    imputed_base_path=configs["imputed_path"]
    label_base_path=configs["label_path"]
    output_base_path=configs["output_features_path"]
    fmat_dir=os.path.join(imputed_base_path,"reduced",split_key)
    lmat_path=os.path.join(label_base_path,"reduced", split_key)
    ml_output_dir=os.path.join(output_base_path,"reduced",split_key)
    var_encoding_dict=mlhc_io.load_pickle(configs["meta_varenc_map_path"])
    var_parameter_dict=mlhc_io.load_pickle(os.path.join(configs["meta_varprop_map_path"],"interval_median_point_est.pickle"))
    hirid_schema_dict=mlhc_io.load_pickle(configs["hirid_v8_dict"])
    pharma_dict=np.load(configs["pharma_acting_period_map_path"],allow_pickle=True).item()
    data_split=mlhc_io.load_pickle(configs["temporal_split_path"])[split_key]
    all_pids=data_split["train"]+data_split["val"]+data_split["test"]

    if configs["compat_thomas_grid"]:
        df_thomas_ref=pd.read_hdf(os.path.join(configs["thomas_grid_compat_path"], "batch_{}.h5".format(batch_idx)), mode='r')

    # Buffers to hold all output matrices for this batch
    df_X_buffer=[]
    df_y_buffer=[]

    if configs["verbose"]:
        print("Number of patient IDs: {}".format(len(all_pids),flush=True))

    batch_map=mlhc_io.load_pickle(configs["pid_map_path"])["chunk_to_pids"]
    pids_batch=batch_map[batch_idx]
    selected_pids=list(set(pids_batch).intersection(all_pids))
    print("Number of selected PIDs for this batch: {}".format(len(selected_pids)),flush=True)
    batch_path=os.path.join(fmat_dir,"batch_{}.h5".format(batch_idx))
    n_skipped_patients=0

    if not os.path.exists(batch_path):
        print("WARNING: No input data for batch, skipping...",flush=True)
        print("Generated path: {}".format(batch_path))
        return 0

    # Compute complex features on a special set of variables
    if configs["list_special_variables"] is not None and configs["add_complex_features"]:
        imp_vars=mlhc_io.read_list_from_file(configs["list_special_variables"])

    # Compute the same features on all variables
    else:
        imp_vars=[]

    lab_path=os.path.join(lmat_path,"batch_{}.h5".format(batch_idx))

    if not os.path.exists(lab_path):
        print("WARNING: No input label data for batch, skipping...",flush=True)
        return 0

    if configs["attach_bowen_hmm_features"]:
        df_bowen_samples_all=pd.read_hdf(os.path.join(configs["bowen_hmm_path"],"v2_batch","batch_{}.h5".format(batch_idx)),mode='r')
        df_bowen_probs_all=pd.read_hdf(os.path.join(configs["bowen_hmm_path"],"v2_stateProb_batch","batch_{}.h5".format(batch_idx)),mode='r')
        hmm_feat_cols=["AbsDatetime"]+list(filter(lambda col: "iohmm" in col, df_bowen_samples_all.columns.values.tolist()))
        hmm_probs_feat_cols=["AbsDatetime"]+list(filter(lambda col: "iohmm" in col, df_bowen_probs_all.columns.values.tolist()))

    first_write=True
    t_begin=timeit.default_timer()

    for pidx,pid in enumerate(selected_pids):
        df_pat=pd.read_hdf(batch_path,mode='r',where="PatientID={}".format(pid))

        try:
            df_label_pat=pd.read_hdf(lab_path,"/labels_{}".format(pid), mode='r')
        except KeyError:
            try:
                df_label_pat=pd.read_hdf(lab_path,mode='r',where="PatientID={}".format(pid))
            except ValueError:
                df_label_pat=None

        df_ep_pat=None

        if configs["attach_bowen_hmm_features"]:
            df_bowen_samples_pid=df_bowen_samples_all[df_bowen_samples_all.PatientID==pid]
            df_bowen_probs_pid=df_bowen_probs_all[df_bowen_probs_all.PatientID==pid]
            df_bowen_samples_pid=df_bowen_samples_pid[hmm_feat_cols]
            df_bowen_probs_pid=df_bowen_probs_pid[hmm_probs_feat_cols]

            if df_bowen_samples_pid.shape[0]>0:
                df_bowen_samples_pid.set_index(keys="AbsDatetime",inplace=True,verify_integrity=True)
                df_bowen_samples_pid=df_bowen_samples_pid.reindex(index=df_pat.AbsDatetime,method="nearest")

            if df_bowen_probs_pid.shape[0]>0:
                df_bowen_probs_pid.set_index(keys="AbsDatetime",inplace=True,verify_integrity=True)
                df_bowen_probs_pid=df_bowen_probs_pid.reindex(index=df_pat.AbsDatetime,method="nearest")

        if df_pat.shape[0]==0 or df_label_pat is None or df_label_pat.shape[0]==0:
            n_skipped_patients+=1
            continue

        assert(df_pat.shape[0]==df_label_pat.shape[0])
        assert(is_df_sorted(df_label_pat,"AbsDatetime"))
        assert(is_df_sorted(df_pat,"AbsDatetime"))

        if configs["attach_bowen_hmm_features"]:
            assert(df_bowen_samples_pid.shape[0]==0 or df_pat.shape[0]==df_bowen_samples_pid.shape[0])
            assert(df_bowen_probs_pid.shape[0]==0 or df_pat.shape[0]==df_bowen_probs_pid.shape[0])
            df_X, df_y=gen_features_df(df_pat,df_label_pat,df_ep_pat,pid=pid,configs=configs,imp_vars=imp_vars,
                                       interval_dict=var_parameter_dict, hirid_dict=hirid_schema_dict,bowen_hmm=df_bowen_samples_pid,
                                       bowen_hmm_probs=df_bowen_probs_pid)
            
        else:
            df_X, df_y=gen_features_df(df_pat,df_label_pat,df_ep_pat,pid=pid,configs=configs,imp_vars=imp_vars,
                                       interval_dict=var_parameter_dict, hirid_dict=hirid_schema_dict)            

        if df_X is None or df_y is None:
            print("WARNING: Features could not be generated for PID: {}".format(pid), flush=True)
            n_skipped_patients+=1
            continue

        assert(df_X.shape[0]==df_y.shape[0])
        assert(df_X.shape[0]==df_pat.shape[0])

        # Compatibility with Thomas grid, intersect the data-frame with his grid
        if configs["compat_thomas_grid"]:
            df_thomas_pid=df_thomas_ref[df_thomas_ref.PatientID==pid]

            # Should only generate patients which are also present for Thomas
            if df_thomas_pid.shape[0]==0:
                print("Skipping because there is no patient for Thomas...")
                continue

            tg_grid=list(df_thomas_pid.AbsDatetime.unique())
            df_X=df_X[df_X.AbsDatetime.isin(tg_grid)]
            df_y=df_y[df_y.AbsDatetime.isin(tg_grid)]

            tg_cols=df_thomas_pid.columns.values.tolist()
            dyn_cols=list(filter(lambda col: col[:2] in ["vm","pm"], tg_cols))

            # Double check the feature columns are still the same
            for col in dyn_cols:
                tg_arr=np.array(df_thomas_pid[col]).astype(np.float)
                mh_arr=np.array(df_X["plain_{}".format(col)])
                assert(np.allclose(tg_arr,mh_arr,equal_nan=True))
        
        df_X_buffer.append(df_X)
        df_y_buffer.append(df_y)

        print("Job {}: {:.2f} %".format(batch_idx, (pidx+1)/len(selected_pids)*100),flush=True)
        t_current=timeit.default_timer()
        tpp=(t_current-t_begin)/(pidx+1)
        eta_minutes=(len(selected_pids)-(pidx+1))*tpp/60.0
        print("ETA [minutes]: {:.2f}".format(eta_minutes),flush=True)
        print("Number of skipped patients: {}".format(n_skipped_patients),flush=True)
        mlhc_memory.print_memory_diags()

    df_X_all=pd.concat(df_X_buffer,axis=0)
    df_y_all=pd.concat(df_y_buffer,axis=0)

    if not configs["debug_mode"]:
        df_X_all.to_hdf(os.path.join(ml_output_dir,"batch_{}.h5".format(batch_idx)),"/X",format="fixed",append=False,
                        mode='w', complevel=configs["hdf_comp_level"],complib=configs["hdf_comp_alg"])
        df_y_all.to_hdf(os.path.join(ml_output_dir,"batch_{}.h5".format(batch_idx)),"/y",format="fixed",append=False,
                        mode='a',complevel=configs["hdf_comp_level"],complib=configs["hdf_comp_alg"])

    return 0



@gin.configurable
def parse_gin_args(old_configs,gin_configs=None):
    gin_configs=gin.query_parameter("parse_gin_args.gin_configs")
    for k in old_configs.keys():
        if old_configs[k] is not None:
            gin_configs[k]=old_configs[k]
    gin.bind_parameter("parse_gin_args.gin_configs",gin_configs)
    return gin_configs



if __name__=="__main__":
    parser=argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--debug_mode", default=None, action="store_true", help="Should debug mode be used?")
    parser.add_argument("--verbose", default=None, action="store_true", help="Should verbose logging messages be printed?")
    parser.add_argument("--run_mode", default=None, help="Execution mode")
    parser.add_argument("--split_key", default=None, help="Split to process")
    parser.add_argument("--batch_idx", default=None, type=int, help="Batch to process")

    #parser.add_argument("--gin_config", default="./configs/save_ml_input.gin", help="Location of GIN config file")
    parser.add_argument("--gin_config", default="./configs/save_ml_input_MIMIC.gin", help="Location of GIN config file")
    
    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    run_mode=configs["run_mode"]
    split_key=configs["split_key"]
    batch_idx=configs["batch_idx"]
    output_features_path=configs["output_features_path"]

    execute(configs)

    print("JOB STATUS: EXITED SUCCESSFULLY")


    


