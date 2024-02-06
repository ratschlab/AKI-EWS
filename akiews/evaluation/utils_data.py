import gc
import json
import h5py
import pandas as pd

from os.path import join
from os import listdir

def read_model_data(model_json, pids):
    with open(model_json, "r") as tmp:
        models = json.load(tmp)["models"]
        
    df_model = dict()
    for model_name, model_config in models.items():
        if model_config["file_in_batches"] in ["yes", "y"]:

            if "clinical_baseline" in model_config and model_config["clinical_baseline"]=="yes":
                df = []
                for f in listdir(model_config["data_path"]):
                    df_b = pd.read_hdf(join(model_config["data_path"], f), columns=["PatientID"])
                    if len(set(pids)&set(df_b.PatientID))==0:
                        del df_b
                        gc.collect()
                        continue
                    df_b = pd.read_hdf(join(model_config["data_path"], f), columns=[v for _,v in model_config["usecols"].items()])
                    df_b = df_b.drop(df_b.index[~df_b.PatientID.isin(pids)])
                    gc.collect()
                    df.append(df_b)
                df = pd.concat(df).reset_index(drop=True)
                df = df.rename(columns={v: k for k,v in model_config["usecols"].items()})                    
                df = df.sort_values(["PatientID", "Datetime"]).reset_index(drop=True)
                df.loc[:,"Prediction"] = df["spo2"] / df["fio2"]
                max_val = df.Prediction.max()
                min_val = df.Prediction.min()
                df.loc[:,"Prediction"] = (df["Prediction"] - min_val) / (max_val - min_val)
                
            else:
                df = []
                for f in listdir(model_config["data_path"]):
                    if not( ".h5" in f or ".hdf5" in f):
                        continue
                    with h5py.File(join(model_config["data_path"], f), "r") as tmp:
                        batch_pids = [int(x[1:]) for x in tmp.keys()]
                        
                    if len(set(pids)&set(batch_pids)) == 0:
                        continue
                    
                    batch_pids = list(set(pids)&set(batch_pids))
                    for pid in batch_pids:
                        df_p = pd.read_hdf(join(model_config["data_path"], f), "p%d"%pid)
                        df_p = df_p.rename(columns={v: k for k,v in model_config["usecols"].items()})
                        df_p = df_p.sort_values("Datetime")
                        df.append(df_p)
                df = pd.concat(df).reset_index(drop=True)
        elif model_config["file_in_batches"] in ["no", "n"]:
            
            if model_config["data_format"]=="csv":
                func_read_data = pd.read_csv
            elif model_config["data_format"]=="parquet":
                func_read_data = pd.read_parquet
                
            df = func_read_data(model_config["data_path"])
            df = df.rename(columns={v: k for k,v in model_config["usecols"].items()})
            df = df[df.PatientID.isin(pids)]
            df = df.sort_values(["PatientID", "Datetime"]).reset_index(drop=True)
            
        else:
            raise Exception("Only choose 'yes'/'y' or 'no'/'n' for file_in_batches.")

        df = df[["PatientID", "Datetime", "Prediction"]]
        df.loc[:,"PatientID"] = df["PatientID"].astype(int)
        df.loc[:,"Datetime"] = pd.to_datetime(df["Datetime"]).dt.floor("5T")
        
        if len(models) == 1:
            pass
        else:
            df = df.set_index(["PatientID", "Datetime"])
            df = df.rename(columns={"Prediction": model_name})
            
        df_model.update({model_name: df})

    if len(models) == 1:
        
        return df_model[model_name]
    
    else:
        
        df_merge = pd.concat([df_m for _, df_m in df_model.items()], axis=1, join="inner")
        for model_name in models.keys():
            if models[model_name]["use"] in ["yes", "y"]:
                select_model = model_name
                break

        df = df_merge[[select_model]].reset_index()
        df = df.rename(columns={select_model: "Prediction"})
        return df

def get_result(respath, configs, calibrated_s=None, RANDOM=False, random_seed=None, onset_type=None):
    '''
    read TA, FA, ME, CE information and compute rec and prec
    '''
    f_key = get_keystr(configs)
    if RANDOM:
        if random_seed is None:
            f_key = "rand_" + f_key
        else:
            f_key = "rand%d_"%random_seed + f_key
    else:
        pass
    
    if onset_type == "first":
        f_key += "_first"
        
    aggrfile = [f for f in os.listdir(respath) if f==(f_key+".csv")]  
    if len(aggrfile) > 0:
        df = pd.read_csv(os.path.join(respath, aggrfile[0]))
        df = compute_precision_recall(df, calibrated_s=calibrated_s)
    else:
        if onset_type=="first":
            batchfiles = [f for f in os.listdir(respath) if f[:len(f_key)]==f_key and "batch" in f]
        else:
            batchfiles = [f for f in os.listdir(respath) if f[:len(f_key)]==f_key and "batch" in f and "first" not in f]
            
        if len(batchfiles) > 0:
            df = [pd.read_csv(os.path.join(respath, f)) for f in batchfiles]
            df = pd.concat(df).groupby("tau").sum().reset_index()
        else:
            if onset_type=="first":
                cntfiles = [f for f in os.listdir(respath) if f[:len(f_key)]==f_key and "cnts" in f]
            else:
                cntfiles = [f for f in os.listdir(respath) if f[:len(f_key)]==f_key and "cnts" in f and "first" not in f]
                df = [pd.read_csv(os.path.join(respath, f)) for f in cntfiles]    
                df = pd.concat(df)
                
        df.sort_values("tau", inplace=True)
        df = df.reset_index(drop=True)
        df2write = df.copy()
        df = compute_precision_recall(df, calibrated_s=calibrated_s)
        df2write.to_csv(os.path.join(respath, f_key+".csv"), index=False)
        # if RANDOM:
        #     df.loc[df.index[df.rec<0.6],"prec"] = df.loc[df.index[df.rec>=0.6],"prec"].iloc[0]
    return df
