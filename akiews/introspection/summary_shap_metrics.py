''' Compute a summary variable ranking based on SHAP values in different splits'''

import csv
import os.path
import os
import ipdb
import argparse

import numpy as np
import gin

import akiews.utils.io as mlhc_io

def execute(configs):
    score_dict={}
    name_dict=np.load(configs["mid_dict"],allow_pickle=True).item()
    backup_dict=configs["backup_dict"]
    
    for split in configs["SPLITS"]:
        print("Analyzing split: {}".format(split))
        spath=os.path.join(configs["pred_path"], split, configs["task_key"],"best_model_shapley_values.tsv")
        with open(spath,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for desc,score in csv_fp:
                if desc not in score_dict:
                    score_dict[desc]=[]
                score_dict[desc].append(float(score))
            
    for k in score_dict.keys():
        score_dict[k]=np.mean(score_dict[k])

    sorted_feats=sorted(score_dict.keys(), key=lambda k: score_dict[k],reverse=True)

    top_f=list(sorted_feats)[:configs["nfeats"]]

    if configs["output_feats"]:
        mlhc_io.write_list_to_file(configs["feat_list"],top_f)
    
    top_feats=[]

    # Save the top variables
    for f in sorted_feats:
        if len(top_feats)>=configs["nvars"]:
            break
        if "plain" in f:
            var=f.split("_")[-1].strip()
        elif "static" in f:
            var=f
        else:
            var=f.split("_")[0].strip()
        if var not in top_feats:
            top_feats.append(var)

    for idx,f in enumerate(top_feats):
        var_name=name_dict[f] if f in name_dict else (backup_dict[f] if f in backup_dict else f)
        print("{}: {} ({})".format(idx+1,var_name,f))

    if configs["output_vars"]:
        mlhc_io.write_list_to_file(configs["var_list"],top_feats)


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

    parser.add_argument("--gin_config", default="./configs/summary_shap.gin")

    configs=vars(parser.parse_args())
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    execute(configs)
