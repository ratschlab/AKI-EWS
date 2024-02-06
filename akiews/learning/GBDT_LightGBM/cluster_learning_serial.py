''' 
Cluster dispatch script for ML learning
'''

import subprocess
import os
import os.path
import sys
import argparse
import itertools
import random
import ipdb
import glob
import csv
import json
import gin

import akiews.utils.filesystem as mlhc_fs


def execute(configs):
    ''' Computes ML features in the current configuration on all possible label+imputed data configurations'''
    random.seed(configs["random_seed"])
    job_index=0
    mem_in_mbytes=configs["mbytes_per_job"]
    n_cpu_cores=configs["num_cpu_cores"]
    n_compute_hours=configs["hours_per_job"]
    is_dry_run=configs["dry_run"]
    ml_model=configs["ml_model"]
    col_desc=configs["col_desc"]
    bad_hosts=configs["bad_hosts"]
    
    if not is_dry_run and not configs["preserve_logs"]:
        print("Deleting previous log files...")
        for logf in os.listdir(configs["log_dir"]):
            os.remove(os.path.join(configs["log_dir"], logf))

    for bern_split_key,mimic_split_key in configs["SPLIT_CONFIGS"]:

        if configs["ext_val_mode"] in ["internal","validation"]:
            split_key=bern_split_key
        else:
            split_key=mimic_split_key
            
        split_dir=os.path.join(configs["pred_dir"],"reduced",split_key)

        if not is_dry_run:
            mlhc_fs.create_dir_if_not_exist(split_dir)

        for task_key,eval_key in configs["ALL_TASKS"]:
            output_base_key="{}_{}_{}".format(task_key, col_desc, ml_model)
            pred_output_dir=os.path.join(split_dir,output_base_key)
            if not is_dry_run:
                mlhc_fs.create_dir_if_not_exist(pred_output_dir)

            print("Fit ML model for split {}, task: {}, ML model: {}".format(split_key,task_key,ml_model))
            job_name="mlfit_{}_{}_{}".format(configs["col_desc"],split_key,task_key,ml_model)
            log_stdout_file=os.path.join(configs["log_dir"],"{}.stdout".format(job_name))
            log_stderr_file=os.path.join(configs["log_dir"],"{}.stderr".format(job_name))                        
            mlhc_fs.delete_if_exist(log_stdout_file)
            mlhc_fs.delete_if_exist(log_stderr_file)

            if ml_model in ["lightgbm","tree","logreg","lgbm_flaml"]:
                cmd_line=" ".join(["sbatch", "--mem-per-cpu {}".format(mem_in_mbytes), 
                                   "-n", "{}".format(n_cpu_cores), 
                                   "--time", "{}:00:00".format(n_compute_hours),
                                   "--exclude=gpu-biomed-16,gpu-biomed-23,gpu-biomed-22,gpu-biomed-08,compute-biomed-10", 
                                   "--partition=compute",
                                   "--mail-type FAIL",                                   
                                   "--job-name","{}".format(job_name), "-o", log_stdout_file,"-e", log_stderr_file, "--wrap",
                                   '\"python3', configs["compute_script_path"], "--run_mode CLUSTER", "--gin_config {}".format(configs["script_gin_file"]),
                                   "--bern_split_key {}".format(bern_split_key),
                                   "--mimic_split_key {}".format(mimic_split_key),
                                   "--column_set {}".format(configs["col_desc"]),
                                   "--label_key {}".format(task_key),"" if eval_key is None else "--eval_label_key {}".format(eval_key),
                                   "--ml_model {}".format(ml_model), '\"'])

            assert(" rm " not in cmd_line)
            job_index+=1

            if configs["dry_run"]:
                print("CMD: {}".format(cmd_line))
            else:
                subprocess.call([cmd_line], shell=True)

                if configs["debug_mode"]:
                    sys.exit(0)

    print("Generated {} jobs...".format(job_index))


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

    parser.add_argument("--dry_run", default=None, action="store_true", help="Should a dry-run be used?")
    parser.add_argument("--debug_mode", default=None, action="store_true", help="Debugging mode, run only one job")
    parser.add_argument("--preserve_logs", default=None, action="store_true", help="Should logging files be preserved?")
    parser.add_argument("--1percent_sample", default=None, action="store_true", help="Should a 1 % sample of train/val be used, for debugging")
    
    #parser.add_argument("--gin_config", default="./configs/cluster.gin", help="GIN config to use")
    parser.add_argument("--gin_config", default="./configs/cluster_separate.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_separate_MIMIC.gin", help="GIN config to use") 

    #parser.add_argument("--gin_config", default="./configs/cluster_internal.gin", help="GIN config to use")    
    #parser.add_argument("--gin_config", default="./configs/cluster_val.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_retrain.gin", help="GIN config to use")        

    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    execute(configs)
