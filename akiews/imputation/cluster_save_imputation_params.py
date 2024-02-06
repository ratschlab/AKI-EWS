'''
Cluster dispatcher for the script <save_imputation_params.py>
'''

import subprocess
import os
import os.path
import sys
import argparse
import gin

import mlhc_data_manager.util.filesystem as mlhc_fs
import mlhc_data_manager.util.array as mlhc_array
import mlhc_data_manager.util.io as mlhc_io
import mlhc_data_manager.util.memory as mlhc_memory

def execute(configs):
    compute_script_path=configs["compute_script_path"]
    job_index=0
    mem_in_mbytes=configs["mem_in_mbytes"]
    n_cpu_cores=configs["n_cpu_cores"]
    n_compute_hours=configs["n_compute_hours"]
    bad_hosts=[]

    # Generate data for the different splits
    for split_key in configs["split_configs"]:

        print("Generating imputation parameters for split {}".format(split_key))
        job_name="imputationparams_{}".format(split_key)
        log_result_file=os.path.join(configs["log_dir"],"{}_RESULT.txt".format(job_name))
        mlhc_fs.delete_if_exist(log_result_file)
        cmd_line=" ".join(["bsub", "-G", "ms_raets", "-R", "rusage[mem={}]".format(mem_in_mbytes), "-n", "{}".format(n_cpu_cores), "-r", "-W", "{}:00".format(n_compute_hours),
                           " ".join(['-R "select[hname!=\'{}\']"'.format(bad_host) for bad_host in bad_hosts]),
                           "-J","{}".format(job_name), "-o", log_result_file, "python3", compute_script_path, "--run_mode CLUSTER",
                           "--split_key {}".format(split_key), "--gin_config {}".format(configs["script_gin_file"])])
        assert(" rm " not in cmd_line)
        job_index+=1

        if configs["dry_run"]:
            print(cmd_line)
        else:
            subprocess.call([cmd_line], shell=True)

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
    parser.add_argument("--gin_config", default="./gin_configs/cluster_save_parameters_kidney.gin")
    parser.add_argument("--dry_run", default=None, action="store_true")
    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    execute(configs)


