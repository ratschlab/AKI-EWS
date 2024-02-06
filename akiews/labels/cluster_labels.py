''' 
Cluster dispatch script for label generation
'''

import subprocess
import os
import os.path
import sys
import argparse
import gin
import itertools

import mlhc_data_manager.util.filesystem as mlhc_fs
import mlhc_data_manager.util.array as mlhc_array
import mlhc_data_manager.util.io as mlhc_io
import mlhc_data_manager.util.memory as mlhc_memory

def execute(configs):
    ''' Computes labels for all possible impute data and label type combinations'''
    compute_script_path=configs["compute_script_path"]
    job_index=0
    mem_in_mbytes=configs["compute_mem"]
    n_cpu_cores=configs["compute_n_cores"]
    n_compute_hours=configs["compute_n_hours"]
    bad_hosts=["le-amd-fp-004","le-a2-043","le-g2-005"]
    label_base_path=configs["label_dir"]
    is_dry_run=configs["dry_run"]
    logdir=os.path.join(configs["log_dir"])

    if not is_dry_run and not configs["preserve_logs"]:
        print("Deleting previous log-files...")
        for logf in os.listdir(logdir):
            os.remove(os.path.join(logdir, logf))

    batch_range=list(range(configs["BATCHES"]))
            
    for split_key in configs["SPLIT_MODES"]:
        for batch_idx in batch_range:

            print("Create label patient data for split {}, batch {}".format(split_key,batch_idx))
            job_name="labelgen_{}_{}_{}".format(configs["endpoint"], split_key,batch_idx)
            log_stdout_file=os.path.join(logdir, "{}.stdout".format(job_name))
            log_stderr_file=os.path.join(logdir, "{}.stderr".format(job_name))            
            mlhc_fs.delete_if_exist(log_stdout_file)
            mlhc_fs.delete_if_exist(log_stderr_file)            
            cmd_line=" ".join(["sbatch", "--mem-per-cpu", "{}".format(mem_in_mbytes), "-n", "{}".format(n_cpu_cores), "--time {}:00:00".format(n_compute_hours),
                               "--exclude=gpu-biomed-16,gpu-biomed-23,compute-biomed-10",
                               "--partition=compute",                               
                               "-o", log_stdout_file, "-e", log_stderr_file, "--job-name","{}".format(job_name), "--wrap", '\"python3', compute_script_path,
                               "--run_mode CLUSTER", "--split_key {}".format(split_key),
                               "--gin_config {}".format(configs["script_gin_config"]), "--batch_idx {}".format(batch_idx), '\"'])
            assert(" rm " not in cmd_line)
            job_index+=1

            if is_dry_run:
                print("CMD: {}".format(cmd_line))
            else:
                subprocess.call([cmd_line], shell=True)
                if configs["debug_mode"]:
                    sys.exit(0)

    print("Number of generated jobs: {}".format(job_index))

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

    parser.add_argument("--circ_label_dir", default="/cluster/work/grlab/clinical/hirid2/research/6a_labels_circ/v8")
    parser.add_argument("--dry_run", default=None, action="store_true", help="Should a dry run be run, without dispatching the jobs?")
    parser.add_argument("--debug_mode", default=None, action="store_true", help="Debug mode?")
    parser.add_argument("--preserve_logs", default=None, action="store_true", help="Should logs be preserved?")
    
    #parser.add_argument("--gin_config", default="./configs/cluster_label_gen.gin", help="Location of config file")
    parser.add_argument("--gin_config", default="./configs/cluster_label_gen_MIMIC.gin", help="Location of config file")    

    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    execute(configs)
