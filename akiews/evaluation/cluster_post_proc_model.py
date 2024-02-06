''' Cluster dispatcher'''

import os
import os.path
import argparse
import subprocess
import ipdb

def execute(configs):
    mem_in_mbytes=configs["mbytes_per_job"]
    n_cpu_cores=1
    n_compute_hours=configs["hours_per_job"]

    batch_range=list(range(100))
    #batch_range=list(range(50))    
    
    for batch_idx in batch_range:
        job_name="icukidney_postproc_pred_{}".format(batch_idx)
        log_stdout_file=os.path.join(configs["log_dir"],"icukidney_postproc_{}_RESULT.stdout".format(batch_idx))
        log_stderr_file=os.path.join(configs["log_dir"],"icukidney_postproc_{}_RESULT.stderr".format(batch_idx))        
        if os.path.exists(log_stdout_file):
            os.remove(log_stdout_file)
        if os.path.exists(log_stderr_file):
            os.remove(log_stderr_file)
        
        cmd_line=" ".join(["sbatch", "--mem-per-cpu {}".format(mem_in_mbytes), 
                           "-n", "{}".format(n_cpu_cores),
                           "--time", "{}:00:00".format(n_compute_hours),
                           "--exclude gpu-biomed-08,gpu-biomed-16,gpu-biomed-23,compute-biomed-10",
                           "--mail-type FAIL",
                           "--partition=compute",
                           "--job-name","{}".format(job_name),
                           "-o", log_stdout_file,
                           "-e", log_stderr_file,
                           "--wrap",
                           '\"python3', configs["compute_script_path"], "--run_mode BATCH",
                           "--batch_idx {}".format(batch_idx), '\"'])
        assert " rm " not in cmd_line

        if configs["dry_run"]:
            print("CMD: {}".format(cmd_line))
        else:
            subprocess.call([cmd_line],shell=True)
    

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--compute_script_path",
                        default="/cluster/home/mhueser/git/projects/2022/kidnews_public/kidnews/evaluation/post_proc_separate_model.py",
                        help="Script to launch")
    
    # Output paths
    parser.add_argument("--log_dir", default="/cluster/home/mhueser/log_files/icu_score_renal", help="Log directory")

    # Arguments
    parser.add_argument("--mbytes_per_job", type=int, default=32000, help="Memory per job")
    parser.add_argument("--hours_per_job", type=int, default=4, help="Time per job")
    parser.add_argument("--dry_run", default=False, action="store_true", help="Dry run without command launch")

    configs=vars(parser.parse_args())

    execute(configs)
