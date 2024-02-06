
import argparse
import os
import os.path
import sys
import gin

from label_renal import label_gen_renal

def execute(configs):
    label_gen_renal(configs)

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
    parser.add_argument("--split_key", default=None, help="For which split should labels be produced?")
    parser.add_argument("--batch_idx", type=int, default=None, help="On which batch should this process operate?")
    parser.add_argument("--debug_mode", action="store_true", default=None, help="Debug mode for testing, no output created to file-system")
    parser.add_argument("--run_mode", default=None, help="Execution mode")

    # GIN config
    #parser.add_argument("--gin_config", default="./configs/label_gen.gin",help="Location of GIN config to load, and overwrite the arguments")
    parser.add_argument("--gin_config", default="./configs/label_gen_MIMIC.gin",help="Location of GIN config to load, and overwrite the arguments")    

    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    split_key=configs["split_key"]
    batch_idx=configs["batch_idx"]

    execute(configs)

    print("JOB STATUS: EXITED SUCCESSFULLY")
