#!/bin/sh
module load python_cpu/3.7.1 hdf5/1.10.1

# bsub -R "rusage[mem=5000]" -W 4:00 "python -W ignore load_mimic.py"
bsub -R "rusage[mem=5000]" -W 4:00 "python -W ignore analyse_creatinine.py"

bsub -R "rusage[mem=5000]" -n 20 -W 120:00 "python -W ignore Renal_failure_endpoints.py --number -1 --output_to_disk 1 --baseline_creatinine min_i --gap 0 --halflife_urine 0"

for n in {0..50}
    do
        bsub -R "rusage[mem=5000]" -n 20 -W 4:00 "python -W ignore Renal_failure_endpoints.py --number $n --output_to_disk 1 --baseline_creatinine min_i --gap 0 --halflife_urine 0"
    done


bsub -R "rusage[mem=5000]" -W 4:00 "python -W ignore Renal_failure_classfication_stats.py"
