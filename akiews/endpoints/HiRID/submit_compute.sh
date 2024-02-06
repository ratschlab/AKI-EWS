#!/bin/sh
module load python_cpu/3.7.1 hdf5/1.10.1

# for b in {"literature","min","literature_i"}
# do
# for h in {0,2}
# do
    # for g in {0,1,2,4,8,12,16,24,36,48}
    # do
for i in {0..100}
do
    # bsub -R "rusage[mem=5000]" -n 20 -W 24:00 "python -W ignore Renal_failure_endpoints.py --number $i --output_to_disk 1 --baseline_creatinine min_i --gap 0 --halflife_urine 0"
    bsub -R "rusage[mem=5000]" -n 5 -W 24:00 "python -W ignore Renal_failure_endpoints.py --number $i --output_to_disk 1 --baseline_creatinine min_i --gap 0 --halflife_urine 0"
done
    # done
# done
# done
