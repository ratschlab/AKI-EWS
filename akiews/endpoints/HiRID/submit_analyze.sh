#!/bin/sh
module load gcc/4.8.5 python_cpu/3.7.1 hdf5/1.10.1



bsub -R "rusage[mem=5000]" -n 5 -W 4:00 "python -W ignore Renal_failure_classfication_stats.py --baseline_creatinine min_i --gap 0 --halflife_urine 0 --transition geq1 --masked_urine_treatment 1"

bsub -R "rusage[mem=5000]" -n 5 -W 4:00 "python -W ignore Renal_failure_classfication_stats.py --baseline_creatinine min_i --gap 0 --halflife_urine 0 --transition geq1 --masked_urine_treatment 0"


# for t in {"geq1","geq2","geq3"}
# do
#     for k in {"True","False"}
#     do
#         bsub -R "rusage[mem=5000]" -n 5 -W 4:00 "python -W ignore Renal_failure_classfication_stats.py --baseline_creatinine min_i --gap 0 --halflife_urine 0 --transition $t --masked_urine_treatment $k"
#     done
# done

# bsub -R "rusage[mem=5000]" -n 20 -W 4:00 "python -W ignore Merging_stats.py"

