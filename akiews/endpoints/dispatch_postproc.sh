#!/bin/bash

for gl in 4 8 12 16 20 24 28 32
do
    for se in 4 6 8 10 12
    do
	echo ${se} ${gl}
	bsub -o /cluster/home/mhueser/log_files/icu_score_renal/exit_log_se${se}_gl${gl}.txt python3 ./postproc_endpoint.py --small_event_length ${se} --gap_length ${gl} --cluster_mode
    done
done
