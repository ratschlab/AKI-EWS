hirid1_path="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital"
hirid2_path="/cluster/work/grlab/clinical/hirid2/research"
chunking_info_file="${hirid2_path}/misc_derived/id_lists/PID_all_vars_chunking_100.csv"

#####################
##  Filter patients
#####################
# input_path="${hirid2_path}/pg_db_export/"
# pid_excl_file="${hirid1_path}/misc_derived/id_lists/PID_Exclusion_GeneralConsent.csv ${hirid2_path}/research/misc_derived/id_lists/hospital_wide_gc.csv"
# output_file="${hirid2_path}/research/misc_derived/id_lists/PID_WithConsent.csv"
# bsub -G ms_raets -J step0a -R "rusage[mem=16384]" python step0a_remove_patients_wo_consent.py -data_path $input_path -pid_excl_file $pid_excl_file -output_file $output_file


# id_path="${hirid2_path}/research/misc_derived/id_lists"
# data_path="${hirid2_path}/pg_db_export/"
# pids_file="${hirid2_path}/research/misc_derived/id_lists/PID_WithConsent_201103.csv"
# bsub -J step0c -R "rusage[mem=4096]" python step0c_filter_patients.py -id_path $id_path -data_path $data_path -consent_pids_file $pids_file -num_chunk 100 -output_file $chunking_info_file


#####################
##  fix datetime
#####################
# for tbl in  monvals dervals observrec labres pharmarec
# do
# for i in $(seq 0 99)
# do
# input_path="${hirid2_path}/pg_db_export/"
# output_path="${hirid2_path}/research/1a_hdf5_clean/v8/datetime_fixed"
# bsub -G ms_raets -J step1a_1 -R "rusage[mem=8192]" python step1a_1_fix_datetimes.py -tbl $tbl -input_path $input_path -output_path $output_path -chunking_info_file $chunking_info_file -batch_id $i --id_path $id_path  --write_to_disk
# sleep 0.005
# done
# done


#####################
## Remove OOR value
#####################
for tbl in labres
 # observrec monvals dervals
do
for i in $(seq 0 49)
do
input_path="${hirid1_path}/1a_hdf5_clean/v6b/datetime_fixed"
output_path="${hirid1_path}/1a_hdf5_clean/v6b/oor_removed"
voi_path="${hirid1_path}/misc_derived/ref_excel"
version="v5b"
bsub -G ms_raets -J step1a_2 -R "rusage[mem=8192]" python step1a_2_remove_oor_values.py -tbl $tbl -input_path $input_path -output_path $output_path -voi_path $voi_path -chunking_info_file $chunking_info_file -batch_id $i -version $version 
# --write_to_disk
sleep 0.005
done
done


#####################
##  Compute std
#####################
# for tbl in observrec dervals pharamrec  labres
# do
# input_path="${hirid2_path}/research/1a_hdf5_clean/v8/oor_removed"
# output_path="${hirid2_path}/research/misc_derived/global_mean_std/v8"
# voi_path="${hirid2_path}/research/misc_derived/ref_excel"
# version="v8_temp"
# bsub -G ms_raets -J step1a_3 -R "rusage[mem=8192]" python step1a_3_compute_global_std.py -tbl $tbl -input_path $input_path -output_path $output_path -voi_path $voi_path -version $version
# sleep 0.005
# done

# for i in $(seq 0 67)
# do
# input_path="${hirid2_path}/research/1a_hdf5_clean/201103/oor_removed"
# output_path="${hirid2_path}/research/misc_derived/global_mean_std/201103"
# voi_path="${hirid2_path}/research/misc_derived/ref_excel"
# version="v8_temp"
# bsub -G ms_raets -J step1a_3 -R "rusage[mem=8192]" python step1a_3_compute_global_std.py -tbl monvals -input_path $input_path -output_path $output_path -voi_path $voi_path -version $version --idx_vid $i
# sleep 0.005
# done


#####################
## Remove duplicates
#####################
# for tbl in observrec
# # monvals dervals observrec labres
# do
# for i in $(seq 0 99)
# do
# input_path="${hirid2_path}/research/1a_hdf5_clean/v8/oor_removed"
# output_path="${hirid2_path}/research/1a_hdf5_clean/v8/duplicates_removed"
# std_path="${hirid2_path}/research/misc_derived/global_mean_std/v8"
# voi_path="${hirid2_path}/research/misc_derived/ref_excel"
# version="v8_temp"
# bsub -G ms_raets -J step1a_4 -R "rusage[mem=8192]" python step1a_4_remove_duplicates.py -tbl $tbl -input_path $input_path -output_path $output_path -std_path $std_path -voi_path $voi_path -chunking_info_file $chunking_info_file -batch_id $i -version $version --write_to_disk
# sleep 0.005
# done
# done

# for i in $(seq 0 99)
# do
# input_path="${hirid2_path}/research/1a_hdf5_clean/201103/datetime_fixed"
# output_path="${hirid2_path}/research/1a_hdf5_clean/201103/duplicates_removed"
# std_path="${hirid2_path}/research/misc_derived/global_mean_std/201103"
# voi_path="${hirid2_path}/research/misc_derived/ref_excel"
# version="v8_temp"
# bsub -G ms_raets -J step1a_4 -R "rusage[mem=8192]" python step1a_4_remove_duplicates.py -tbl pharmarec -input_path $input_path -output_path $output_path -std_path $std_path -voi_path $voi_path -chunking_info_file $chunking_info_file -batch_id $i -version $version --write_to_disk
# sleep 0.005
# done


#####################
##  Static
#####################
# hdf_path="${hirid2_path}/research/1a_hdf5_clean/v8/datetime_fixed"
# height_path="${hirid2_path}/research/1a_hdf5_clean/v8/oor_removed/height"
# generaldata_path="${hirid2_path}/pg_db_export/p_generaldata"
# # bsub -G ms_raets -J step1b -R "rusage[mem=8192]" 
# python step1b_extract_static_info.py -hdf_path $hdf_path -generaldata_path $generaldata_path -chunking_info_file $chunking_info_file -height_path $height_path

#####################
##  Pivot
#####################
# for tbl in observrec
# # monvals dervals pharamrec observrec labres
# do
# for i in $(seq 0 99)
# do
# input_path="${hirid2_path}/research/1a_hdf5_clean/v8/duplicates_removed"
# output_path="${hirid2_path}/research/2_pivoted/v8"
# voi_path="${hirid2_path}/research/misc_derived/ref_excel"
# version="v8_temp"
# bsub -G ms_raets -J step2 -R "rusage[mem=8192]" python step2_pivot_to_feature_mats.py -tbl $tbl -input_path $input_path -output_path $output_path -voi_path $voi_path -chunking_info_file $chunking_info_file -batch_id $i -version $version --write_to_disk
# sleep 0.005
# done
# done


#####################
##  Merge
#####################
# for i in $(seq 0 99)
# do
# input_path="${hirid2_path}/research/2_pivoted/v8"
# output_path="${hirid2_path}/research/3_merged/v8_tmp"
# urine_sum_path="${hirid2_path}/research/1a_hdf5_clean/v8/oor_removed/urine_sum"
# version="v8_temp"
# bsub -G ms_raets -J step3 -R "rusage[mem=8192]" python step3_merge_feature_mats.py -input_path $input_path -output_path $output_path -urine_sum_path $urine_sum_path -chunking_info_file $chunking_info_file -batch_id $i --write_to_disk
# sleep 0.005
# done

# for i in $(seq 1 89)
# do
# bsub -R "rusage[mem=8192]" -G ms_raets -J los python general_patient_statics.py $i
# done


# for i in $(seq 0 23)
# do
# bsub -R "rusage[mem=8192]" -R "rusage[mem=10240]" python step3b_remove_drug_oor.py -merged_path "/cluster/work/grlab/clinical/umcdb/preprocessed/merged/2021-11-10" -batch_id $i --dataset umc
# done
