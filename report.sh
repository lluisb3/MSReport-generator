#!/bin/bash

# Record the start time
start_time=$(date +%s)

# Script to obtain automatic report using SAMSEG
# Usage: sudo ./report.sh /home/lluis/MSReport_generator/data
FSLDIR=/home/lluis/msxplain/fsl # Path to the installation FSL
PATH=${FSLDIR}/bin:${PATH} # Created when installed
. ${FSLDIR}/etc/fslconf/fsl.sh # Created when installed
FREESURFER_HOME=/home/lluis/msxplain/freesurfer # Path to the installation Freesurfer
. $FREESURFER_HOME/SetUpFreeSurfer.sh
CONDA_PATH=/home/lluis/miniconda3
. $CONDA_PATH/etc/profile.d/conda.sh
conda activate msreport

# File inputs
FLAIR=flair.nii.gz
MPRAGE=mprage.nii.gz

python /home/lluis/msxplain/MSReport-generator/predict.py --model_checkpoint /home/lluis/msxplain/MSReport-generator/model_epoch_31.pth --input_val_paths $1 $1 --input_prefixes $FLAIR $MPRAGE --num_workers 0 --cache_rate 0.01 --threshold 0.3
echo "Prediction file saved"

FILES="$1/$MPRAGE"

for f_t1 in $FILES; do
       	f=$(dirname $f_t1)
       	echo "Processing directory: $f"
        #relative_path=$(echo "$f" | awk -F '/home/lluis/' '{print $2}')
	echo $f_t1
	run_samseg --input $f/$MPRAGE --output $f/SAMSEG --threads 2 > /dev/null
        echo "echo1"
        mri_convert $f/SAMSEG/seg.mgz $f/SAMSEG/seg.nii.gz
	echo "echo2"
	#docker run -v /home/lluis:/root a63c687a06d9 run_samseg --input $relative_path/t1n_3d_s.nii.gz --output $relative_path/SAMSEG --threads 2
	#run_samseg --input $f/t1n_3d_s.nii.gz --output $f/SAMSEG --threads 2
	#run_samseg --input $f/t1n_3d_s.nii.gz --output $f/SAMSEG --threads 2 > /dev/null
	#docker run -v /home/lluis:/root a63c687a06d9 mri_convert $relative_path/SAMSEG/seg.mgz $relative_path/SAMSEG/seg.nii.gz
	#mri_convert $f/SAMSEG/seg.mgz $f/SAMSEG/seg.nii.gz
	#obtain mask of each structure
	fslmaths $f/SAMSEG/seg.nii.gz -thr 1.5 -uthr 2.5 $f/SAMSEG/LeftWM.nii.gz
	fslmaths $f/SAMSEG/seg.nii.gz -thr 2.5 -uthr 3.5 $f/SAMSEG/LeftCerebralCortex.nii.gz
	fslmaths $f/SAMSEG/seg.nii.gz -thr 3.5 -uthr 4.5 $f/SAMSEG/LeftLateralVentricle.nii.gz
	fslmaths $f/SAMSEG/seg.nii.gz -thr 6.5 -uthr 7.5 $f/SAMSEG/LeftCerebellumWM.nii.gz
	fslmaths $f/SAMSEG/seg.nii.gz -thr 7.5 -uthr 8.5 $f/SAMSEG/LeftCerebellumCortex.nii.gz
	fslmaths $f/SAMSEG/seg.nii.gz -thr 15.5 -uthr 16.5 $f/SAMSEG/Brainstem.nii.gz
	fslmaths $f/SAMSEG/seg.nii.gz -thr 40.5 -uthr 41.5 $f/SAMSEG/RightWM.nii.gz
	fslmaths $f/SAMSEG/seg.nii.gz -thr 41.5 -uthr 42.5 $f/SAMSEG/RightCerebralCortex.nii.gz
	fslmaths $f/SAMSEG/seg.nii.gz -thr 42.5 -uthr 43.5 $f/SAMSEG/RightLateralVentricle.nii.gz
	fslmaths $f/SAMSEG/seg.nii.gz -thr 45.5 -uthr 46.5 $f/SAMSEG/RightCerebellumWM.nii.gz
	fslmaths $f/SAMSEG/seg.nii.gz -thr 46.5 -uthr 47.5 $f/SAMSEG/RightCerebellumCortex.nii.gz
	
	#obtain WM mask
	fslmaths $f/SAMSEG/LeftWM.nii.gz -add $f/SAMSEG/RightWM.nii.gz $f/SAMSEG/WM_Mask.nii.gz
	fslmaths $f/SAMSEG/WM_Mask.nii.gz -bin $f/SAMSEG/WM_Mask.nii.gz 	
	
	#obtain cortex
	fslmaths $f/SAMSEG/LeftCerebralCortex.nii.gz -add $f/SAMSEG/RightCerebralCortex.nii.gz $f/SAMSEG/CerebralCortex.nii.gz
	fslmaths $f/SAMSEG/CerebralCortex.nii.gz -bin $f/SAMSEG/CerebralCortex.nii.gz
	fslmaths $f/SAMSEG/CerebralCortex.nii.gz -mul $f/SAMSEG/pred.nii.gz $f/SAMSEG/common.nii.gz 
	fslmaths $f/SAMSEG/CerebralCortex.nii.gz -sub $f/SAMSEG/common.nii.gz $f/SAMSEG/Cortex.nii.gz
	
	#obtain ventricles
	fslmaths $f/SAMSEG/LeftLateralVentricle.nii.gz -add $f/SAMSEG/RightLateralVentricle.nii.gz $f/SAMSEG/LateralVentricles.nii.gz
	fslmaths $f/SAMSEG/LateralVentricles.nii.gz -bin $f/SAMSEG/LateralVentricles.nii.gz
	fslmaths $f/SAMSEG/LateralVentricles.nii.gz -mul $f/SAMSEG/pred.nii.gz $f/SAMSEG/common2.nii.gz
	fslmaths $f/SAMSEG/LateralVentricles.nii.gz -sub $f/SAMSEG/common2.nii.gz $f/SAMSEG/Ventricles.nii.gz
	
	#obtain infratentorial
	#fslmaths $f/SAMSEG/Brainstem.nii.gz -add $f/SAMSEG/LeftCerebellumWM.nii.gz -add $f/SAMSEG/RightCerebellumWM.nii.gz -add $f/SAMSEG/RightCerebellumCortex.nii.gz -add $f/SAMSEG/LeftCerebellumCortex.nii.gz $f/SAMSEG/Infratentorial.nii.gz
	fslmaths $f/SAMSEG/Brainstem.nii.gz -add $f/SAMSEG/LeftCerebellumWM.nii.gz -add $f/SAMSEG/RightCerebellumWM.nii.gz $f/SAMSEG/Infratentorial.nii.gz
	fslmaths $f/SAMSEG/Infratentorial.nii.gz -bin $f/SAMSEG/Infratentorial.nii.gz
	
	#remove masks of each structure
	#rm $f/SAMSEG/LeftWM.nii.gz
	#rm $f/SAMSEG/LeftCerebralCortex.nii.gz
	#rm $f/SAMSEG/LeftLateralVentricle.nii.gz
	#rm $f/SAMSEG/LeftInfLateralVentricle.nii.gz
	#rm $f/SAMSEG/LeftCerebellumWM.nii.gz
	#rm $f/SAMSEG/LeftCerebellumCortex.nii.gz
	#rm $f/SAMSEG/Brainstem.nii.gz
	#rm $f/SAMSEG/RightWM.nii.gz
	#rm $f/SAMSEG/RightCerebralCortex.nii.gz
	#rm $f/SAMSEG/RightLateralVentricle.nii.gz
	#rm $f/SAMSEG/RightInfLateralVentricle.nii.gz
	#rm $f/SAMSEG/RightCerebellumWM.nii.gz
	#rm $f/SAMSEG/RightCerebellumCortex.nii.gz
	rm $f/SAMSEG/common.nii.gz
	rm $f/SAMSEG/common2.nii.gz
	
	python /home/lluis/msxplain/MSReport-generator/lesion_information.py report $f/$FLAIR $f/SAMSEG/pred.nii.gz $f/SAMSEG
done

chown -R lluis:lluis $1

# Record the end time
end_time=$(date +%s)

# Calculate elapsed time
elapsed_time=$((end_time - start_time))

# Convert elapsed time to hours, minutes, and seconds
hours=$((elapsed_time / 3600))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$((elapsed_time % 60))

# Display the result
printf "====== Elapsed time: %02d hours, %02d minutes, %02d seconds\n" $hours $minutes $seconds
