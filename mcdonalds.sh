# Script to obtain automatic report using SAMSEG
FILES="/home/federicospagnolo/storage/groups/think/Federico/Report_generation/SAMSEG/*"

for f in $FILES
do
	if [[ -f $f/segtoFLAIR.nii.gz ]] 
then
	#mri_convert $f/seg.mgz $f/segtoFLAIR.nii.gz
	
	#obtain mask of each structure
	fslmaths $f/segtoFLAIR.nii.gz -thr 1.5 -uthr 2.5 $f/LeftWM.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 2.5 -uthr 3.5 $f/LeftCerebralCortex.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 3.5 -uthr 4.5 $f/LeftLateralVentricle.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 4.5 -uthr 5.5 $f/LeftInfLateralVentricle.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 6.5 -uthr 7.5 $f/LeftCerebellumWM.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 7.5 -uthr 8.5 $f/LeftCerebellumCortex.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 9.5 -uthr 10.5 $f/LeftThalamus.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 10.5 -uthr 11.5 $f/LeftCaudate.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 11.5 -uthr 12.5 $f/LeftPutamen.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 12.5 -uthr 13.5 $f/LeftPallidum.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 13.5 -uthr 14.5 $f/ThirdVentricle.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 14.5 -uthr 15.5 $f/FourthVentricle.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 15.5 -uthr 16.5 $f/Brainstem.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 16.5 -uthr 17.5 $f/LeftHippocampus.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 17.5 -uthr 18.5 $f/LeftAmygdala.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 25.5 -uthr 26.5 $f/LeftAccumbens.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 27.5 -uthr 28.5 $f/LeftVentralDC.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 28.5 -uthr 29.5 $f/LeftUndetermined.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 29.5 -uthr 30.5 $f/LeftVessel.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 30.5 -uthr 31.5 $f/LeftChoroidPlexus.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 40.5 -uthr 41.5 $f/RightWM.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 41.5 -uthr 42.5 $f/RightCerebralCortex.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 42.5 -uthr 43.5 $f/RightLateralVentricle.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 43.5 -uthr 44.5 $f/RightInfLateralVentricle.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 45.5 -uthr 46.5 $f/RightCerebellumWM.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 46.5 -uthr 47.5 $f/RightCerebellumCortex.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 48.5 -uthr 49.5 $f/RightThalamus.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 49.5 -uthr 50.5 $f/RightCaudate.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 50.5 -uthr 51.5 $f/RightPutamen.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 51.5 -uthr 52.5 $f/RightPallidum.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 52.5 -uthr 53.5 $f/RightHippocampus.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 53.5 -uthr 54.5 $f/RightAmygdala.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 57.5 -uthr 58.5 $f/RightAccumbens.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 59.5 -uthr 60.5 $f/RightVentralDC.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 61.5 -uthr 62.5 $f/RightVessel.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 62.5 -uthr 63.5 $f/RightChoroidPlexus.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 71.5 -uthr 72.5 $f/FifthVentricle.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 84.5 -uthr 85.5 $f/OpticChiasm.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 98.5 -uthr 99.5 $f/Lesion.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 164.5 -uthr 165.5 $f/Skull.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 257.5 -uthr 258.5 $f/HeadExtraCerebral.nii.gz
	fslmaths $f/segtoFLAIR.nii.gz -thr 258.5 -uthr 259.5 $f/EyeFluid.nii.gz
	
	#obtain WM mask
	fslmaths $f/LeftWM.nii.gz -add $f/RightWM.nii.gz -add $f/pred.nii.gz $f/WM_Mask.nii.gz
	fslmaths $f/WM_Mask.nii.gz -bin $f/WM_Mask.nii.gz 	
	
	#obtain cortex
	fslmaths $f/LeftCerebralCortex.nii.gz -add $f/RightCerebralCortex.nii.gz $f/CerebralCortex.nii.gz
	fslmaths $f/CerebralCortex.nii.gz -bin $f/CerebralCortex.nii.gz
	fslmaths $f/CerebralCortex.nii.gz -mul $f/pred.nii.gz $f/common.nii.gz 
	fslmaths $f/CerebralCortex.nii.gz -sub $f/common.nii.gz $f/Cortex.nii.gz
	
	#obtain ventricles
	fslmaths $f/LeftLateralVentricle.nii.gz -add $f/RightLateralVentricle.nii.gz $f/Ventricles.nii.gz
	fslmaths $f/Ventricles.nii.gz -bin $f/Ventricles.nii.gz
	
	#obtain infratentorial
	fslmaths $f/Brainstem.nii.gz -add $f/LeftCerebellumWM.nii.gz -add $f/RightCerebellumWM.nii.gz -add $f/RightCerebellumCortex.nii.gz -add $f/LeftCerebellumCortex.nii.gz $f/Infratentorial.nii.gz
	fslmaths $f/Infratentorial.nii.gz -bin $f/Infratentorial.nii.gz
	
	#remove masks of each structure
	rm $f/LeftWM.nii.gz
	rm $f/LeftCerebralCortex.nii.gz
	rm $f/LeftLateralVentricle.nii.gz
	rm $f/LeftInfLateralVentricle.nii.gz
	rm $f/LeftCerebellumWM.nii.gz
	rm $f/LeftCerebellumCortex.nii.gz
	rm $f/LeftThalamus.nii.gz
	rm $f/LeftCaudate.nii.gz
	rm $f/LeftPutamen.nii.gz
	rm $f/LeftPallidum.nii.gz
	rm $f/ThirdVentricle.nii.gz
	rm $f/FourthVentricle.nii.gz
	rm $f/Brainstem.nii.gz
	rm $f/LeftHippocampus.nii.gz
	rm $f/LeftAmygdala.nii.gz
	rm $f/LeftAccumbens.nii.gz
	rm $f/LeftVentralDC.nii.gz
	rm $f/LeftUndetermined.nii.gz
	rm $f/LeftVessel.nii.gz
	rm $f/LeftChoroidPlexus.nii.gz
	rm $f/RightWM.nii.gz
	rm $f/RightCerebralCortex.nii.gz
	rm $f/RightLateralVentricle.nii.gz
	rm $f/RightInfLateralVentricle.nii.gz
	rm $f/RightCerebellumWM.nii.gz
	rm $f/RightCerebellumCortex.nii.gz
	rm $f/RightThalamus.nii.gz
	rm $f/RightCaudate.nii.gz
	rm $f/RightPutamen.nii.gz
	rm $f/RightPallidum.nii.gz
	rm $f/RightHippocampus.nii.gz
	rm $f/RightAmygdala.nii.gz
	rm $f/RightAccumbens.nii.gz
	rm $f/RightVentralDC.nii.gz
	rm $f/RightVessel.nii.gz
	rm $f/RightChoroidPlexus.nii.gz
	rm $f/FifthVentricle.nii.gz
	rm $f/OpticChiasm.nii.gz
	rm $f/Lesion.nii.gz
	rm $f/Skull.nii.gz
	rm $f/HeadExtraCerebral.nii.gz
	rm $f/EyeFluid.nii.gz
	rm $f/common.nii.gz
	
	fi
	
	python lesion_information.py report $f/flair_3d_sbr.nii.gz $f/pred.nii.gz $f
done
