import nibabel as nib
import pydicom as py
import numpy as np

data = nib.load("/home/lluis/msxplain/MSReport-generator/data/DICOMS/4031-5900/20220705/final_results/flair_3d.nii.gz")
data_nii = nib.load("/home/lluis/msxplain/MSReport-generator/data/DICOMS/4031-5900/20220705/final_results/t1n_3d.nii.gz")


# 192x240x256
# /home/msxplain/Report_generation/DICOMS
 
print(np.shape(data.get_fdata()))

print(np.shape(data_nii.get_fdata()))
