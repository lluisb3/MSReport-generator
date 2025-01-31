import multiprocessing
import os
import shutil
import subprocess
import re
import glob
import argparse
from pathlib import Path

app_directory = Path(__file__).parent.absolute()

# Set FSLDIR and  FREESURFER and ANTs PATH
fsl_dir = "/home/lluis/msxplain/fsl"
os.environ["FSLDIR"] = fsl_dir
os.environ["PATH"] += os.pathsep + os.path.join(fsl_dir, "bin")

freesurfer_home = "/home/lluis/msxplain/freesurfer_7.4.1"
os.environ["FREESURFER_HOME"] = freesurfer_home
os.environ["PATH"] += os.pathsep + os.path.join(freesurfer_home, "bin")

ants_dir = "/home/lluis/msxplain/ANTs/install"
os.environ["ANTsDIR"] = ants_dir
os.environ["PATH"] += os.pathsep + os.path.join(ants_dir, "bin")

# Functions to run CMD comnands
def dcm2nixx(path_input, path_output, filename):
    subprocess.run(
        [
            "dcm2niix",
            "-o",
            path_output,
            "-f",
            filename,
            path_input,
        ]
    )

def n4biasfieldcorrection(path, path_sb):
    subprocess.run(
        [
            "N4BiasFieldCorrection",
            "-i",
            path,
            "-o",
            path_sb,
        ]
    )


def fslorient(path):
    subprocess.run(["fslorient", "-copysform2qform", path])

def fslreorient2std(path, path_s):
    subprocess.run(["fslreorient2std", path, path_s])



        # Additional parameter used for registration
registration_parameters_path = ("configs/Parameters_Rigid.txt"
        )

# Load data path
parser = argparse.ArgumentParser(
description='''Convert DICOM to NIFTI and perform the registration of the Flair image to 
the Mprage for a single patient. Both images needed on the folder''',
add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', metavar='-d', dest='data_path', help="Path where the data is")

args = parser.parse_args()        

data_path = Path(args.data_path)

for image_folder in os.listdir(data_path):
    image_path = os.path.join(data_path, image_folder)

    if os.path.isdir(image_path): # Ensure it is a folder
        if "FLAIR" in image_folder.upper():
            flair_name = image_folder
            flair_dicom = image_path
        
        elif "MPRAGE" or "T1N" in image_folder.upper():
            mprage_name = image_folder
            mprage_dicom = image_path

            
# Convert to NIfTI
print("============= Convert DICOM to Nifti =============")
nii_files_output = (Path(data_path) / "nii_files")
Path(nii_files_output).mkdir(parents=True, exist_ok=True)

# dcm2nixx
dcm2nixx(flair_dicom, nii_files_output, flair_name)
dcm2nixx(mprage_dicom, nii_files_output, mprage_name)


# Registration

# Locate new .nii files
flair_nii = (data_path / f"nii_files/{flair_name}.nii")
mprage_nii = (data_path / f"nii_files/{mprage_name}.nii")

# Create Outputs directory
reg_out_dir = (data_path / f"outputs")
reg_out_dir_elastix = (reg_out_dir / "elastix")
Path(reg_out_dir_elastix).mkdir(parents=True, exist_ok=True)

# Create intermediate files needed by FSL and elastix
reg_result_path = f"{reg_out_dir_elastix}/result.0.nii.gz"
flair_nii_sbr = f"{reg_out_dir}/{flair_name}_registered.nii.gz"
flair_nii_s = f"{reg_out_dir}/{flair_name}_s.nii.gz"
flair_nii_sb = f"{reg_out_dir}/{flair_name}_sb.nii.gz"
mprage_nii_sb = f"{reg_out_dir}/{mprage_name}_sb.nii.gz"
mprage_nii_s = f"{reg_out_dir}/{mprage_name}_s.nii.gz"

# Apply pre-processing steps

# FSLOrient
print("============= FSLOrient =============")
fslorient(flair_nii)
fslorient(mprage_nii)

# FSL2Reorient2Std
print("============= FSL2Reorient2Std =============")
fslreorient2std(flair_nii, flair_nii_s)
fslreorient2std(mprage_nii, mprage_nii_s)

# N4BiasFieldCorrection from ANTs
print("============= N4BiasFieldCorrection =============")
n4biasfieldcorrection(flair_nii_s, flair_nii_sb)
n4biasfieldcorrection(mprage_nii_s, mprage_nii_sb)

# Elastix Registration
print("============= Elastix Registration =============")
subprocess.run(
    [
        "elastix",
        "-f",
        mprage_nii_sb,
        "-m",
        flair_nii_sb,
        "-out",
        reg_out_dir_elastix,
        "-p",
        registration_parameters_path,
    ]
)

# FSLCpGeom
shutil.move(reg_result_path, flair_nii_sbr)
subprocess.run(["fslcpgeom", mprage_nii_sb, flair_nii_sbr])

# Move important outputs to final directory
final_results_dir = (data_path / "final_results")
Path(final_results_dir).mkdir(parents=True, exist_ok=True)

print(f"============= Save finals results in {final_results_dir} =============")
shutil.move(flair_nii_sbr, (final_results_dir / f"flair.nii.gz"))
shutil.move(mprage_nii_sb, (final_results_dir / f"mprage.nii.gz"))

# Remove intermediate files
# Could be interesting to remove the nii_files and outputs directories as they create intermediate files
