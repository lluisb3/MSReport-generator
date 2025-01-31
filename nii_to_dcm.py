import nibabel as nib
import pydicom
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, SecondaryCaptureImageStorage
import numpy as np
import os
from pathlib import Path

data_folder = Path("data/Patient01-1/final_results")

# Load NIfTI file
nifti_file = data_folder / "lesion_map.nii.gz" # Change to your file path
nii_img = nib.load(nifti_file)
data = nii_img.get_fdata()

# Create output folder for DICOM slices
output_folder = data_folder / "lesion_map_dcm"
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Convert each slice into a separate DICOM file
for i in range(data.shape[2]):
    dicom_slice = pydicom.Dataset()
    
    # Required metadata
    dicom_slice.is_little_endian = True
    dicom_slice.is_implicit_VR = False  # Explicit VR is recommended
    dicom_slice.PatientName = "Test Patient"
    dicom_slice.PatientID = "123456"
    dicom_slice.StudyInstanceUID = "1.2.826.0.1.3680043.10.511.3"
    dicom_slice.SeriesInstanceUID = "1.2.826.0.1.3680043.10.511.4"
    dicom_slice.SOPInstanceUID = f"1.2.826.0.1.3680043.10.511.4.{i}"
    dicom_slice.Modality = "MR"
    dicom_slice.Rows, dicom_slice.Columns = data.shape[:2]
    
    # Pixel data settings
    dicom_slice.SamplesPerPixel = 1
    dicom_slice.PhotometricInterpretation = "MONOCHROME2"
    dicom_slice.BitsAllocated = 16
    dicom_slice.BitsStored = 16
    dicom_slice.HighBit = 15
    dicom_slice.PixelRepresentation = 1
    dicom_slice.PixelData = data[:, :, i].astype(np.uint16).tobytes()
    
    # **Fix: Add required DICOM File Meta Information**
    dicom_slice.file_meta = pydicom.dataset.FileMetaDataset()
    dicom_slice.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage  # Required!
    dicom_slice.file_meta.MediaStorageSOPInstanceUID = dicom_slice.SOPInstanceUID
    dicom_slice.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    # Save DICOM slice
    dicom_slice.save_as(os.path.join(output_folder, f"slice_{i}.dcm"), write_like_original=False)

print(f"✅ Conversion complete! DICOM files saved in: {output_folder}")