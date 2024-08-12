<h1 align="center">MSReport-generator </h1>

> Scripts to generate a structured report for MS patients in the form of a `.xlsx` table. Such table contains useful per-patient information, such as the total number of MS lesions, their location, center of mass and volume from the MRI.
> <br /> `MSReport-generator` can provide clinical details in line with the updated McDonald criteria (periventricular, infratentorial, juxtacortical and white matter lesion location), as well as detect false positive examples located outside of the brain.
> The pipeline is based on an automatic lesion segmentation method, brain parcellation using SAMSEG, and FreeSurfer.

## 🚀 Usage

First, make sure you have python >=3.9 and FreeSurfer 7.4.1 installed.

To build the environment, an installation of conda or miniconda is needed. Once you have it, please use
```sh
conda env create -f environment.yml
```
to build the tested environment using the provided `environment.yml` file. 

The script `report.sh` computes all the steps (automatic segmentation, brain parcellation, and report generation) and saves the report as a file `report.xlsx`.
The environment variables should be changed according to the user's local machine.
Usage is the following:
```sh
sudo ./report.sh {PATH_TO_DATA}
```

<figure style="display: inline-block; margin-right: 10px;">
  <img src="samseg.png" alt="Image 1" width="300">
  <figcaption>SAMSEG parcellation</figcaption>
</figure>

<figure style="display: inline-block;">
  <img src="T1w_ventr_cort.png" alt="Image 2" width="300">
  <figcaption>Segmentation of ventricles and cortex</figcaption>
</figure>

## Code Contributors

This work is part of the project MSxplain.

## Author

👤 **Federico Spagnolo**

- Github: [@federicospagnolo](https://github.com/federicospagnolo)

## References
1. Fast and sequence-adaptive whole-brain segmentation using parametric Bayesian modeling. O. Puonti, J.E. Iglesias, K. Van Leemput. NeuroImage, 143, 235-249, 2016.
