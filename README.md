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
![Alt text](./samseg.png?raw=true "Optional Title")

## Code Contributors

This work is part of the project MSxplain.

## Author

👤 **Federico Spagnolo**

- Github: [@federicospagnolo](https://github.com/federicospagnolo)

## References
1. Fast and sequence-adaptive whole-brain segmentation using parametric Bayesian modeling. O. Puonti, J.E. Iglesias, K. Van Leemput. NeuroImage, 143, 235-249, 2016.
