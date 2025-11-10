# SPARC-RAR-Python-Pipeline
Python pipeline for analyzing the SPARC galaxy dataset to reproduce the Radial Acceleration Relation (RAR), including data preprocessing, uncertainty propagation, nonlinear fitting, and rotation curve visualization.

# SPARC RAR Analysis Pipeline
This repository contains a Python pipeline to process SPARC galaxy rotation curve data, visualize rotation curves and the Radial Acceleration Relation (RAR), apply quality cuts, fit Burkert halo profiles, and generate summary reports. The goal is to study observed vs baryonic accelerations in galaxies and test theoretical predictions like MOND or alternative gravity models.

## Features
- Convert SPARC RTF reports to text and CSV formats (`report_converter.py`).
- Preprocess SPARC dataset and compute observed and baryonic accelerations (`data_preprocess.py`).
- Plot observed and predicted rotation curves (`plot_rotation_curve.py`).
- Plot Radial Acceleration Relation (RAR) (`plot_rar.py`).
- Apply data quality cuts (`apply_quality_cut.py`).
- Fit Burkert halo profile to galaxy rotation curves (`fit_burkert.py`).
- Generate a summary report with number of galaxies, total points, and best-fit g0 (`generate_report.py`).

## Installation
Clone the repository: `git clone https://github.com/mind-fool/SPARC-RAR-Python-Pipeline.git` and `cd SPARC-RAR-Python-Pipeline`.  
Create and activate a virtual environment: `python3 -m venv ven` then `source ven/bin/activate`.  
Install required packages: `pip install -r requirements.txt`.

## Usage
- Preprocess Data: `python data_preprocess.py`  
- Convert SPARC Reports: `python report_converter.py`  
- Plot Rotation Curve for a galaxy: `python plot_rotation_curve.py`  
- Plot RAR for the dataset: `python plot_rar.py`  
- Apply Quality Cut: `python apply_quality_cut.py`  
- Fit Burkert Halo: `python fit_burkert.py`  
- Generate Summary Report: `python generate_report.py`

## Example
Rotation curve and RAR plots for UGC02953 are included in the outputs. The RAR plot compares baryonic accelerations (`g_bar`) to observed accelerations (`g_obs`) for the full dataset.

## Future Work
Expand the dataset to include more galaxies. Test alternative gravity models such as MOND or quantum gravity predictions. Improve report generation with more automated visualizations and statistics.

## License
This project is released under the MIT License.

