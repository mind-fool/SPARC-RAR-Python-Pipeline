##SPARC-RAR-Python-Pipeline

A Python pipeline for analyzing the SPARC galaxy dataset to reproduce the Radial Acceleration Relation (RAR), including data preprocessing, uncertainty propagation, nonlinear fitting, and rotation curve visualization.

##Overview

This repository provides a complete workflow to process SPARC galaxy rotation curve data, compute observed and baryonic accelerations, filter high-error points, visualize rotation curves, plot the Radial Acceleration Relation (RAR), fit Burkert halo profiles, and generate a summary report including the best-fit acceleration scale g0. The project has recently been updated to streamline the structure: several files were renamed, removed, or merged to reduce redundancy and improve clarity. The main modules now include report_converter.py, data_preprocess.py, filter_high_error_data.py, plot_rotation_curve.py, visualize_rotation_rar.py, burkert_halo_fit.py, best_fit_calculation.py, rar_summary.py, and main.py as the pipeline controller.

##Features

* Convert SPARC RTF reports to text and CSV formats (report_converter.py)

* Preprocess SPARC dataset and compute observed and baryonic accelerations (data_preprocess.py)

* Apply quality cuts to filter unreliable data points (filter_high_error_data.py)

* Plot rotation curves for individual galaxies (plot_rotation_curve.py)

* Plot the full Radial Acceleration Relation (visualize_rotation_rar.py)

* Fit Burkert halo profiles to selected galaxies (burkert_halo_fit.py)

* Compute the best-fit RAR acceleration scale g0 (best_fit_calculation.py)

* Generate a complete summary report including galaxy count, total points, and g0 (rar_summary.py)

##Installation

* Clone the repository: git clone https://github.com/mind-fool/SPARC-RAR-Python-Pipeline.git
 and navigate to the project folder.

* Create and activate a Python virtual environment: python3 -m venv ven and then source ven/bin/activate (Linux/macOS) or ven\Scripts\activate (Windows).

* Install the required packages: pip install -r requirements.txt.

##Usage

* Run the full pipeline using main.py, which handles all steps automatically including data conversion, preprocessing, filtering, plotting, halo fitting, g0 computation, and summary report generation.

* Example outputs include rotation curve plots and RAR plots for galaxies such as UGC02953, showing baryonic versus observed accelerations for the full dataset.

* Users can input specific galaxy IDs to generate individual rotation curve plots and halo fits.

##Future Work

* Expand the dataset to include more galaxies.

* Incorporate alternative gravity models such as MOND or conformal gravity.

* Automate report generation with additional visualizations and statistics.

* Extend the pipeline to support other halo models and batch processing.

##License

*  This project is released under the MIT License.
