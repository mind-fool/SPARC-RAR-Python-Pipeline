### SPARC-RAR-Python-Pipeline

A Python pipeline for reproducing the radial acceleration relation (RAR) and comparing galaxy rotation-curve models using public SPARC data.

### Overview

The RAR workflow computes observed and baryonic accelerations and estimates the characteristic acceleration scale `g0`. The rotation-curve workflow compares baryons plus a Burkert halo with baryons plus an NFW halo using the same measured velocities and quoted uncertainties. It plots both fitted curves and their residuals against radius.

The rotation curves and baryonic components come from the [SPARC Newtonian mass-model table](https://astroweb.case.edu/SPARC/MassModels_Lelli2016c.mrt).

### Features

* Convert SPARC RTF reports to text and CSV formats (report_converter.py)

* Preprocess SPARC dataset and compute observed and baryonic accelerations (data_preprocess.py)

* Apply quality cuts to filter unreliable data points (filter_high_error_data.py)

* Plot rotation curves for individual galaxies (plot_rotation_curve.py)

* Plot the full Radial Acceleration Relation (visualize_rotation_rar.py)

* Fit baryons plus a Burkert or NFW halo and flag fitted parameters near their bounds (`halo_models.py`, `burkert_halo_fit.py`).

* Compare both models on the same galaxy with a rotation-curve plot and a residual panel (`compare_halo_fits.py`).

* Verify the halo equations by numerical integration and test parameter recovery on synthetic rotation curves (`tests/`).

* Compute the best-fit RAR acceleration scale g0 (best_fit_calculation.py)

* Generate a complete summary report including galaxy count, total points, and g0 (rar_summary.py)

##Installation

* Clone the repository: git clone https://github.com/mind-fool/SPARC-RAR-Python-Pipeline.git
 and navigate to the project folder.

* Create and activate a Python virtual environment: python3 -m venv ven and then source ven/bin/activate (Linux/macOS) or ven\Scripts\activate (Windows).

* Install the required packages: pip install -r requirements.txt.

##Usage

Run `python3 main.py` from the project folder, or open `main.py` and press Run.
The menu offers the RAR analysis, selected-galaxy rotation curves and halo fits,
the NFW radius-limit sweep, and the stellar mass-to-light sweep. Choose **5**
to run all analyses. The pipeline prepares the input CSV if needed and saves
the results under `outputs/`.

The RAR plot, fitted `g0`, and RAR summary use the same relative-error cut.
The halo fits use the preprocessed rotation curves before that cut.
The two sweeps save the CSVs and manuscript outputs in `outputs/sweeps/`.

* Option **3** saves `nfw_bound_sweep.csv`, `nfw_bound_sweep_table_rows.tex`,
  and `ngc0100_nfw_bound_sweep.png`.
* Option **4** saves `mass_to_light_sweep.csv` and `mass_to_light_sweep.png`.
* Option **5** generates all five files in one run.

### Compare Burkert and NFW

From the repository folder with the virtual environment activated, generate
the CSV from the bundled SPARC data and run a benchmark.

```bash
python3 report_converter.py
python3 compare_halo_fits.py NGC0100
```

Generate the CSV once per checkout. Run `python3 compare_halo_fits.py` to enter
a galaxy ID interactively. The script saves the figure as
`<galaxy_id>_halo_comparison.png`.

#### Change the NFW scale-radius limit

The default upper limit on the fitted scale radius is 100 kpc. Use `--max-scale-radius-kpc` to set a different limit for a single run.

```bash
# Fit NFW with the default 100 kpc limit
python3 burkert_halo_fit.py NGC0100 --model nfw

# Repeat with a 300 kpc limit
python3 burkert_halo_fit.py NGC0100 --model nfw --max-scale-radius-kpc 300
```

Running `compare_halo_fits.py` from the editor prompts for a galaxy ID and uses the default limit for both fits. Run `python3 burkert_halo_fit.py --help` to see the individual fitter's options.

### Tests

Run the full test suite from the repository folder.

```bash
python3 -m unittest discover -s tests -v
```

### Future Work
* Validate a conformal-gravity implementation against a reference rotation curve, then compare it with the halo fits using the same SPARC measurements.

### License

*  This project is released under the MIT License.
