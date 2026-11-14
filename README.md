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
the NFW radius-limit sweep, the stellar mass-to-light sweep, and an optimizer
starting-point check. Choose **5**
to run all analyses. The pipeline prepares the input CSV if needed and saves
the results under `outputs/`.

The RAR plot, fitted `g0`, and RAR summary use the same relative-error cut.
The halo fits use the preprocessed rotation curves before that cut.
The two sweeps save the CSVs and manuscript outputs in `outputs/sweeps/`.

* Option **3** saves `nfw_bound_sweep.csv`, `nfw_bound_sweep_table_rows.tex`,
  and `ngc0100_nfw_bound_sweep.png`.
* Option **4** saves `mass_to_light_sweep.csv` and `mass_to_light_sweep.png`.
* Option **5** generates these five files and runs the optimizer check.
* Option **6** runs only the optimizer check, leaving existing sweep outputs untouched.
* Option **7** runs the separate full-sample halo-bound sensitivity study. It is not
  included in option **5**, because it deliberately performs many more fits.

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

### Selecting halo fits

Galaxy plots and both halo sweeps select the lowest total chi-squared among the
ten starting-point trials described below. The same selected fit supplies the
reported parameters, model curve, residuals, and comparison score. Plots do not
rerun the optimizer. Individual halo scripts use this selection too.

The sweep CSVs record the selection strategy, the chosen trial, successful and
failed trial counts, and the number agreeing with the lowest score. The
`burkert_selection_trials_json` and `nfw_selection_trials_json` columns retain
all trial diagnostics as JSON records, including failures and warnings.
The existing bound flags describe the selected trial; they do not establish
that a parameter is physically well constrained.

The validation workflow still compares the original single-start result with
the best among the tested starts. Option **5** runs both the production sweeps
and this separate validation, so it repeats fits and takes longer. Use options
**3** or **4** to regenerate just the corresponding sweep.

### Check optimizer starting points

Run `main.py` and choose **6**. Enter galaxy IDs separated by commas, or press
Enter to check all galaxies. For a short first run, enter `NGC0100,UGC01281,NGC6195`.
Option **5** also runs this check for all galaxies after the other analyses.

Each fit is repeated from ten starting points: the existing default and nine
combinations of density and radius. The nine starts use the 10%, 50%, and 90%
positions within each logarithmic parameter interval. The density interval is
`1e-5` to `10 Msun/pc^3`; the radius interval is `0.01 kpc` to the selected ceiling.
The check covers Burkert at 100 kpc and NFW at 100, 300, and 1,000 kpc with the
baseline stellar ratios, plus both models at 100 kpc for the two other stellar
settings. This gives 14,000 trial fits for 175 galaxies and may take several minutes.

Each run creates a new dated folder under `outputs/validation/` containing:

* `optimizer_trials.csv`: every requested and effective starting point, fitted
  parameters, chi-squared, optimizer status, boundary flags, warnings, and failures.
* `optimizer_summary.csv`: the default and best-found fit for each case, trial
  counts, agreement counts, and the parameter ranges among agreeing fits.
* `optimizer_comparisons.csv`: the default and best-found Burkert/NFW rankings.
* `manifest.json`: the input and source hashes, code revision, software versions,
  starting points, settings, tolerances, and whether the run completed.

Successful trials agree with the best-found fit when their total chi-squared
values differ by no more than `1e-6 + 1e-6 * abs(best_chi_squared)`. This is a
numerical tolerance, not a statistical significance threshold. Model ranking
uses the same reduced-chi-squared near-tie rule as the existing sweeps.
A failed default fit is recorded separately from an improved fit. A run interrupted
between configurations can leave partial CSVs; its manifest will still say `running`.

Signed baryonic contributions can require moving a starting point into the region
where total squared speed is positive. Both the requested and effective starts are
saved, and the summary counts distinct effective starts among successful trials.
Agreement across these starts does not prove that the global minimum was found.
The check does not automatically replace the original sweep results.

### Test halo-fit boundaries

Run `main.py` and choose **7** to test the numerical boundaries without changing
the original paper sweep. The study changes one bound family at a time, while the
other limits retain their baseline values:

* scale-radius ceilings of 10, 30, 100, 300, 1,000, 3,000, and 10,000 kpc;
* density floors of `1e-8`, `1e-7`, `1e-6`, `1e-5`, and `1e-4 Msun/pc^3`;
* density ceilings of 10, 30, 100, 300, and `1e3 Msun/pc^3`;
* scale-radius floors of 0.001, 0.01, and 0.1 kpc.

Duplicate baseline configurations are run once, giving 17 configurations in total.
Both halo models use the same bound configuration in each matched comparison.
The study uses the same ten-start selection as the rest of the pipeline. It saves
a dated folder under `outputs/boundary_sensitivity/` containing every fit,
Burkert--NFW comparisons, configuration summaries, and a manifest with the source
revision, input hash, software versions, and exact configurations. This is a
numerical-sensitivity study: it does not establish a global optimum or identify a
physically preferred model on its own.

### Tests

Run the full test suite from the repository folder.

```bash
python3 -m unittest discover -s tests -v
```

### Future Work
* Validate a conformal-gravity implementation against a reference rotation curve, then compare it with the halo fits using the same SPARC measurements.

### License

*  This project is released under the MIT License.
