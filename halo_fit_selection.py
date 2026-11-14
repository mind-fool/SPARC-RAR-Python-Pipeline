"""Select the lowest chi-squared among explicitly recorded starting-point trials."""

from contextlib import redirect_stdout
from io import StringIO
import warnings

import numpy as np

from burkert_halo_fit import fit_halo, plot_halo_fit, HALO_MODELS
from data_preprocess import load_sparc_dataset
from optimizer_starts import CHI2_ATOL, CHI2_RTOL, starting_points


def fit_best_halo(
    galaxy_id, df=None, output_path=None, *, model="burkert",
    max_scale_radius_kpc=100.0, min_scale_radius_kpc=0.01,
    density_bounds=(1e-5, 10.0), plot=False, show=True, report=False,
):
    """Return the best found fit and diagnostics, with optional reporting and plotting.

    This uses the same objective, physical bounds and ten starting points as
    the validation workflow. It does not establish a global minimum.
    """
    if model not in ("burkert", "nfw"):
        raise ValueError(f"Unknown halo model: {model!r}")
    starts = starting_points(
        max_scale_radius_kpc,
        min_scale_radius_kpc=min_scale_radius_kpc,
        density_bounds=density_bounds,
    )
    if df is None:
        df = load_sparc_dataset()
    best = None
    selected_trial = None
    trials = []
    for trial, start in enumerate(starts):
        diagnostic = {
            "trial": trial, "requested_initial_density_msun_pc3": start[0],
            "requested_initial_radius_kpc": start[1], "success": False,
            "fit_error": "", "warnings": "",
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with redirect_stdout(StringIO()):
                    fit = fit_halo(
                        galaxy_id, df=df, model=model, plot=False,
                        max_scale_radius_kpc=max_scale_radius_kpc,
                        min_scale_radius_kpc=min_scale_radius_kpc,
                        density_bounds=density_bounds, initial_guess=start,
                    )
                if fit["optimizer_status"] not in (1, 2, 3, 4) or not np.isfinite(fit["chi_squared"]):
                    raise ValueError("Optimizer did not return a successful finite fit.")
                if best is not None and not np.array_equal(fit["radius_kpc"], best["radius_kpc"]):
                    raise RuntimeError("Trials used different observed radii.")
                diagnostic.update({
                    key: value for key, value in fit.items()
                    if np.isscalar(value) or key == "bound_hits"
                })
                diagnostic["success"] = True
                if best is None or fit["chi_squared"] < best["chi_squared"]:
                    best, selected_trial = fit, trial
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                diagnostic["fit_error"] = str(exc) or type(exc).__name__
        diagnostic["warnings"] = " | ".join(str(item.message) for item in caught)
        trials.append(diagnostic)
    if best is None:
        details = "; ".join(f"trial {row['trial']}: {row['fit_error']}" for row in trials)
        raise RuntimeError(f"All {len(trials)} starting-point trials failed for {galaxy_id}: {details}")
    successful = [row for row in trials if row["success"]]
    tolerance = CHI2_ATOL + CHI2_RTOL * abs(best["chi_squared"])
    agreeing = sum(row["chi_squared"] - best["chi_squared"] <= tolerance for row in successful)
    # Preserve the chosen parameters and curves together; do not rerun the fit.
    result = dict(
        best, fit_strategy="best_of_tested_starts",
        selection_trial_count=len(trials), selection_success_count=len(successful),
        selection_failed_count=len(trials) - len(successful),
        selection_selected_trial=selected_trial, selection_agreement_count=agreeing,
        selection_trial_results=trials,
    )
    if report:
        _, label, density_name, radius_name = HALO_MODELS[model]
        print(f"{label} fit for {galaxy_id} (best of {len(successful)}/{len(trials)} successful starts)")
        print(f"{density_name} = {result[f'{density_name}_msun_pc3']:.6g} "
              f"± {result[f'{density_name}_error_msun_pc3']:.2g} Msun/pc^3")
        print(f"{radius_name} = {result[f'{radius_name}_kpc']:.6g} "
              f"± {result[f'{radius_name}_error_kpc']:.2g} kpc")
        print(f"reduced chi-squared = {result['reduced_chi_squared']:.3f}")
        print(f"Starts agreeing in chi-squared: {agreeing}/{len(successful)}")
        for parameter, side in result["bound_hits"].items():
            print(f"Warning: {parameter} is near its {side} fit bound.")
        if result["physical_boundary_hit"]:
            print("Warning: fit is near the zero total-speed boundary; "
                  "symmetric parameter uncertainties are not reported (NaN).")
        if len(successful) < len(trials):
            print(f"Warning: {len(trials) - len(successful)} starting-point trials failed.")
    if plot:
        plot_halo_fit(galaxy_id, result, output_path=output_path, model=model,
                      max_scale_radius_kpc=max_scale_radius_kpc, show=show)
    return result
