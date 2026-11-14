"""Test one halo-fit bound at a time on the full SPARC sample.

This diagnostic does not replace the paper's NFW 100/300/1000 kpc sweep.
It records how fitted parameters, fit-bound hits, and Burkert--NFW rankings
respond when one numerical fitting bound is changed while the other bounds
remain at their baseline values.
"""

from contextlib import redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy

from data_preprocess import load_sparc_dataset
from halo_fit_selection import fit_best_halo


PROJECT_DIR = Path(__file__).resolve().parent
BASELINE_DENSITY_BOUNDS = (1e-5, 10.0)
BASELINE_MIN_SCALE_RADIUS_KPC = 0.01
BASELINE_MAX_SCALE_RADIUS_KPC = 100.0
RADIUS_CEILINGS_KPC = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
DENSITY_FLOORS_MSUN_PC3 = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
DENSITY_CEILINGS_MSUN_PC3 = (10.0, 30.0, 100.0, 300.0, 1000.0)
RADIUS_FLOORS_KPC = (0.001, 0.01, 0.1)


def boundary_configurations():
    """Return the pre-specified one-bound-at-a-time configurations.

    The baseline configuration is included in the radius-ceiling family.
    Duplicate numerical configurations are removed but retain the first,
    most direct label.
    """
    configs = []
    for ceiling in RADIUS_CEILINGS_KPC:
        configs.append({
            "family": "radius_ceiling",
            "label": f"radius_ceiling_{ceiling:g}_kpc",
            "density_bounds": BASELINE_DENSITY_BOUNDS,
            "min_scale_radius_kpc": BASELINE_MIN_SCALE_RADIUS_KPC,
            "max_scale_radius_kpc": ceiling,
        })
    for floor in DENSITY_FLOORS_MSUN_PC3:
        configs.append({
            "family": "density_floor",
            "label": f"density_floor_{floor:.0e}_msun_pc3",
            "density_bounds": (floor, BASELINE_DENSITY_BOUNDS[1]),
            "min_scale_radius_kpc": BASELINE_MIN_SCALE_RADIUS_KPC,
            "max_scale_radius_kpc": BASELINE_MAX_SCALE_RADIUS_KPC,
        })
    for ceiling in DENSITY_CEILINGS_MSUN_PC3:
        configs.append({
            "family": "density_ceiling",
            "label": f"density_ceiling_{ceiling:g}_msun_pc3",
            "density_bounds": (BASELINE_DENSITY_BOUNDS[0], ceiling),
            "min_scale_radius_kpc": BASELINE_MIN_SCALE_RADIUS_KPC,
            "max_scale_radius_kpc": BASELINE_MAX_SCALE_RADIUS_KPC,
        })
    for floor in RADIUS_FLOORS_KPC:
        configs.append({
            "family": "radius_floor",
            "label": f"radius_floor_{floor:g}_kpc",
            "density_bounds": BASELINE_DENSITY_BOUNDS,
            "min_scale_radius_kpc": floor,
            "max_scale_radius_kpc": BASELINE_MAX_SCALE_RADIUS_KPC,
        })

    unique = []
    seen = set()
    for config in configs:
        key = (
            config["density_bounds"], config["min_scale_radius_kpc"],
            config["max_scale_radius_kpc"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(config)
    return unique


def classify_scores(burkert_score, nfw_score):
    """Classify a matched Burkert--NFW comparison using the paper tolerance."""
    if not np.isfinite(burkert_score) or not np.isfinite(nfw_score):
        return "fit_failed"
    if np.isclose(nfw_score, burkert_score, rtol=1e-5, atol=1e-8):
        return "near_tie"
    return "burkert" if burkert_score < nfw_score else "nfw"


def fit_for_boundary_study(galaxy_id, dataset, model, configuration):
    """Fit one model and return fields suitable for a transparent CSV record."""
    density_name, radius_name = (
        ("rho0", "r0") if model == "burkert" else ("rho_s", "r_s")
    )
    fields = {
        "n_points": np.nan,
        "density_msun_pc3": np.nan,
        "scale_radius_kpc": np.nan,
        "chi_squared": np.nan,
        "reduced_chi_squared": np.nan,
        "density_bound_hit": "",
        "scale_radius_bound_hit": "",
        "fit_parameterization": "",
        "physical_boundary_hit": "",
        "fit_strategy": "",
        "selection_trial_count": np.nan,
        "selection_success_count": np.nan,
        "selection_failed_count": np.nan,
        "selection_selected_trial": np.nan,
        "selection_agreement_count": np.nan,
        "fit_error": "",
    }
    try:
        with redirect_stdout(StringIO()):
            fit = fit_best_halo(
                galaxy_id, df=dataset, model=model, plot=False,
                min_scale_radius_kpc=configuration["min_scale_radius_kpc"],
                max_scale_radius_kpc=configuration["max_scale_radius_kpc"],
                density_bounds=configuration["density_bounds"],
            )
        if not np.isfinite(fit["reduced_chi_squared"]):
            raise ValueError("Reduced chi-squared is not finite.")
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        fields["fit_error"] = str(error) or type(error).__name__
    else:
        fields.update(
            n_points=len(fit["radius_kpc"]),
            density_msun_pc3=fit[f"{density_name}_msun_pc3"],
            scale_radius_kpc=fit[f"{radius_name}_kpc"],
            chi_squared=fit["chi_squared"],
            reduced_chi_squared=fit["reduced_chi_squared"],
            density_bound_hit=fit["bound_hits"].get(density_name, ""),
            scale_radius_bound_hit=fit["bound_hits"].get(radius_name, ""),
            fit_parameterization=fit["fit_parameterization"],
            physical_boundary_hit=fit["physical_boundary_hit"],
            fit_strategy=fit["fit_strategy"],
            selection_trial_count=fit["selection_trial_count"],
            selection_success_count=fit["selection_success_count"],
            selection_failed_count=fit["selection_failed_count"],
            selection_selected_trial=fit["selection_selected_trial"],
            selection_agreement_count=fit["selection_agreement_count"],
        )
    return fields


def comparison_rows(results):
    """Join the two models at each identical boundary configuration."""
    rows = []
    grouping = results.groupby(["configuration", "galaxy_id"], sort=False)
    for (configuration, galaxy_id), group in grouping:
        by_model = group.set_index("model")
        burkert = by_model.loc["burkert"]
        nfw = by_model.loc["nfw"]
        burkert_score = burkert["reduced_chi_squared"]
        nfw_score = nfw["reduced_chi_squared"]
        rows.append({
            "configuration": configuration,
            "family": burkert["family"],
            "galaxy_id": galaxy_id,
            "burkert_reduced_chi_squared": burkert_score,
            "nfw_reduced_chi_squared": nfw_score,
            "delta_reduced_chi_squared_nfw_minus_burkert": nfw_score - burkert_score,
            "lower_chi_squared_model": classify_scores(burkert_score, nfw_score),
        })
    return pd.DataFrame(rows)


def _source_hashes():
    names = (
        "burkert_halo_fit.py", "halo_fit_selection.py", "optimizer_starts.py",
        "halo_models.py", "data_preprocess.py", "run_boundary_sensitivity.py",
    )
    return {
        name: sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
        for name in names if (PROJECT_DIR / name).is_file()
    }


def _git_metadata():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_DIR, text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", None
    return commit, dirty


def run_boundary_sensitivity(output_dir=None, *, data_path=None, configurations=None):
    """Run the pre-specified study and write a unique, self-describing folder."""
    data_path = Path(data_path) if data_path else PROJECT_DIR / "GALAXY_CSVDATA.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing input data: {data_path}")
    configs = boundary_configurations() if configurations is None else list(configurations)
    if not configs:
        raise ValueError("At least one boundary configuration is required.")
    dataset = load_sparc_dataset(data_path)
    galaxy_ids = sorted(dataset["ID"].unique())
    if not galaxy_ids:
        raise ValueError("No galaxies are available after preprocessing.")

    root = Path(output_dir) if output_dir else PROJECT_DIR / "outputs" / "boundary_sensitivity"
    run_dir = root / datetime.now(timezone.utc).strftime("bounds_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    commit, dirty = _git_metadata()
    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "One-bound-at-a-time numerical boundary-sensitivity study.",
        "interpretation": (
            "Best among ten tested starts; stable results do not establish a global "
            "minimum or identify a physically preferred halo model."
        ),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "input_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "galaxy_count": len(galaxy_ids),
        "galaxy_ids": galaxy_ids,
        "configurations": configs,
        "source_sha256": _source_hashes(),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total_fits = len(galaxy_ids) * len(configs) * 2
    print(
        f"Fitting {len(galaxy_ids)} galaxies in {len(configs)} configurations "
        f"({total_fits} model fits; ten starts per fit).",
        flush=True,
    )
    rows = []
    for config_number, configuration in enumerate(configs, 1):
        print(
            f"{config_number}/{len(configs)} {configuration['label']}: "
            f"density={configuration['density_bounds']}, "
            f"radius={configuration['min_scale_radius_kpc']:g}-"
            f"{configuration['max_scale_radius_kpc']:g} kpc",
            flush=True,
        )
        for number, galaxy_id in enumerate(galaxy_ids, 1):
            for model in ("burkert", "nfw"):
                fields = fit_for_boundary_study(galaxy_id, dataset, model, configuration)
                rows.append({
                    "configuration": configuration["label"],
                    "family": configuration["family"],
                    "galaxy_id": galaxy_id,
                    "model": model,
                    "density_lower_bound_msun_pc3": configuration["density_bounds"][0],
                    "density_upper_bound_msun_pc3": configuration["density_bounds"][1],
                    "scale_radius_lower_bound_kpc": configuration["min_scale_radius_kpc"],
                    "scale_radius_upper_bound_kpc": configuration["max_scale_radius_kpc"],
                    **fields,
                })
            if number % 25 == 0 or number == len(galaxy_ids):
                print(f"  {number}/{len(galaxy_ids)} galaxies fit", flush=True)
        pd.DataFrame(rows).to_csv(run_dir / "fit_results.csv", index=False)

    results = pd.DataFrame(rows)
    comparisons = comparison_rows(results)
    comparisons.to_csv(run_dir / "model_comparisons.csv", index=False)
    summary = (
        results.assign(
            density_lower_hit=results["density_bound_hit"].eq("lower"),
            density_upper_hit=results["density_bound_hit"].eq("upper"),
            radius_lower_hit=results["scale_radius_bound_hit"].eq("lower"),
            radius_upper_hit=results["scale_radius_bound_hit"].eq("upper"),
            failed=results["fit_error"].ne(""),
        )
        .groupby(["configuration", "family", "model"], as_index=False)
        .agg(
            galaxies=("galaxy_id", "size"),
            failed_fits=("failed", "sum"),
            density_lower_hits=("density_lower_hit", "sum"),
            density_upper_hits=("density_upper_hit", "sum"),
            radius_lower_hits=("radius_lower_hit", "sum"),
            radius_upper_hits=("radius_upper_hit", "sum"),
            median_reduced_chi_squared=("reduced_chi_squared", "median"),
        )
    )
    summary.to_csv(run_dir / "configuration_summary.csv", index=False)
    comparison_summary = (
        comparisons.groupby("configuration")["lower_chi_squared_model"]
        .value_counts().unstack(fill_value=0).reset_index()
    )
    comparison_summary.to_csv(run_dir / "comparison_summary.csv", index=False)
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved boundary-sensitivity results under {run_dir}")
    return run_dir


if __name__ == "__main__":
    run_boundary_sensitivity()
