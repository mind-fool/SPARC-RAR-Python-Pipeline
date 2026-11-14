"""Check dependence on optimizer starting points without replacing sweep outputs."""

from contextlib import redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import scipy

from burkert_halo_fit import fit_halo
from data_preprocess import load_sparc_dataset
from run_mass_to_light_sweep import STELLAR_SETTINGS
from run_nfw_bound_sweep import BURKERT_CAP_KPC, NFW_CAPS_KPC

# Numerical agreement in total chi-squared, not a significance threshold.
CHI2_ATOL = 1e-6
CHI2_RTOL = 1e-6
PROJECT_DIR = Path(__file__).resolve().parent


def starting_points(cap_kpc):
    """Original start plus a 3-by-3 logarithmic grid inside the fit bounds."""
    if not np.isfinite(cap_kpc) or cap_kpc <= 0.01:
        raise ValueError("Radius ceiling must be finite and greater than 0.01 kpc.")
    default = (0.05, min(5.0, (0.01 + cap_kpc) / 2.0))
    # Interior points at 10%, 50%, and 90% of each log-parameter interval.
    fractions = np.array([0.1, 0.5, 0.9])
    densities = np.exp(np.log(1e-5) + fractions * np.log(10.0 / 1e-5))
    radii = np.exp(np.log(0.01) + fractions * np.log(cap_kpc / 0.01))
    return [default] + [(float(d), float(r)) for d, r in product(densities, radii)]


def validate_fit(galaxy_id, dataset, model, cap_kpc, starts=None):
    """Return every trial and a summary; the default trial is always first."""
    starts = starting_points(cap_kpc) if starts is None else list(starts)
    if not starts or starts[0] != starting_points(cap_kpc)[0]:
        raise ValueError("The first starting point must be the original default.")
    density_name, radius_name = ("rho0", "r0") if model == "burkert" else ("rho_s", "r_s")
    trials = []
    for index, start in enumerate(starts):
        row = {
            "galaxy_id": galaxy_id, "model": model, "cap_kpc": cap_kpc,
            "trial": index, "is_default": index == 0,
            "requested_initial_density_msun_pc3": start[0],
            "requested_initial_radius_kpc": start[1],
            "success": False, "fit_error": "", "warnings": "",
        }
        for field in ("chi_squared", "reduced_chi_squared", "density_msun_pc3",
                      "scale_radius_kpc", "n_points", "function_evaluations",
                      "effective_initial_density_msun_pc3", "effective_initial_radius_kpc",
                      "optimizer_status"):
            row[field] = np.nan
        for field in ("optimizer_message", "fit_parameterization", "initial_guess_adjusted",
                      "density_bound_hit", "scale_radius_bound_hit", "physical_boundary_hit"):
            row[field] = ""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with redirect_stdout(StringIO()):
                    fit = fit_halo(
                        galaxy_id, df=dataset, model=model, plot=False,
                        max_scale_radius_kpc=cap_kpc, initial_guess=start,
                    )
                if fit["optimizer_status"] not in (1, 2, 3, 4) or not np.isfinite(fit["chi_squared"]):
                    raise ValueError("Optimizer did not return a successful finite fit.")
                row.update({key: fit[key] for key in (
                    "chi_squared", "reduced_chi_squared", "function_evaluations",
                    "effective_initial_density_msun_pc3", "effective_initial_radius_kpc",
                    "optimizer_status", "optimizer_message", "fit_parameterization",
                    "initial_guess_adjusted", "physical_boundary_hit",
                )})
                row.update(
                    success=True, n_points=len(fit["radius_kpc"]),
                    density_msun_pc3=fit[f"{density_name}_msun_pc3"],
                    scale_radius_kpc=fit[f"{radius_name}_kpc"],
                    density_bound_hit=fit["bound_hits"].get(density_name, ""),
                    scale_radius_bound_hit=fit["bound_hits"].get(radius_name, ""),
                )
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                row["fit_error"] = str(exc) or type(exc).__name__
        row["warnings"] = " | ".join(str(item.message) for item in caught)
        trials.append(row)
    return trials, summarize_trials(trials)


def summarize_trials(trials):
    """Keep failures visible and compare successful trials with the best found."""
    default = trials[0]
    successful = [row for row in trials if row["success"]]
    summary = {key: default[key] for key in ("galaxy_id", "model", "cap_kpc")}
    summary.update(n_trials=len(trials), n_successful=len(successful),
                   n_failed=len(trials) - len(successful),
                   default_success=default["success"],
                   default_fit_error=default["fit_error"],
                   default_chi_squared=default["chi_squared"],
                   default_reduced_chi_squared=default["reduced_chi_squared"],
                   default_density_msun_pc3=default["density_msun_pc3"],
                   default_scale_radius_kpc=default["scale_radius_kpc"],
                   default_minus_best_chi_squared=np.nan,
                   improved_over_default="", n_agree_with_best=0,
                   n_distinct_effective_starts=0)
    for field in ("trial", "chi_squared", "reduced_chi_squared", "density_msun_pc3",
                  "scale_radius_kpc", "density_bound_hit", "scale_radius_bound_hit",
                  "physical_boundary_hit"):
        summary[f"best_{field}"] = np.nan if field not in (
            "density_bound_hit", "scale_radius_bound_hit", "physical_boundary_hit") else ""
    if not successful:
        return summary
    if len({row["n_points"] for row in successful}) != 1:
        raise ValueError("Starting-point trials used different numbers of observations.")
    best = min(successful, key=lambda row: row["chi_squared"])
    for field in ("trial", "chi_squared", "reduced_chi_squared", "density_msun_pc3",
                  "scale_radius_kpc", "density_bound_hit", "scale_radius_bound_hit",
                  "physical_boundary_hit"):
        summary[f"best_{field}"] = best[field]
    tolerance = CHI2_ATOL + CHI2_RTOL * abs(best["chi_squared"])
    agreeing = [row for row in successful if row["chi_squared"] - best["chi_squared"] <= tolerance]
    summary.update(
        n_points=best["n_points"], n_agree_with_best=len(agreeing),
        n_distinct_effective_starts=len({
            (row["effective_initial_density_msun_pc3"], row["effective_initial_radius_kpc"])
            for row in successful
        }),
        min_radius_agreeing_kpc=min(row["scale_radius_kpc"] for row in agreeing),
        max_radius_agreeing_kpc=max(row["scale_radius_kpc"] for row in agreeing),
        min_density_agreeing_msun_pc3=min(row["density_msun_pc3"] for row in agreeing),
        max_density_agreeing_msun_pc3=max(row["density_msun_pc3"] for row in agreeing),
    )
    if default["success"]:
        change = default["chi_squared"] - best["chi_squared"]
        summary.update(default_minus_best_chi_squared=change,
                       improved_over_default=bool(change > tolerance))
    return summary


def classify_scores(burkert, nfw):
    if not np.isfinite(burkert) or not np.isfinite(nfw):
        return "fit_failed"
    if np.isclose(nfw, burkert, rtol=1e-5, atol=1e-8):
        return "near_tie"
    return "burkert" if burkert < nfw else "nfw"


def compare_models(summaries):
    """Compare the default and best-found rankings for each tested setting."""
    rows = []
    for (setting, galaxy_id), group in summaries.groupby(["setting", "galaxy_id"], sort=False):
        burkert = group.loc[group.model.eq("burkert")].iloc[0]
        for _, nfw in group.loc[group.model.eq("nfw")].iterrows():
            original = classify_scores(burkert.default_reduced_chi_squared, nfw.default_reduced_chi_squared)
            best = classify_scores(burkert.best_reduced_chi_squared, nfw.best_reduced_chi_squared)
            available = original != "fit_failed" and best != "fit_failed"
            rows.append({
                "setting": setting, "galaxy_id": galaxy_id,
                "burkert_cap_kpc": burkert.cap_kpc, "nfw_cap_kpc": nfw.cap_kpc,
                "default_lower_chi_squared_model": original,
                "best_lower_chi_squared_model": best,
                "comparison_available": available,
                "classification_changed": original != best if available else "",
                "model_switch": {original, best} == {"burkert", "nfw"} if available else "",
                "default_delta_chi_squared_nfw_minus_burkert": nfw.default_chi_squared - burkert.default_chi_squared,
                "best_delta_chi_squared_nfw_minus_burkert": nfw.best_chi_squared - burkert.best_chi_squared,
            })
    return pd.DataFrame(rows)


def run_validation(output_dir=None, galaxy_ids=None, *, data_path=None):
    """Validate both paper sweeps; write a separate, uniquely named run folder."""
    data_path = Path(data_path) if data_path else PROJECT_DIR / "GALAXY_CSVDATA.csv"
    datasets = {
        name: load_sparc_dataset(data_path, disk_mass_to_light=disk, bulge_mass_to_light=bulge)
        for name, disk, bulge in STELLAR_SETTINGS
    }
    baseline_keys = pd.MultiIndex.from_frame(datasets["baseline"][["ID", "R_kpc"]])
    for dataset in datasets.values():
        keys = pd.MultiIndex.from_frame(dataset[["ID", "R_kpc"]])
        if not keys.is_unique or not keys.equals(baseline_keys):
            raise ValueError("Validation requires identical usable observations across stellar settings.")
    available = {name.casefold(): name for name in datasets["baseline"]["ID"].unique()}
    if galaxy_ids is None:
        selected = sorted(available.values())
    else:
        selected = []
        for name in galaxy_ids:
            if name.strip().casefold() not in available:
                raise ValueError(f"Galaxy {name!r} was not found.")
            canonical = available[name.strip().casefold()]
            if canonical not in selected:
                selected.append(canonical)
        if not selected:
            raise ValueError("Choose at least one galaxy.")
    configs = []
    for setting, disk, bulge in STELLAR_SETTINGS:
        configs.append((setting, disk, bulge, "burkert", BURKERT_CAP_KPC))
        caps = NFW_CAPS_KPC if setting == "baseline" else (BURKERT_CAP_KPC,)
        configs.extend((setting, disk, bulge, "nfw", cap) for cap in caps)
    root = Path(output_dir) if output_dir else PROJECT_DIR / "outputs" / "validation"
    run_dir = root / datetime.now(timezone.utc).strftime("optimizer_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_DIR, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "working_tree_dirty": dirty,
        "input_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "galaxy_ids": selected, "stellar_settings": STELLAR_SETTINGS,
        "chi_squared_agreement_atol": CHI2_ATOL, "chi_squared_agreement_rtol": CHI2_RTOL,
        "ranking_atol": 1e-8, "ranking_rtol": 1e-5,
        "starts_by_cap_kpc": {str(cap): starting_points(cap) for cap in sorted({c[4] for c in configs})},
        "source_sha256": {
            name: sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
            for name in ("burkert_halo_fit.py", "halo_models.py", "data_preprocess.py",
                         "run_optimizer_validation.py", "run_mass_to_light_sweep.py",
                         "run_nfw_bound_sweep.py", "main.py") if (PROJECT_DIR / name).is_file()
        },
        "interpretation": "Best among the tested starts; agreement does not prove a global minimum.",
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    total_trials = len(selected) * sum(len(starting_points(c[4])) for c in configs)
    print(f"Checking {len(selected)} galaxies with {total_trials:,} trial fits. This may take several minutes.", flush=True)
    all_trials, all_summaries = [], []
    for setting, disk, bulge, model, cap in configs:
        print(f"{setting}: {model}, {cap:g} kpc", flush=True)
        for number, galaxy_id in enumerate(selected, 1):
            trials, summary = validate_fit(galaxy_id, datasets[setting], model, cap)
            metadata = {"setting": setting, "disk_mass_to_light": disk, "bulge_mass_to_light": bulge}
            all_trials.extend(dict(row, **metadata) for row in trials)
            all_summaries.append(dict(summary, **metadata))
            if number % 25 == 0 or number == len(selected):
                print(f"  {number}/{len(selected)} galaxies checked", flush=True)
        # Save completed configurations so an interrupted run remains inspectable.
        pd.DataFrame(all_trials).to_csv(run_dir / "optimizer_trials.csv", index=False)
        pd.DataFrame(all_summaries).to_csv(run_dir / "optimizer_summary.csv", index=False)
    summaries = pd.DataFrame(all_summaries)
    comparisons = compare_models(summaries)
    comparisons.to_csv(run_dir / "optimizer_comparisons.csv", index=False)
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Fits improved beyond numerical tolerance: {summaries.improved_over_default.eq(True).sum()}/{len(summaries)}")
    print(f"Failed trial attempts: {int(summaries.n_failed.sum())}/{len(all_trials)}")
    print(f"Fit cases with no successful trial: {int(summaries.n_successful.eq(0).sum())}")
    print(f"Burkert/NFW ranking reversals: {comparisons.model_switch.eq(True).sum()}")
    print("Agreement across these starts does not prove that the global minimum was found.")
    print(f"Validation results saved under {run_dir}")
    return run_dir


if __name__ == "__main__":
    requested = input("Galaxy IDs separated by commas [all galaxies]: ").strip()
    run_validation(galaxy_ids=requested.split(",") if requested else None)
