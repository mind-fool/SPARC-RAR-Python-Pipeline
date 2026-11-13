"""Compare SPARC halo fits under three stellar mass-to-light assumptions."""

from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocess import (
    BULGE_MASS_TO_LIGHT,
    DISK_MASS_TO_LIGHT,
    load_sparc_dataset,
)
from run_nfw_bound_sweep import fit_for_sweep


HALO_CAP_KPC = 100.0
STELLAR_SETTINGS = (
    ("baseline", DISK_MASS_TO_LIGHT, BULGE_MASS_TO_LIGHT),
    ("higher_disk", 0.6, BULGE_MASS_TO_LIGHT),
    ("lower_bulge", DISK_MASS_TO_LIGHT, 0.5),
)


def main(output_path=None):
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "GALAXY_CSVDATA.csv"
    if not data_path.is_file():
        raise SystemExit(f"Missing {data_path.name}. Run report_converter.py first.")

    datasets = {}
    for setting, disk_ratio, bulge_ratio in STELLAR_SETTINGS:
        dataset = load_sparc_dataset(
            data_path, disk_mass_to_light=disk_ratio, bulge_mass_to_light=bulge_ratio
        )
        datasets[setting] = dataset.set_index(["ID", "R_kpc"], drop=False, verify_integrity=True)

    # Baryonic preprocessing can exclude different radii at different ratios.
    shared_points = datasets["baseline"].index
    for dataset in datasets.values():
        shared_points = shared_points.intersection(dataset.index, sort=False)
    if shared_points.empty:
        raise SystemExit("No SPARC observations are shared across the stellar settings.")

    galaxy_ids = sorted(shared_points.get_level_values("ID").unique())
    print(
        f"Fitting {len(galaxy_ids)} galaxies using {len(shared_points)} shared observations.",
        flush=True,
    )
    rows = []
    for setting, disk_ratio, bulge_ratio in STELLAR_SETTINGS:
        dataset = datasets[setting].loc[shared_points].reset_index(drop=True)
        print(f"Running {setting} (disk={disk_ratio:g}, bulge={bulge_ratio:g})...", flush=True)
        for galaxy_id in galaxy_ids:
            burkert = fit_for_sweep(galaxy_id, dataset, "burkert", HALO_CAP_KPC)
            nfw = fit_for_sweep(galaxy_id, dataset, "nfw", HALO_CAP_KPC)
            rows.append({
                "galaxy_id": galaxy_id,
                "setting": setting,
                "disk_mass_to_light": disk_ratio,
                "bulge_mass_to_light": bulge_ratio,
                "halo_cap_kpc": HALO_CAP_KPC,
                **burkert,
                **nfw,
            })

    results = pd.DataFrame(rows)
    burkert_score = pd.to_numeric(results["burkert_reduced_chi_squared"], errors="coerce")
    nfw_score = pd.to_numeric(results["nfw_reduced_chi_squared"], errors="coerce")
    delta = nfw_score - burkert_score
    results["delta_reduced_chi_squared_nfw_minus_burkert"] = delta
    results["lower_chi_squared_model"] = np.select(
        [delta.isna(), np.isclose(nfw_score, burkert_score, rtol=1e-5, atol=1e-8), delta > 0],
        ["fit_failed", "near_tie", "burkert"],
        default="nfw",
    )

    if output_path is None:
        output_path = project_dir / "mass_to_light_sweep.csv"
    results.to_csv(output_path, index=False)

    rankings = results.pivot(
        index="galaxy_id", columns="setting", values="lower_chi_squared_model"
    )
    baseline_rank = rankings["baseline"]
    for setting, group in results.groupby("setting", sort=False):
        counts = group["lower_chi_squared_model"].value_counts()
        swaps = (
            (baseline_rank.eq("burkert") & rankings[setting].eq("nfw"))
            | (baseline_rank.eq("nfw") & rankings[setting].eq("burkert"))
        )
        print(
            f"{setting} | Burkert lower: {counts.get('burkert', 0)}"
            f" | NFW lower: {counts.get('nfw', 0)} | Near ties: {counts.get('near_tie', 0)}"
            f" | Failed: {counts.get('fit_failed', 0)} | Model switches vs baseline: {swaps.sum()}"
        )

    print(f"Saved {len(results)} comparisons to {output_path}")


if __name__ == "__main__":
    main()
