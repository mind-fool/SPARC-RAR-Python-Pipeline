"""Measure how the NFW scale-radius cap affects SPARC halo fits."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from burkert_halo_fit import fit_halo
from data_preprocess import (
    BULGE_MASS_TO_LIGHT,
    DISK_MASS_TO_LIGHT,
    load_sparc_dataset,
)
from sweep_plots import plot_nfw_bound_sweep


BURKERT_CAP_KPC = 100.0
NFW_CAPS_KPC = (100.0, 300.0, 1000.0)
EXAMPLE_GALAXY = "NGC0100"


def fit_for_sweep(galaxy_id, dataset, model, cap_kpc):
    """Return one fit's results, including any optimizer failure."""
    fields, _ = _fit_with_curve(galaxy_id, dataset, model, cap_kpc)
    return fields


def _fit_with_curve(galaxy_id, dataset, model, cap_kpc):
    density_name, radius_name = ("rho0", "r0") if model == "burkert" else ("rho_s", "r_s")
    fields = dict.fromkeys(
        ("n_points", "density_msun_pc3", "scale_radius_kpc", "reduced_chi_squared",
         "density_bound_hit", "scale_radius_bound_hit",
         "fit_parameterization", "physical_boundary_hit", "fit_error"),
        "",
    )

    try:
        with redirect_stdout(StringIO()):
            fit = fit_halo(
                galaxy_id, df=dataset, model=model, plot=False,
                max_scale_radius_kpc=cap_kpc,
            )
        if not np.isfinite(fit["reduced_chi_squared"]):
            raise ValueError("Reduced chi-squared is not finite.")
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        fields["fit_error"] = str(error) or type(error).__name__
        fit = None
    else:
        fields.update(
            n_points=len(fit["radius_kpc"]),
            density_msun_pc3=fit[f"{density_name}_msun_pc3"],
            scale_radius_kpc=fit[f"{radius_name}_kpc"],
            reduced_chi_squared=fit["reduced_chi_squared"],
            density_bound_hit=fit["bound_hits"].get(density_name, ""),
            scale_radius_bound_hit=fit["bound_hits"].get(radius_name, ""),
            fit_parameterization=fit["fit_parameterization"],
            physical_boundary_hit=fit["physical_boundary_hit"],
        )

    return {f"{model}_{name}": value for name, value in fields.items()}, fit


def main(output_path=None):
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "GALAXY_CSVDATA.csv"
    if not data_path.is_file():
        raise SystemExit(f"Missing {data_path.name}. Run report_converter.py first.")

    dataset = load_sparc_dataset(data_path)
    galaxy_ids = sorted(dataset["ID"].unique())
    if not galaxy_ids:
        raise SystemExit("No galaxies are available after SPARC preprocessing.")

    print(f"Fitting {len(galaxy_ids)} SPARC galaxies. This may take a moment.", flush=True)
    rows = []
    example_fits = {}
    for galaxy_id in galaxy_ids:
        burkert = fit_for_sweep(galaxy_id, dataset, "burkert", BURKERT_CAP_KPC)
        for cap_kpc in NFW_CAPS_KPC:
            nfw, curve = _fit_with_curve(galaxy_id, dataset, "nfw", cap_kpc)
            if galaxy_id == EXAMPLE_GALAXY and curve is not None:
                example_fits[cap_kpc] = curve
            delta = ""
            if burkert["burkert_fit_error"] or nfw["nfw_fit_error"]:
                lower_chi_squared_model = "fit_failed"
            else:
                burkert_score = burkert["burkert_reduced_chi_squared"]
                nfw_score = nfw["nfw_reduced_chi_squared"]
                delta = nfw_score - burkert_score
                # Use a numerical tolerance for near ties in the summary counts.
                if np.isclose(nfw_score, burkert_score, rtol=1e-5, atol=1e-8):
                    lower_chi_squared_model = "near_tie"
                else:
                    lower_chi_squared_model = "burkert" if delta > 0 else "nfw"

            rows.append({
                "galaxy_id": galaxy_id,
                "burkert_cap_kpc": BURKERT_CAP_KPC,
                "nfw_cap_kpc": cap_kpc,
                "disk_mass_to_light": DISK_MASS_TO_LIGHT,
                "bulge_mass_to_light": BULGE_MASS_TO_LIGHT,
                **burkert,
                **nfw,
                "delta_reduced_chi_squared_nfw_minus_burkert": delta,
                "lower_chi_squared_model": lower_chi_squared_model,
            })

    results = pd.DataFrame(rows)
    if output_path is None:
        output_path = project_dir / "nfw_bound_sweep.csv"
    output_path = Path(output_path)
    results.to_csv(output_path, index=False)

    table_rows = []
    for cap_kpc, group in results.groupby("nfw_cap_kpc", sort=True):
        counts = group["lower_chi_squared_model"].value_counts()
        upper_hits = group["nfw_scale_radius_bound_hit"].eq("upper").sum()
        table_rows.append(
            f"{cap_kpc:,.0f} & {upper_hits} & {counts.get('burkert', 0)} & "
            f"{counts.get('nfw', 0)} & {counts.get('near_tie', 0)} " + r"\\"
        )
        print(
            f"{cap_kpc:g} kpc | Burkert lower: {counts.get('burkert', 0)}"
            f" | NFW lower: {counts.get('nfw', 0)} | Near ties: {counts.get('near_tie', 0)}"
            f" | NFW upper-bound hits: {upper_hits} | Failed: {counts.get('fit_failed', 0)}"
        )

    rankings = results.pivot(
        index="galaxy_id", columns="nfw_cap_kpc", values="lower_chi_squared_model"
    )
    model_swaps = rankings.index[
        rankings.eq("burkert").any(axis=1) & rankings.eq("nfw").any(axis=1)
    ]
    print(f"Galaxies where the lower-scoring model switches: {', '.join(model_swaps) or 'none'}")
    print(f"Saved {len(results)} comparisons to {output_path}")
    table_path = output_path.with_name(f"{output_path.stem}_table_rows.tex")
    table_path.write_text("\n".join([*table_rows, r"\hline"]) + "\n", encoding="utf-8")
    print(f"Saved {table_path}")

    figure_path = output_path.with_name(f"{EXAMPLE_GALAXY.lower()}_{output_path.stem}.png")
    if len(example_fits) == len(NFW_CAPS_KPC):
        plot_nfw_bound_sweep(EXAMPLE_GALAXY, example_fits, figure_path)
        print(f"Saved {figure_path}")
    else:
        figure_path.unlink(missing_ok=True)
        print(f"Could not plot {EXAMPLE_GALAXY} because the sweep did not produce all its fits.")


if __name__ == "__main__":
    main()
