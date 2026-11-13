import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from data_preprocess import load_sparc_dataset
from halo_models import burkert_halo_velocity, nfw_halo_velocity

# Baseline upper bound on the fitted halo scale radius, in kpc.
# Override it for a single run with --max-scale-radius-kpc.
DEFAULT_MAX_SCALE_RADIUS_KPC = 100.0

# Each entry contains a velocity function, a plot label, and parameter names.
HALO_MODELS = {
    "burkert": (burkert_halo_velocity, "Burkert", "rho0", "r0"),
    "nfw": (nfw_halo_velocity, "NFW", "rho_s", "r_s"),
}


def fit_halo(
    galaxy_id, df=None, output_path=None, *, model="burkert", plot=True,
    max_scale_radius_kpc=DEFAULT_MAX_SCALE_RADIUS_KPC,
    show=True,
):
    """Fit baryons plus the selected halo to a SPARC rotation curve."""
    if model not in HALO_MODELS:
        raise ValueError(f"Unknown halo model: {model!r}")

    halo_speed, label, density_name, radius_name = HALO_MODELS[model]

    if df is None:
        df = load_sparc_dataset()

    galaxy = df.loc[df["ID"] == galaxy_id].sort_values("R_kpc")
    if galaxy.empty:
        raise ValueError(f"Galaxy {galaxy_id!r} was not found.")

    radius = galaxy["R_kpc"].to_numpy(dtype=float)
    observed = galaxy["Vobs_kms"].to_numpy(dtype=float)
    error = galaxy["eV_kms"].to_numpy(dtype=float)
    baryonic = galaxy["Vbar_kms"].to_numpy(dtype=float)

    valid = (
        np.isfinite(radius)
        & np.isfinite(observed)
        & np.isfinite(error)
        & np.isfinite(baryonic)
        & (radius > 0)
        & (error > 0)
    )
    radius, observed, error, baryonic = (
        radius[valid], observed[valid], error[valid], baryonic[valid]
    )

    if radius.size < 3:
        raise ValueError(f"Galaxy {galaxy_id!r} has too few valid observations.")

    def total_speed(radius_kpc, density_msun_pc3, scale_radius_kpc):
        baryonic_speed = np.interp(radius_kpc, radius, baryonic)
        halo = halo_speed(radius_kpc, density_msun_pc3, scale_radius_kpc)
        return np.hypot(baryonic_speed, halo)

    lower_bounds = (1e-5, 0.01)
    min_scale_radius_kpc = lower_bounds[1]

    if not np.isfinite(max_scale_radius_kpc) or max_scale_radius_kpc <= min_scale_radius_kpc:
        raise ValueError(
            f"Maximum scale radius must be finite and above {min_scale_radius_kpc:g} kpc."
        )

    upper_bounds = (10.0, max_scale_radius_kpc)
    radius_guess = min(5.0, (min_scale_radius_kpc + max_scale_radius_kpc) / 2.0)
    
    parameters, covariance = curve_fit(
        total_speed,
        radius,
        observed,
        sigma=error,
        absolute_sigma=True,
        p0=(0.05, radius_guess),
        bounds=(lower_bounds, upper_bounds),
        maxfev=20_000,
    )
    density, scale_radius = parameters
    density_error, scale_radius_error = np.sqrt(np.diag(covariance))

    halo = halo_speed(radius, *parameters)
    total = total_speed(radius, *parameters)
    residuals = observed - total
    reduced_chi_squared = np.sum((residuals / error) ** 2) / (radius.size - 2)

    print(f"{label} fit for {galaxy_id}")
    print(f"{density_name} = {density:.6g} ± {density_error:.2g} Msun/pc^3")
    print(f"{radius_name} = {scale_radius:.6g} ± {scale_radius_error:.2g} kpc")
    print(f"reduced chi-squared = {reduced_chi_squared:.3f}")
    bound_hits = {}
    parameter_labels = ((density_name, "Msun/pc^3"), (radius_name, "kpc"))

    
    for (name, unit), value, lower, upper in zip(
        parameter_labels, parameters, lower_bounds, upper_bounds
    ):
        if np.isclose(value, lower, rtol=1e-5, atol=0.0):
            side, limit = "lower", lower
        elif np.isclose(value, upper, rtol=1e-5, atol=0.0):
            side, limit = "upper", upper
        else:
            continue

        bound_hits[name] = side
        print(f"Warning: {name} is near the {side} fit bound ({limit:g} {unit}).")
    

    result = {
        f"{density_name}_msun_pc3": density,
        f"{density_name}_error_msun_pc3": density_error,
        f"{radius_name}_kpc": scale_radius,
        f"{radius_name}_error_kpc": scale_radius_error,
        "reduced_chi_squared": reduced_chi_squared,
        "radius_kpc": radius,
        "observed_kms": observed,
        "error_kms": error,
        "bound_hits": bound_hits,
        "model_kms": total,
        "residuals_kms": residuals,
    }
    if not plot:
        return result
    
    fig, ax = plt.subplots()
    ax.errorbar(radius, observed, yerr=error, fmt="o", label="Observed")
    ax.plot(radius, baryonic, label="Baryonic")
    ax.plot(radius, halo, label=f"{label} halo")
    ax.plot(radius, total, label="Total model")
    ax.set_title(f"{galaxy_id} - baryonic + {label} fit")
    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel("Circular velocity (km/s)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if output_path is None:
        suffix = ""
        if max_scale_radius_kpc != DEFAULT_MAX_SCALE_RADIUS_KPC:
            suffix = f"_cap{max_scale_radius_kpc:g}kpc"
        output_path = f"{galaxy_id}_{model}{suffix}_fit.png"

    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    return result


def fit_burkert(galaxy_id, df=None, output_path=None):
    """Fit a Burkert halo for callers such as main.py."""
    return fit_halo(galaxy_id, df=df, output_path=output_path, model="burkert")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a halo to a SPARC galaxy.")
    parser.add_argument("galaxy_id", nargs="?", help="SPARC galaxy ID")
    parser.add_argument("--model", choices=HALO_MODELS, default="burkert")
    parser.add_argument(
        "--max-scale-radius-kpc",
        type=float,
        default=DEFAULT_MAX_SCALE_RADIUS_KPC,
        help="Upper fit limit for the halo scale radius in kpc (default: %(default)g)",
    )
    args = parser.parse_args()

    requested_id = (args.galaxy_id or input("Enter a SPARC galaxy ID: ")).strip()
    df = load_sparc_dataset()
    available_ids = {name.casefold(): name for name in df["ID"].unique()}
    galaxy_id = available_ids.get(requested_id.casefold())

    if galaxy_id is None:
        parser.error(f"Galaxy {requested_id!r} was not found in SPARC data.")

    try:
        fit_halo(
            galaxy_id,
            df=df,
            model=args.model,
            max_scale_radius_kpc=args.max_scale_radius_kpc,
        )
    except ValueError as error:
        parser.error(str(error))
