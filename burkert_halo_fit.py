import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, curve_fit

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


def total_velocity_from_squared(baryonic_squared, halo_squared):
    """Combine signed contributions in (km/s)^2 and return speed in km/s."""
    total_squared = (
        np.asarray(baryonic_squared, dtype=float)
        + np.asarray(halo_squared, dtype=float)
    )
    if not np.all(np.isfinite(total_squared)):
        raise ValueError("Total squared circular speed must be finite.")
    if np.any(total_squared < 0):
        raise ValueError("Total squared circular speed cannot be negative.")
    return np.sqrt(total_squared)


def _fit_signed_baryonic_curve(
    total_speed, halo_speed, radius, observed, error, baryonic_squared,
    lower_bounds, upper_bounds, density_guess, radius_guess,
):
    """Use bounded curve_fit while keeping total squared speeds positive.

    At fixed scale radius, halo squared speed is linear in density. The
    negative baryonic terms therefore set a minimum feasible density.
    Fit a fraction of the remaining density interval and the scale radius.
    This preserves the velocity-residual objective and original physical
    parameter bounds, apart from a small numerical margin at zero speed.
    """
    density_min, radius_min = lower_bounds
    density_max, radius_max = upper_bounds
    negative = baryonic_squared < 0

    def density_floor(scale_radius):
        unit_halo_squared = halo_speed(radius[negative], 1.0, scale_radius)**2
        if np.any(unit_halo_squared <= 0) or not np.all(np.isfinite(unit_halo_squared)):
            raise ValueError("Cannot determine a feasible halo density.")
        required = np.max(-baryonic_squared[negative] / unit_halo_squared)
        # Avoid square-root derivatives at exactly zero total speed.
        return max(density_min, required * (1.0 + 1e-12))

    if density_floor(radius_max) >= density_max:
        raise ValueError("No positive total-speed solution within the halo bounds.")

    if density_floor(radius_min) >= density_max:
        # For Burkert and NFW, halo squared speed increases with scale radius
        # at fixed density. Radii below this root cannot offset the baryons.
        edge = brentq(
            lambda scale: density_floor(scale) - density_max,
            radius_min, radius_max,
        )
        radius_min = edge + 1e-8 * max(1.0, abs(edge))
        if radius_min >= radius_max:
            raise ValueError("Feasible radius interval is too narrow for fitting.")

    radius_guess = min(max(radius_guess, radius_min), radius_max)
    floor_at_start = density_floor(radius_guess)
    density_guess = max(density_guess, floor_at_start + 0.01 * (density_max - floor_at_start))
    fraction_guess = (density_guess - floor_at_start) / (density_max - floor_at_start)

    def feasible_speed(radius_kpc, density_fraction, scale_radius):
        floor = density_floor(scale_radius)
        density = floor + density_fraction * (density_max - floor)
        return total_speed(radius_kpc, density, scale_radius)

    transformed, transformed_covariance, info, message, status = curve_fit(
        feasible_speed, radius, observed, sigma=error, absolute_sigma=True,
        p0=(fraction_guess, radius_guess),
        bounds=((0.0, radius_min), (1.0, radius_max)), maxfev=20_000,
        full_output=True,
    )
    fraction, scale_radius = transformed
    floor = density_floor(scale_radius)
    density = floor + fraction * (density_max - floor)

    # Convert the local covariance back to physical density and radius.
    # These remain local approximations, especially near a boundary.
    step = max(1e-8, 1e-6 * scale_radius)
    left = max(radius_min, scale_radius - step)
    right = min(radius_max, scale_radius + step)
    floor_derivative = (density_floor(right) - density_floor(left)) / (right - left)
    jacobian = np.array([
        [density_max - floor, (1.0 - fraction) * floor_derivative],
        [0.0, 1.0],
    ])
    covariance = jacobian @ transformed_covariance @ jacobian.T
    return (
        np.array([density, scale_radius]), covariance,
        np.array([density_guess, radius_guess]), info, message, status,
    )


def fit_halo(
    galaxy_id, df=None, output_path=None, *, model="burkert", plot=True,
    max_scale_radius_kpc=DEFAULT_MAX_SCALE_RADIUS_KPC,
    show=True, initial_guess=None,
):
    """Fit baryons plus the selected halo to a SPARC rotation curve.

    initial_guess is (density in Msun/pc^3, scale radius in kpc).
    None retains the original default. Signed baryonic data may require
    moving the requested start into the positive-total-speed region;
    requested and effective starting points are recorded in the result.
    """
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
    baryonic_squared = galaxy["Vbar_squared_kms2"].to_numpy(dtype=float)

    valid = (
        np.isfinite(radius)
        & np.isfinite(observed)
        & np.isfinite(error)
        & np.isfinite(baryonic_squared)
        & (radius > 0)
        & (error > 0)
    )
    radius, observed, error, baryonic_squared = (
        radius[valid], observed[valid], error[valid], baryonic_squared[valid]
    )

    if radius.size < 3:
        raise ValueError(f"Galaxy {galaxy_id!r} has too few valid observations.")

    def total_speed(radius_kpc, density_msun_pc3, scale_radius_kpc):
        baryonic_term = np.interp(radius_kpc, radius, baryonic_squared)
        halo = halo_speed(radius_kpc, density_msun_pc3, scale_radius_kpc)
        return total_velocity_from_squared(baryonic_term, halo**2)

    lower_bounds = (1e-5, 0.01)
    min_scale_radius_kpc = lower_bounds[1]

    if not np.isfinite(max_scale_radius_kpc) or max_scale_radius_kpc <= min_scale_radius_kpc:
        raise ValueError(
            f"Maximum scale radius must be finite and above {min_scale_radius_kpc:g} kpc."
        )

    upper_bounds = (10.0, max_scale_radius_kpc)
    if initial_guess is None:
        radius_guess = min(5.0, (min_scale_radius_kpc + max_scale_radius_kpc) / 2.0)
        requested_initial = np.array([0.05, radius_guess])
    else:
        try:
            requested_initial = np.asarray(initial_guess, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("initial_guess must contain a finite density and scale radius.") from exc
        if requested_initial.shape != (2,) or not np.all(np.isfinite(requested_initial)):
            raise ValueError("initial_guess must contain a finite density and scale radius.")
        if np.any(requested_initial < lower_bounds) or np.any(requested_initial > upper_bounds):
            raise ValueError("initial_guess must lie within the halo parameter bounds.")
    density_guess, radius_guess = requested_initial

    if np.any(baryonic_squared < 0):
        parameters, covariance, effective_initial, info, message, status = _fit_signed_baryonic_curve(
            total_speed, halo_speed, radius, observed, error,
            baryonic_squared, lower_bounds, upper_bounds, density_guess, radius_guess,
        )
        fit_parameterization = "feasible_density_fraction"
    else:
        parameters, covariance, info, message, status = curve_fit(
            total_speed,
            radius,
            observed,
            sigma=error,
            absolute_sigma=True,
            p0=requested_initial,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20_000,
            full_output=True,
        )
        effective_initial = requested_initial.copy()
        fit_parameterization = "density"
    density, scale_radius = parameters

    halo = halo_speed(radius, *parameters)
    total = total_speed(radius, *parameters)
    # Flag near cancellation of a negative baryonic term by the halo.
    # This diagnostic uses a relative tolerance of 1e-4 in squared speed.
    physical_boundary_hit = bool(np.any(
        (baryonic_squared < 0)
        & np.isclose(halo**2, -baryonic_squared, rtol=1e-4, atol=0.0)
    ))
    if physical_boundary_hit:
        density_error = scale_radius_error = np.nan
    else:
        density_error, scale_radius_error = np.sqrt(np.diag(covariance))
    residuals = observed - total
    chi_squared = float(np.sum((residuals / error) ** 2))
    reduced_chi_squared = chi_squared / (radius.size - 2)

    print(f"{label} fit for {galaxy_id}")
    print(f"{density_name} = {density:.6g} ± {density_error:.2g} Msun/pc^3")
    print(f"{radius_name} = {scale_radius:.6g} ± {scale_radius_error:.2g} kpc")
    print(f"reduced chi-squared = {reduced_chi_squared:.3f}")
    if physical_boundary_hit:
        print("Warning: fit is near the zero total-speed boundary; "
              "symmetric parameter uncertainties are not reported (NaN).")
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
        "chi_squared": chi_squared,
        "reduced_chi_squared": reduced_chi_squared,
        "requested_initial_density_msun_pc3": float(requested_initial[0]),
        "requested_initial_radius_kpc": float(requested_initial[1]),
        "effective_initial_density_msun_pc3": float(effective_initial[0]),
        "effective_initial_radius_kpc": float(effective_initial[1]),
        "initial_guess_adjusted": bool(np.any(effective_initial != requested_initial)),
        "optimizer_status": int(status),
        "optimizer_message": message,
        "function_evaluations": int(info["nfev"]),
        "fit_parameterization": fit_parameterization,
        "physical_boundary_hit": physical_boundary_hit,
        "baryonic_squared_kms2": baryonic_squared,
        "radius_kpc": radius,
        "observed_kms": observed,
        "error_kms": error,
        "bound_hits": bound_hits,
        "model_kms": total,
        "residuals_kms": residuals,
    }
    if plot:
        plot_halo_fit(galaxy_id, result, output_path=output_path, model=model,
                      max_scale_radius_kpc=max_scale_radius_kpc, show=show)
    return result


def plot_halo_fit(
    galaxy_id, result, output_path=None, *, model="burkert",
    max_scale_radius_kpc=DEFAULT_MAX_SCALE_RADIUS_KPC, show=True,
):
    """Plot an existing fit without running the optimizer again."""
    halo_speed, label, density_name, radius_name = HALO_MODELS[model]
    radius = result["radius_kpc"]
    observed = result["observed_kms"]
    error = result["error_kms"]
    baryonic_squared = result["baryonic_squared_kms2"]
    total = result["model_kms"]
    halo = halo_speed(radius, result[f"{density_name}_msun_pc3"], result[f"{radius_name}_kpc"])
    fig, ax = plt.subplots()
    ax.errorbar(radius, observed, yerr=error, fmt="o", label="Observed")
    baryonic = np.sqrt(np.where(baryonic_squared >= 0, baryonic_squared, np.nan))
    ax.plot(radius, baryonic, label="Baryonic (where defined)")
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



def fit_burkert(galaxy_id, df=None, output_path=None):
    """Fit a Burkert halo for callers such as main.py."""
    from halo_fit_selection import fit_best_halo
    return fit_best_halo(galaxy_id, df=df, output_path=output_path,
                         model="burkert", plot=True, report=True)


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

    from halo_fit_selection import fit_best_halo

    try:
        fit_best_halo(
            galaxy_id,
            plot=True, report=True,
            df=df,
            model=args.model,
            max_scale_radius_kpc=args.max_scale_radius_kpc,
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
