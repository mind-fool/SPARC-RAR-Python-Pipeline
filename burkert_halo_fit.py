import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.optimize import curve_fit

from data_preprocess import load_sparc_dataset
from halo_models import burkert_halo_velocity


def fit_burkert(galaxy_id, df=None, output_path=None):
    """Fit a baryonic-plus-Burkert model to one SPARC galaxy."""
    if df is None:
        df = load_sparc_dataset()

    galaxy = (
        df[df["ID"] == galaxy_id]
        .sort_values("R_kpc")
        .copy()
    )

    if galaxy.empty:
        raise ValueError(f"Galaxy '{galaxy_id}' was not found.")

    radius = galaxy["R_kpc"].to_numpy(dtype=float)
    observed_velocity = galaxy["Vobs_kms"].to_numpy(dtype=float)
    velocity_error = galaxy["eV_kms"].to_numpy(dtype=float)
    baryonic_velocity = galaxy["Vbar_kms"].to_numpy(dtype=float)

    valid = (
        np.isfinite(radius)
        & np.isfinite(observed_velocity)
        & np.isfinite(velocity_error)
        & np.isfinite(baryonic_velocity)
        & (radius > 0.0)
        & (velocity_error > 0.0)
    )

    radius = radius[valid]
    observed_velocity = observed_velocity[valid]
    velocity_error = velocity_error[valid]
    baryonic_velocity = baryonic_velocity[valid]

    if len(radius) < 3:
        raise ValueError(
            f"Galaxy '{galaxy_id}' has too few valid observations."
        )

    def total_velocity_model(
        radius_kpc,
        rho0_msun_pc3,
        r0_kpc,
    ):
        baryonic = np.interp(
            radius_kpc,
            radius,
            baryonic_velocity,
        )

        halo = burkert_halo_velocity(
            radius_kpc,
            rho0_msun_pc3,
            r0_kpc,
        )

        return np.sqrt(baryonic**2 + halo**2)

    parameters, covariance = curve_fit(
        total_velocity_model,
        radius,
        observed_velocity,
        sigma=velocity_error,
        absolute_sigma=True,
        p0=(0.05, 5.0),
        bounds=(
            (1.0e-5, 0.01),
            (10.0, 100.0),
        ),
        maxfev=20_000,
    )

    rho0, r0 = parameters
    rho0_error, r0_error = np.sqrt(
        np.diag(covariance)
    )

    halo_velocity = burkert_halo_velocity(
        radius,
        rho0,
        r0,
    )

    total_velocity = np.sqrt(
        baryonic_velocity**2 + halo_velocity**2
    )

    degrees_of_freedom = max(
        len(radius) - len(parameters),
        1,
    )

    reduced_chi_squared = np.sum(
        (
            (observed_velocity - total_velocity)
            / velocity_error
        ) ** 2
    ) / degrees_of_freedom

    print(
        f"rho0 = {rho0:.6g} ± "
        f"{rho0_error:.2g} Msun/pc^3"
    )
    print(
        f"r0 = {r0:.6g} ± "
        f"{r0_error:.2g} kpc"
    )
    print(
        f"reduced chi-squared = "
        f"{reduced_chi_squared:.3f}"
    )

    plt.errorbar(
        radius,
        observed_velocity,
        yerr=velocity_error,
        fmt="o",
        label="Observed",
    )
    plt.plot(
        radius,
        baryonic_velocity,
        label="Baryonic",
    )
    plt.plot(
        radius,
        halo_velocity,
        label="Burkert halo",
    )
    plt.plot(
        radius,
        total_velocity,
        label="Total model",
    )

    plt.title(
        f"{galaxy_id} - baryonic + Burkert fit"
    )
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Circular velocity (km/s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if output_path is None:
        output_path = f"{galaxy_id}_burkert_fit.png"

    plt.savefig(output_path, dpi=300)
    plt.show()

    return {
        "rho0_msun_pc3": rho0,
        "rho0_error_msun_pc3": rho0_error,
        "r0_kpc": r0,
        "r0_error_kpc": r0_error,
        "reduced_chi_squared": reduced_chi_squared,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit a Burkert halo to a SPARC galaxy."
    )
    parser.add_argument("galaxy_id", nargs="?", help="SPARC galaxy ID")
    args = parser.parse_args()

    requested_id = args.galaxy_id or input("Enter a SPARC galaxy ID: ")
    requested_id = requested_id.strip()

    df = load_sparc_dataset()
    available_ids = {name.casefold(): name for name in df["ID"].unique()}
    galaxy_id = available_ids.get(requested_id.casefold())

    if galaxy_id is None:
        parser.error(f"Galaxy {requested_id!r} was not found in SPARC data.")

    fit_burkert(galaxy_id, df=df, output_path=f"{galaxy_id}_burkert_fit.png")
