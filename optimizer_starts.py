"""Shared deterministic starting points and numerical agreement tolerances."""

from itertools import product

import numpy as np

# Numerical agreement in total chi-squared, not a significance threshold.
CHI2_ATOL = 1e-6
CHI2_RTOL = 1e-6


def starting_points(
    cap_kpc, *, min_scale_radius_kpc=0.01, density_bounds=(1e-5, 10.0),
):
    """Original start plus a 3-by-3 logarithmic grid inside the fit bounds."""
    try:
        density_min, density_max = (float(value) for value in density_bounds)
    except (TypeError, ValueError) as exc:
        raise ValueError("density_bounds must contain lower and upper limits.") from exc
    if (
        not np.isfinite(density_min)
        or not np.isfinite(density_max)
        or density_min <= 0
        or density_max <= density_min
    ):
        raise ValueError("density_bounds must satisfy 0 < lower < upper.")
    if (
        not np.isfinite(min_scale_radius_kpc)
        or min_scale_radius_kpc <= 0
        or not np.isfinite(cap_kpc)
        or cap_kpc <= min_scale_radius_kpc
    ):
        raise ValueError("Radius limits must be finite and satisfy 0 < lower < upper.")
    default_density = min(max(0.05, density_min), density_max)
    default = (
        default_density,
        min(max(5.0, min_scale_radius_kpc), (min_scale_radius_kpc + cap_kpc) / 2.0),
    )
    # Interior points at 10%, 50%, and 90% of each log-parameter interval.
    fractions = np.array([0.1, 0.5, 0.9])
    densities = np.exp(
        np.log(density_min) + fractions * np.log(density_max / density_min)
    )
    radii = np.exp(
        np.log(min_scale_radius_kpc)
        + fractions * np.log(cap_kpc / min_scale_radius_kpc)
    )
    return [default] + [(float(d), float(r)) for d, r in product(densities, radii)]

