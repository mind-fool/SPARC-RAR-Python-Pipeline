"""Shared deterministic starting points and numerical agreement tolerances."""

from itertools import product

import numpy as np

# Numerical agreement in total chi-squared, not a significance threshold.
CHI2_ATOL = 1e-6
CHI2_RTOL = 1e-6


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

