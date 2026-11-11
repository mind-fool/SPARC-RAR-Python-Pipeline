"""Circular-velocity profiles for spherical dark-matter halos."""

import numpy as np


G_KPC = 4.30091e-6  # kpc (km/s)^2 / solar mass
PC3_TO_KPC3 = 1.0e9


def burkert_halo_velocity(radius_kpc, rho0_msun_pc3, r0_kpc):
    """Return the Burkert halo's circular velocity in km/s at positive radii."""
    radius_kpc = np.asarray(radius_kpc, dtype=float)
    x = radius_kpc / r0_kpc
    shape = np.log1p(x) + 0.5 * np.log1p(x**2) - np.arctan(x)

    rho0_msun_kpc3 = rho0_msun_pc3 * PC3_TO_KPC3
    enclosed_mass_msun = 2.0 * np.pi * rho0_msun_kpc3 * r0_kpc**3 * shape
    velocity_squared = G_KPC * enclosed_mass_msun / radius_kpc
    return np.sqrt(np.maximum(velocity_squared, 0.0))


def nfw_halo_velocity(radius_kpc, rho_s_msun_pc3, r_s_kpc):
    """Return the NFW halo's circular velocity in km/s at positive radii."""
    radius_kpc = np.asarray(radius_kpc, dtype=float)
    x = radius_kpc / r_s_kpc
    shape = np.log1p(x) - x / (1.0 + x)

    rho_s_msun_kpc3 = rho_s_msun_pc3 * PC3_TO_KPC3
    enclosed_mass_msun = 4.0 * np.pi * rho_s_msun_kpc3 * r_s_kpc**3 * shape
    return np.sqrt(G_KPC * enclosed_mass_msun / radius_kpc)
