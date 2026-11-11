import unittest

import numpy as np
from scipy.integrate import quad

from halo_models import burkert_halo_velocity, nfw_halo_velocity


METERS_PER_PARSEC = 3.08567758149e16
SOLAR_GM_SI = 1.3271244e20  # G times one solar mass, in m^3/s^2


class TestHaloVelocities(unittest.TestCase):
    def assert_matches_integrated_density(self, halo_velocity, integrand):
        radii_kpc = np.array([0.5, 5.0, 20.0])
        scale_kpc = 5.0
        scale_pc = 1000.0 * scale_kpc
        density_msun_pc3 = 0.05
        expected_kms = []

        for radius_kpc in radii_kpc:
            # Integrate x^2 times the normalized density, where x = r / scale.
            integral, _ = quad(integrand, 0.0, radius_kpc / scale_kpc)
            mass_msun = 4.0 * np.pi * density_msun_pc3 * scale_pc**3 * integral
            radius_m = radius_kpc * 1000.0 * METERS_PER_PARSEC
            expected_kms.append(np.sqrt(SOLAR_GM_SI * mass_msun / radius_m) / 1000.0)

        actual_kms = halo_velocity(radii_kpc, density_msun_pc3, scale_kpc)
        np.testing.assert_allclose(actual_kms, expected_kms, rtol=1e-5)

    def test_burkert_matches_integrated_density(self):
        self.assert_matches_integrated_density(
            burkert_halo_velocity,
            lambda x: x**2 / ((1.0 + x) * (1.0 + x**2)),
        )

    def test_nfw_matches_integrated_density(self):
        self.assert_matches_integrated_density(
            nfw_halo_velocity,
            lambda x: x / (1.0 + x)**2,
        )