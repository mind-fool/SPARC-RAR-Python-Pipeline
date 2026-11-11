import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd

from burkert_halo_fit import fit_halo
from halo_models import burkert_halo_velocity, nfw_halo_velocity


class TestHaloFitter(unittest.TestCase):
    def assert_recovers_parameters(self, model, halo_velocity, known, keys):
        radius_kpc = np.geomspace(0.15, 35.0, 36)
        baryonic_kms = 45.0 * np.sqrt(radius_kpc / (radius_kpc + 3.0))
        halo_kms = halo_velocity(radius_kpc, *known)
        observed_kms = np.hypot(baryonic_kms, halo_kms)

        data = pd.DataFrame({
            "ID": "synthetic",
            "R_kpc": radius_kpc,
            "Vobs_kms": observed_kms,
            "eV_kms": 3.0,
            "Vbar_kms": baryonic_kms,
        })

        with redirect_stdout(StringIO()):
            result = fit_halo("synthetic", df=data, model=model, plot=False)

        recovered = [result[key] for key in keys]
        np.testing.assert_allclose(recovered, known, rtol=1e-6)
        np.testing.assert_allclose(result["residuals_kms"], 0.0, atol=1e-6)

    def test_recovers_burkert_parameters(self):
        self.assert_recovers_parameters(
            "burkert",
            burkert_halo_velocity,
            (0.032, 3.6),
            ("rho0_msun_pc3", "r0_kpc"),
        )

    def test_recovers_nfw_parameters(self):
        self.assert_recovers_parameters(
            "nfw",
            nfw_halo_velocity,
            (0.018, 7.5),
            ("rho_s_msun_pc3", "r_s_kpc"),
        )