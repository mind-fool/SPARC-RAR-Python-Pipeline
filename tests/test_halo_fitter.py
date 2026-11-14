import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
            "Vbar_squared_kms2": baryonic_kms**2,
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

    def signed_data(self, halo_velocity, *, zero_first_speed=False):
        radius = np.geomspace(0.08, 30.0, 45)
        known = (0.4, 2.5)
        halo_squared = halo_velocity(radius, *known)**2
        baryonic_squared = 400.0 * radius / (radius + 3.0)
        baryonic_squared[0] = -halo_squared[0] if zero_first_speed else -33.0
        data = pd.DataFrame({
            "ID": "signed", "R_kpc": radius,
            "Vobs_kms": np.sqrt(baryonic_squared + halo_squared),
            "eV_kms": 3.0, "Vbar_squared_kms2": baryonic_squared,
            # Undefined baryonic-only speed must not remove this observation.
            "Vbar_kms": np.sqrt(np.where(baryonic_squared >= 0, baryonic_squared, np.nan)),
        })
        return data, known

    def assert_signed_recovery(self, model, halo_velocity, keys):
        data, known = self.signed_data(halo_velocity)
        with redirect_stdout(StringIO()):
            result = fit_halo("signed", df=data, model=model, plot=False)
        self.assertEqual(len(result["radius_kpc"]), len(data))
        self.assertEqual(result["fit_parameterization"], "feasible_density_fraction")
        self.assertLess(result["baryonic_squared_kms2"][0], 0)
        np.testing.assert_allclose([result[key] for key in keys], known, rtol=1e-5)
        np.testing.assert_allclose(result["model_kms"], data["Vobs_kms"], atol=1e-5)

    def test_recovers_burkert_with_negative_baryonic_contribution(self):
        # The old initial density of 0.05 gives an invalid total at the first radius.
        self.assert_signed_recovery(
            "burkert", burkert_halo_velocity, ("rho0_msun_pc3", "r0_kpc"),
        )

    def test_recovers_nfw_with_negative_baryonic_contribution(self):
        self.assert_signed_recovery(
            "nfw", nfw_halo_velocity, ("rho_s_msun_pc3", "r_s_kpc"),
        )

    def test_infeasible_signed_data_raise_explicit_error(self):
        data, _ = self.signed_data(burkert_halo_velocity)
        data["Vbar_squared_kms2"] = -1e20
        for model in ("burkert", "nfw"):
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, "No positive total-speed solution"):
                    fit_halo("signed", df=data, model=model, plot=False)

    def test_physical_boundary_flags_unreliable_symmetric_errors(self):
        data, _ = self.signed_data(burkert_halo_velocity, zero_first_speed=True)
        with redirect_stdout(StringIO()):
            result = fit_halo("signed", df=data, model="burkert", plot=False)
        self.assertTrue(result["physical_boundary_hit"])
        self.assertTrue(np.isnan(result["rho0_error_msun_pc3"]))
        self.assertTrue(np.isnan(result["r0_error_kpc"]))
        self.assertEqual(len(result["radius_kpc"]), len(data))

if __name__ == "__main__":
    unittest.main(verbosity=2)