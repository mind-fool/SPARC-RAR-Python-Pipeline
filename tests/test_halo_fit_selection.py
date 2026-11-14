import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from halo_fit_selection import fit_best_halo
from halo_models import nfw_halo_velocity


class TestHaloFitSelection(unittest.TestCase):
    def fit(self, score, marker):
        return {
            "chi_squared": score, "optimizer_status": 1,
            "radius_kpc": np.array([1., 2., 3.]),
            "model_kms": np.array([marker, marker, marker]),
            "rho_s_msun_pc3": marker, "r_s_kpc": 5.,
        }

    def test_selected_parameters_and_curve_come_from_same_best_trial(self):
        candidates = [self.fit(10., 1.), RuntimeError("failed"), self.fit(3., 2.)]
        candidates += [self.fit(8., 3.) for _ in range(7)]
        with patch("halo_fit_selection.fit_halo", side_effect=candidates) as fitter:
            result = fit_best_halo("example", df=pd.DataFrame(), model="nfw")
        self.assertEqual(fitter.call_count, 10)
        self.assertEqual(result["selection_selected_trial"], 2)
        self.assertEqual(result["chi_squared"], 3.)
        self.assertEqual(result["rho_s_msun_pc3"], 2.)
        np.testing.assert_array_equal(result["model_kms"], [2., 2., 2.])
        self.assertEqual(result["selection_failed_count"], 1)
        self.assertEqual(result["selection_trial_results"][1]["fit_error"], "failed")

    def test_all_failed_trials_raise_instead_of_returning_a_fit(self):
        with patch("halo_fit_selection.fit_halo", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "All 10 starting-point trials failed"):
                fit_best_halo("example", df=pd.DataFrame())

    def test_nonfinite_score_is_not_selected(self):
        candidates = [self.fit(np.nan, 1.)] + [self.fit(5., 2.) for _ in range(9)]
        with patch("halo_fit_selection.fit_halo", side_effect=candidates):
            result = fit_best_halo("example", df=pd.DataFrame(), model="nfw")
        self.assertEqual(result["selection_failed_count"], 1)
        self.assertEqual(result["selection_agreement_count"], 9)
        self.assertEqual(result["selection_selected_trial"], 1)

    def test_custom_bounds_are_forwarded_to_starts_and_trials(self):
        candidate = self.fit(5.0, 2.0)
        with patch("halo_fit_selection.starting_points", return_value=[(0.2, 4.0)]) as starts:
            with patch("halo_fit_selection.fit_halo", return_value=candidate) as fitter:
                fit_best_halo(
                    "example", df=pd.DataFrame(), model="nfw",
                    min_scale_radius_kpc=0.1, max_scale_radius_kpc=50.0,
                    density_bounds=(1e-4, 1.0),
                )
        starts.assert_called_once_with(
            50.0, min_scale_radius_kpc=0.1, density_bounds=(1e-4, 1.0),
        )
        self.assertEqual(fitter.call_args.kwargs["min_scale_radius_kpc"], 0.1)
        self.assertEqual(fitter.call_args.kwargs["max_scale_radius_kpc"], 50.0)
        self.assertEqual(fitter.call_args.kwargs["density_bounds"], (1e-4, 1.0))

    def test_real_synthetic_data_recovers_parameters(self):
        radius = np.geomspace(.1, 30., 25)
        baryonic = 300. * radius / (radius + 3.)
        known = (.03, 4.)
        data = pd.DataFrame({
            "ID": "synthetic", "R_kpc": radius, "eV_kms": 3.,
            "Vobs_kms": np.sqrt(baryonic + nfw_halo_velocity(radius, *known)**2),
            "Vbar_squared_kms2": baryonic,
        })
        result = fit_best_halo("synthetic", df=data, model="nfw")
        np.testing.assert_allclose([result["rho_s_msun_pc3"], result["r_s_kpc"]], known, rtol=1e-5)
        self.assertEqual(result["selection_success_count"], 10)
        self.assertEqual(result["fit_strategy"], "best_of_tested_starts")


if __name__ == "__main__":
    unittest.main(verbosity=2)
