import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as pipeline_main
from halo_models import nfw_halo_velocity
from run_optimizer_validation import starting_points, validate_fit, compare_models


class TestOptimizerValidation(unittest.TestCase):
    def data(self):
        radius = np.geomspace(0.1, 30.0, 25)
        baryons = 300.0 * radius / (radius + 3.0)
        observed = np.sqrt(baryons + nfw_halo_velocity(radius, 0.03, 4.0)**2)
        return pd.DataFrame({
            "ID": "synthetic", "R_kpc": radius, "Vobs_kms": observed,
            "eV_kms": 3.0, "Vbar_squared_kms2": baryons,
        })

    def fake_fit(self, score):
        return {
            "optimizer_status": 1, "optimizer_message": "test convergence",
            "chi_squared": score, "reduced_chi_squared": score / 8,
            "function_evaluations": 5, "radius_kpc": np.arange(1, 11),
            "effective_initial_density_msun_pc3": 0.05,
            "effective_initial_radius_kpc": 5.0, "initial_guess_adjusted": False,
            "fit_parameterization": "density", "physical_boundary_hit": False,
            "rho_s_msun_pc3": 0.03, "r_s_kpc": 4.0, "bound_hits": {},
        }

    def test_grid_is_deterministic_inside_bounds_and_contains_default(self):
        for cap in (100.0, 300.0, 1000.0):
            with self.subTest(cap=cap):
                starts = starting_points(cap)
                self.assertEqual(starts, starting_points(cap))
                self.assertEqual(starts[0], (0.05, 5.0))
                self.assertEqual(len(set(starts)), 10)
                for density, radius in starts:
                    self.assertTrue(1e-5 < density < 10.0)
                    self.assertTrue(0.01 < radius < cap)

    def test_real_synthetic_trials_recover_known_solution(self):
        trials, summary = validate_fit("synthetic", self.data(), "nfw", 100.0)
        self.assertEqual(len(trials), 10)
        self.assertEqual(summary["n_failed"], 0)
        self.assertLessEqual(summary["best_chi_squared"], summary["default_chi_squared"])
        np.testing.assert_allclose(
            [summary["best_density_msun_pc3"], summary["best_scale_radius_kpc"]],
            [0.03, 4.0], rtol=1e-5,
        )

    def test_lower_score_is_selected_and_failed_attempt_is_retained(self):
        starts = [(0.05, 5.0), (0.01, 1.0), (1.0, 20.0)]
        with patch("run_optimizer_validation.fit_halo", side_effect=[
            self.fake_fit(10.0), RuntimeError("did not converge"), self.fake_fit(5.0),
        ]):
            trials, summary = validate_fit("synthetic", None, "nfw", 100.0, starts)
        self.assertEqual(summary["best_trial"], 2)
        self.assertEqual(summary["n_failed"], 1)
        self.assertEqual(summary["default_minus_best_chi_squared"], 5.0)
        self.assertTrue(summary["improved_over_default"])
        self.assertEqual(trials[1]["fit_error"], "did not converge")

    def test_all_failures_do_not_create_a_best_fit(self):
        with patch("run_optimizer_validation.fit_halo", side_effect=RuntimeError("failed")):
            trials, summary = validate_fit("synthetic", None, "nfw", 100.0)
        self.assertEqual(summary["n_successful"], 0)
        self.assertEqual(summary["n_failed"], 10)
        self.assertTrue(np.isnan(summary["best_chi_squared"]))
        self.assertEqual(summary["improved_over_default"], "")

    def test_default_failure_is_distinguished_from_improvement(self):
        with patch("run_optimizer_validation.fit_halo", side_effect=[
            RuntimeError("default failed"), self.fake_fit(5.0),
        ]):
            trials, summary = validate_fit(
                "synthetic", None, "nfw", 100.0, [(0.05, 5.0), (0.01, 1.0)],
            )
        self.assertFalse(summary["default_success"])
        self.assertEqual(summary["best_trial"], 1)
        self.assertTrue(np.isnan(summary["default_minus_best_chi_squared"]))
        self.assertEqual(summary["improved_over_default"], "")

    def test_model_reversal_and_missing_comparison_are_distinguished(self):
        rows = []
        for galaxy, default_nfw in (("changed", 12.0), ("failed", np.nan)):
            for model, default, best in (("burkert", 10.0, 10.0), ("nfw", default_nfw, 5.0)):
                rows.append({
                    "galaxy_id": galaxy, "setting": "baseline", "model": model,
                    "cap_kpc": 100.0, "default_chi_squared": default,
                    "best_chi_squared": best, "default_reduced_chi_squared": default / 8,
                    "best_reduced_chi_squared": best / 8,
                })
        comparisons = compare_models(pd.DataFrame(rows)).set_index("galaxy_id")
        self.assertTrue(comparisons.loc["changed", "model_switch"])
        self.assertFalse(comparisons.loc["failed", "comparison_available"])
        self.assertEqual(comparisons.loc["failed", "model_switch"], "")

    def test_main_option_six_routes_selected_galaxies_without_running_sweeps(self):
        with TemporaryDirectory() as directory:
            with patch.object(pipeline_main, "OUTPUT_DIR", Path(directory)), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "run_validation") as validation, \
                 patch.object(pipeline_main, "run_nfw_bound_sweep") as radius_sweep, \
                 patch.object(pipeline_main, "run_mass_to_light_sweep") as stellar_sweep, \
                 patch("builtins.input", side_effect=["6", "NGC0100,UGC01281"]), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            validation.assert_called_once_with(
                output_dir=Path(directory) / "validation",
                galaxy_ids=["NGC0100", "UGC01281"], data_path=pipeline_main.DATA_PATH,
            )
            radius_sweep.assert_not_called()
            stellar_sweep.assert_not_called()

    def test_main_option_seven_runs_only_boundary_sensitivity_study(self):
        with TemporaryDirectory() as directory:
            with patch.object(pipeline_main, "OUTPUT_DIR", Path(directory)), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "run_boundary_sensitivity") as boundary_study, \
                 patch.object(pipeline_main, "run_nfw_bound_sweep") as radius_sweep, \
                 patch.object(pipeline_main, "run_mass_to_light_sweep") as stellar_sweep, \
                 patch.object(pipeline_main, "run_validation") as validation, \
                 patch("builtins.input", return_value="7"), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            boundary_study.assert_called_once_with(
                output_dir=Path(directory) / "boundary_sensitivity",
                data_path=pipeline_main.DATA_PATH,
            )
            radius_sweep.assert_not_called()
            stellar_sweep.assert_not_called()
            validation.assert_not_called()



    def test_main_option_five_includes_full_validation(self):
        with TemporaryDirectory() as directory:
            with patch.object(pipeline_main, "OUTPUT_DIR", Path(directory)), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "load_sparc_dataset", return_value=self.data()), \
                 patch.object(pipeline_main, "run_galaxy_workflow"), \
                 patch.object(pipeline_main, "run_rar_workflow"), \
                 patch.object(pipeline_main, "run_nfw_bound_sweep"), \
                 patch.object(pipeline_main, "run_mass_to_light_sweep"), \
                 patch.object(pipeline_main, "run_validation") as validation, \
                 patch("builtins.input", return_value="5"), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            validation.assert_called_once_with(
                output_dir=Path(directory) / "validation", galaxy_ids=None,
                data_path=pipeline_main.DATA_PATH,
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)
