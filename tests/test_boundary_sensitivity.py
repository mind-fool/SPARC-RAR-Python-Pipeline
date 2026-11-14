import unittest

import numpy as np
import pandas as pd

from run_boundary_sensitivity import (
    BASELINE_DENSITY_BOUNDS,
    BASELINE_MAX_SCALE_RADIUS_KPC,
    BASELINE_MIN_SCALE_RADIUS_KPC,
    boundary_configurations,
    classify_scores,
    comparison_rows,
)


class TestBoundarySensitivity(unittest.TestCase):
    def test_configurations_change_only_one_bound_family_at_a_time(self):
        configurations = boundary_configurations()
        self.assertEqual(len(configurations), 17)
        labels = [config["label"] for config in configurations]
        self.assertEqual(len(labels), len(set(labels)))
        for config in configurations:
            with self.subTest(config=config["label"]):
                if config["family"] == "radius_ceiling":
                    self.assertEqual(config["density_bounds"], BASELINE_DENSITY_BOUNDS)
                    self.assertEqual(
                        config["min_scale_radius_kpc"], BASELINE_MIN_SCALE_RADIUS_KPC,
                    )
                elif config["family"] == "density_floor":
                    self.assertEqual(
                        config["max_scale_radius_kpc"], BASELINE_MAX_SCALE_RADIUS_KPC,
                    )
                    self.assertEqual(
                        config["min_scale_radius_kpc"], BASELINE_MIN_SCALE_RADIUS_KPC,
                    )
                elif config["family"] == "radius_floor":
                    self.assertEqual(config["density_bounds"], BASELINE_DENSITY_BOUNDS)
                    self.assertEqual(
                        config["max_scale_radius_kpc"], BASELINE_MAX_SCALE_RADIUS_KPC,
                    )
                elif config["family"] == "density_ceiling":
                    self.assertEqual(
                        config["min_scale_radius_kpc"], BASELINE_MIN_SCALE_RADIUS_KPC,
                    )
                    self.assertEqual(
                        config["max_scale_radius_kpc"], BASELINE_MAX_SCALE_RADIUS_KPC,
                    )
                else:
                    self.fail(f"Unexpected family: {config['family']}")

    def test_score_classification_uses_expected_direction_and_tolerance(self):
        self.assertEqual(classify_scores(1.0, 2.0), "burkert")
        self.assertEqual(classify_scores(2.0, 1.0), "nfw")
        self.assertEqual(classify_scores(1.0, 1.0 + 1e-9), "near_tie")
        self.assertEqual(classify_scores(np.nan, 1.0), "fit_failed")

    def test_comparison_rows_match_models_within_each_configuration(self):
        results = pd.DataFrame({
            "configuration": ["a", "a", "b", "b"],
            "family": ["radius_ceiling"] * 4,
            "galaxy_id": ["G1", "G1", "G1", "G1"],
            "model": ["burkert", "nfw", "burkert", "nfw"],
            "reduced_chi_squared": [1.0, 2.0, 3.0, 2.0],
        })
        comparisons = comparison_rows(results)
        self.assertEqual(len(comparisons), 2)
        self.assertEqual(
            comparisons.set_index("configuration").loc["a", "lower_chi_squared_model"],
            "burkert",
        )
        self.assertEqual(
            comparisons.set_index("configuration").loc["b", "lower_chi_squared_model"],
            "nfw",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
