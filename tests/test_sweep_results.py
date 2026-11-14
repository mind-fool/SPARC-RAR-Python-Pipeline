import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_nfw_bound_sweep import fit_for_sweep


class TestSweepResults(unittest.TestCase):
    def test_csv_preserves_fit_parameterization_and_physical_boundary(self):
        for model, density, radius in (
            ("burkert", "rho0", "r0"), ("nfw", "rho_s", "r_s"),
        ):
            for flagged in (False, True):
                with self.subTest(model=model, flagged=flagged):
                    fit = {
                        "radius_kpc": [1.0, 2.0, 3.0],
                        f"{density}_msun_pc3": 0.03,
                        f"{radius}_kpc": 5.0,
                        "reduced_chi_squared": 1.0,
                        "bound_hits": {},
                        "fit_parameterization": "feasible_density_fraction",
                        "physical_boundary_hit": flagged,
                    }
                    with patch("run_nfw_bound_sweep.fit_halo", return_value=fit):
                        row = fit_for_sweep("example", None, model, 100.0)
                    saved = pd.read_csv(StringIO(pd.DataFrame([row]).to_csv(index=False)))
                    self.assertEqual(saved.loc[0, f"{model}_physical_boundary_hit"], flagged)
                    self.assertEqual(
                        saved.loc[0, f"{model}_fit_parameterization"],
                        "feasible_density_fraction",
                    )
                    self.assertEqual(row[f"{model}_fit_error"], "")

    def test_failure_leaves_diagnostics_unknown(self):
        with patch("run_nfw_bound_sweep.fit_halo", side_effect=ValueError("infeasible")):
            row = fit_for_sweep("example", None, "burkert", 100.0)
        self.assertEqual(row["burkert_fit_error"], "infeasible")
        self.assertEqual(row["burkert_physical_boundary_hit"], "")
        self.assertEqual(row["burkert_fit_parameterization"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
