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
from halo_fit_selection import fit_best_halo


class TestProductionFitSelection(unittest.TestCase):
    def test_plot_receives_selected_result_without_an_extra_fit(self):
        def result(score):
            return {
                "chi_squared": score, "optimizer_status": 1,
                "radius_kpc": np.array([1., 2., 3.]),
                "model_kms": np.array([score, score, score]),
            }
        trials = [result(10.), result(2.)] + [result(5.) for _ in range(8)]
        with patch("halo_fit_selection.fit_halo", side_effect=trials) as fitter, \
             patch("halo_fit_selection.plot_halo_fit") as plot:
            selected = fit_best_halo("example", df=pd.DataFrame(), plot=True,
                                     output_path="unused.png", show=False)
        self.assertEqual(fitter.call_count, 10)
        self.assertEqual(selected["selection_selected_trial"], 1)
        plot.assert_called_once()
        self.assertIs(plot.call_args.args[1], selected)
        np.testing.assert_array_equal(selected["model_kms"], [2., 2., 2.])

    def test_main_galaxy_workflow_uses_selected_results_for_comparison(self):
        data = pd.DataFrame({"ID": ["CamB"]})
        burkert = {"selected": "burkert"}
        nfw = {"selected": "nfw"}
        with TemporaryDirectory() as directory:
            with patch.object(pipeline_main, "OUTPUT_DIR", Path(directory)), \
                 patch.object(pipeline_main, "fit_best_halo", side_effect=[burkert, nfw]) as fitter, \
                 patch.object(pipeline_main, "plot_rotation_curve"), \
                 patch.object(pipeline_main, "plot_halo_comparison") as compare, \
                 patch("builtins.input", side_effect=["CamB", "100"]), \
                 redirect_stdout(StringIO()):
                pipeline_main.run_galaxy_workflow(data)
        self.assertEqual(fitter.call_count, 2)
        self.assertTrue(all(call.kwargs["plot"] for call in fitter.call_args_list))
        compare.assert_called_once()
        self.assertIs(compare.call_args.args[1], burkert)
        self.assertIs(compare.call_args.args[2], nfw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
