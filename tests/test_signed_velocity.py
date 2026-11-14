import sys
import unittest
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from burkert_halo_fit import total_velocity_from_squared


class TestSignedVelocity(unittest.TestCase):
    def test_negative_baryonic_contribution_with_positive_total(self):
        # -9 + 25 = 16, so the total speed is 4 km/s.
        result = total_velocity_from_squared([-9.0, 16.0], [25.0, 9.0])
        np.testing.assert_allclose(result, [4.0, 5.0])

    def test_zero_total_is_allowed(self):
        self.assertEqual(total_velocity_from_squared(-9.0, 9.0), 0.0)

    def test_negative_total_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            total_velocity_from_squared(-9.0, 4.0)

    def test_nonfinite_total_is_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    total_velocity_from_squared(value, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
