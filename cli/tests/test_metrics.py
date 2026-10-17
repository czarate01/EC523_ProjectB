import unittest

import numpy as np

from metrics import leaderboard, normalize_pesq, normalize_si_sdr, quality, si_sdr


class MetricTests(unittest.TestCase):
    def test_identical_has_large_si_sdr(self):
        x = np.sin(np.linspace(0, 20, 2000))
        self.assertGreater(si_sdr(x, x), 80.0)

    def test_scale_invariant(self):
        x = np.sin(np.linspace(0, 20, 2000))
        estimate = x + 0.01 * np.cos(np.linspace(0, 70, 2000))
        self.assertAlmostEqual(si_sdr(x, estimate), si_sdr(x, 0.25*estimate), places=4)

    def test_silent_reference_rejected(self):
        with self.assertRaises(ValueError):
            si_sdr(np.zeros(20), np.zeros(20))

    def test_shape_rejected(self):
        with self.assertRaises(ValueError):
            si_sdr(np.ones(20), np.ones(19))

    def test_normalization_caps(self):
        self.assertEqual(normalize_pesq(1.0), 0.0)
        self.assertEqual(normalize_pesq(4.5), 1.0)
        self.assertEqual(normalize_si_sdr(-10.0), 0.0)
        self.assertEqual(normalize_si_sdr(30.0), 1.0)

    def test_quality_known(self):
        q = quality([2.75], [0.8], [10.0])
        self.assertAlmostEqual(q["p_norm"], 0.5)
        self.assertAlmostEqual(q["r_norm"], 0.5)
        self.assertAlmostEqual(q["quality"], 0.59)

    def test_utterance_mean_before_composite(self):
        q = quality([1.0, 4.5], [0.0, 1.0], [-10.0, 30.0])
        self.assertAlmostEqual(q["quality"], 0.5)

    def test_leaderboard(self):
        self.assertAlmostEqual(leaderboard(0.6, 0.9), 64.5)

    def test_bad_quality_rejected(self):
        with self.assertRaises(ValueError):
            quality([], [], [])
        with self.assertRaises(ValueError):
            leaderboard(1.1, 0.5)


if __name__ == "__main__":
    unittest.main()
