import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from common import SAMPLE_RATE, read_wav, write_wav
from submission.model import build_enhancer


class InterfaceTests(unittest.TestCase):
    def test_exact_length_and_finite(self):
        model = build_enhancer(SAMPLE_RATE).eval()
        for length in (200, 16_000, 19_321):
            x = torch.randn(1, length) * 0.05
            with torch.inference_mode():
                y = model.enhance(x, SAMPLE_RATE)
            self.assertEqual(tuple(y.shape), (1, length))
            self.assertTrue(torch.isfinite(y).all())

    def test_wrong_sample_rate_rejected(self):
        model = build_enhancer(SAMPLE_RATE)
        with self.assertRaises(ValueError):
            model.enhance(torch.zeros(1, 1000), 8_000)

    def test_wav_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.wav"
            x = np.linspace(-0.2, 0.2, 1000, dtype=np.float32)
            write_wav(p, SAMPLE_RATE, x)
            rate, y = read_wav(p)
            self.assertEqual(rate, SAMPLE_RATE)
            np.testing.assert_array_equal(x, y)

    def test_write_refuses_clipping(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                write_wav(Path(d) / "bad.wav", SAMPLE_RATE, np.array([1.01], np.float32))


if __name__ == "__main__":
    unittest.main()

