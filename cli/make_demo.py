#!/usr/bin/env python3
"""Create synthetic, non-speech interface data and untrained weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors.torch import save_file

from common import SAMPLE_RATE, write_wav
from submission.model import build_enhancer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(523)
    for index, seconds in enumerate((1.0, 1.35), 1):
        t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
        envelope = np.sin(np.pi * np.minimum(t / seconds, 1.0)) ** 2
        clean = envelope * (0.20 * np.sin(2*np.pi*180*t) + 0.08 * np.sin(2*np.pi*360*t))
        noisy = clean + 0.035 * rng.standard_normal(len(t), dtype=np.float32)
        write_wav(args.output / "clean" / f"demo_{index}.wav", SAMPLE_RATE, clean)
        write_wav(args.output / "noisy" / f"demo_{index}.wav", SAMPLE_RATE, noisy)
    model = build_enhancer(SAMPLE_RATE)
    save_file(model.state_dict(), str(args.output / "model.safetensors"))
    print(args.output)


if __name__ == "__main__":
    main()

