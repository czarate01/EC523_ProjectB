#!/usr/bin/env python3
"""Check paired data before training, and record each pair's SNR.

For every pair: sample rate, equal length, finite values, peak level,
time offset between clean and noisy (should be 0), and the input SNR
(clean energy vs. noise = noisy - clean). Writes a CSV you can later pass
to analyze.py for per-SNR breakdowns.

    python scripts/check_pairs.py --manifest DEV --out dev_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import SAMPLE_RATE, load_pairs, read_audio  # noqa: E402


def lag_samples(clean, noisy, max_lag=800, seconds=3.0):
    n = min(len(clean), int(seconds * SAMPLE_RATE))
    c = correlate(noisy[:n], clean[:n], mode="full", method="fft")
    mid = n - 1
    window = c[mid - max_lag: mid + max_lag + 1]
    return int(np.argmax(window)) - max_lag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest"); ap.add_argument("--clean-dir"); ap.add_argument("--noisy-dir")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pairs = load_pairs(args.manifest, args.clean_dir, args.noisy_dir)
    problems, rows = [], []
    for item in pairs:
        try:
            clean, noisy = read_audio(item["clean"]), read_audio(item["noisy"])
        except ValueError as exc:
            problems.append(f"{item['id']}: {exc}")
            continue
        if len(clean) != len(noisy):
            problems.append(f"{item['id']}: length {len(clean)} vs {len(noisy)}")
            continue
        if not (np.isfinite(clean).all() and np.isfinite(noisy).all()):
            problems.append(f"{item['id']}: non-finite samples")
            continue
        noise = noisy - clean
        snr = 10 * np.log10((clean ** 2).sum() / max((noise ** 2).sum(), 1e-12))
        lag = lag_samples(clean, noisy)
        if lag != 0:
            problems.append(f"{item['id']}: clean/noisy offset of {lag} samples")
        rows.append({"id": item["id"], "seconds": round(len(clean) / SAMPLE_RATE, 3),
                     "snr_db": round(float(snr), 2), "lag": lag,
                     "peak_clean": round(float(np.abs(clean).max()), 4),
                     "peak_noisy": round(float(np.abs(noisy).max()), 4)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    snrs = np.array([r["snr_db"] for r in rows])
    secs = np.array([r["seconds"] for r in rows])
    print(f"pairs checked: {len(pairs)}, ok: {len(rows)}, problems: {len(problems)}")
    print(f"duration: total {secs.sum() / 3600:.2f} h, median {np.median(secs):.2f} s")
    print("input SNR (dB) percentiles 5/25/50/75/95:",
          np.round(np.percentile(snrs, [5, 25, 50, 75, 95]), 1))
    for line in problems[:20]:
        print("  PROBLEM", line)
    if len(problems) > 20:
        print(f"  ... and {len(problems) - 20} more")


if __name__ == "__main__":
    main()
