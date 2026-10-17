#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import leaderboard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy-report", type=Path, required=True)
    ap.add_argument("--clean-report", type=Path, required=True)
    args = ap.parse_args()
    noisy = json.loads(args.noisy_report.read_text())
    clean = json.loads(args.clean_report.read_text())
    score = leaderboard(float(noisy["quality"]), float(clean["quality"]))
    print(f"Score: {score:.8f}")


if __name__ == "__main__":
    main()

