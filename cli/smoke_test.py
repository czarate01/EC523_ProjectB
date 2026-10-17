#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("smoke_outputs"))
    args = ap.parse_args()
    run("make_demo.py", "--output", str(args.output / "demo"))
    run("enhance.py", "--checkpoint", str(args.output / "demo/model.safetensors"),
        "--input", str(args.output / "demo/noisy"),
        "--output", str(args.output / "enhanced"), "--device", "cpu")
    run("-m", "unittest", "discover", "-s", "tests", "-v")
    print("PASS: interface smoke test; no real speech and no performance claim.")


if __name__ == "__main__":
    main()

