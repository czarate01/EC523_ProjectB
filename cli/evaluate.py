#!/usr/bin/env python3
"""Reference evaluator for the public PESQ/STOI/SI-SDR score."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import SAMPLE_RATE, read_wav, wav_paths, write_json
from metrics import quality, si_sdr


def metric_functions():
    try:
        from pesq import pesq
        from pystoi import stoi
    except ImportError as exc:
        raise RuntimeError(
            "evaluation requires the pinned pesq and pystoi packages"
        ) from exc
    return pesq, stoi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--enhanced", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    pesq_fn, stoi_fn = metric_functions()

    ids = [p.relative_to(args.clean) for p in wav_paths(args.clean)]
    expected = {p.as_posix() for p in ids}
    actual = {p.relative_to(args.enhanced).as_posix() for p in wav_paths(args.enhanced)}
    if actual != expected:
        raise ValueError(f"output IDs differ: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")

    rows, pv, sv, rv = [], [], [], []
    for rel in ids:
        cr, clean = read_wav(args.clean / rel)
        er, enhanced = read_wav(args.enhanced / rel)
        if cr != SAMPLE_RATE or er != SAMPLE_RATE or clean.shape != enhanced.shape:
            raise ValueError(f"{rel}: sample rate or length mismatch")
        if float(abs(enhanced).max()) > 1.0:
            raise ValueError(f"{rel}: output peak exceeds 1.0")
        p = float(pesq_fn(SAMPLE_RATE, clean, enhanced, "wb"))
        s = float(stoi_fn(clean, enhanced, SAMPLE_RATE, extended=False))
        r = si_sdr(clean, enhanced)
        rows.append({"id": rel.as_posix(), "pesq": p, "stoi": s, "si_sdr": r})
        pv.append(p); sv.append(s); rv.append(r)
    result = quality(pv, sv, rv)
    result["files"] = rows
    write_json(args.report, result)
    print(f"A(D) = {result['quality']:.8f}")


if __name__ == "__main__":
    main()
