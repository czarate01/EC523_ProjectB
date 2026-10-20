#!/usr/bin/env python3
"""Compare systems from cli/evaluate.py reports: overall table, per-SNR
breakdown, and paired bootstrap 95% intervals against a baseline.

    python scripts/analyze.py \
        --report noisy=reports/noisy.json \
        --report dccrn_e=reports/dccrn_e.json \
        --report clean_prob=reports/clean_prob.json \
        --baseline dccrn_e --meta dev_pairs.csv --out results.md

--meta is the CSV written by check_pairs.py (needs columns id and snr_db;
any other column, e.g. a noise type, can be used with --group-col).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = ["pesq", "stoi", "si_sdr", "composite"]


def composite(p, s, r):
    """Per-utterance version of the course's A = 0.4 P + 0.3 S + 0.3 R.
    Its mean over utterances equals the course's A(D)."""
    return (0.40 * np.clip((p - 1.0) / 3.5, 0, 1) + 0.30 * np.clip(s, 0, 1)
            + 0.30 * (np.clip(r, -10, 30) + 10) / 40)


def load_report(path: Path) -> dict[str, dict[str, float]]:
    files = json.loads(path.read_text())["files"]
    out = {}
    for f in files:
        m = {"pesq": f["pesq"], "stoi": f["stoi"], "si_sdr": f["si_sdr"]}
        m["composite"] = float(composite(np.float64(f["pesq"]), np.float64(f["stoi"]),
                                         np.float64(f["si_sdr"])))
        out[f["id"]] = m
    return out


def paired_bootstrap(diff: np.ndarray, n: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return diff.mean(), lo, hi


def snr_group(snr: float, edges: list[float]) -> str:
    labels = [f"< {edges[0]:g}"] + [f"{a:g} to {b:g}" for a, b in zip(edges, edges[1:])] + [f">= {edges[-1]:g}"]
    return labels[int(np.searchsorted(edges, snr, side="right"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="append", required=True, help="name=path/to/report.json")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--snr-edges", default="2.5,7.5,12.5", help="SNR bin edges in dB")
    ap.add_argument("--group-col", help="extra --meta column to break down by, e.g. noise type")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    reports = {}
    for spec in args.report:
        name, path = spec.split("=", 1)
        reports[name] = load_report(Path(path))
    ids = sorted(set.intersection(*(set(r) for r in reports.values())))
    if any(len(r) != len(ids) for r in reports.values()):
        raise ValueError("reports cover different utterances; compare on the same set")
    vals = {n: {m: np.array([r[i][m] for i in ids]) for m in METRICS} for n, r in reports.items()}

    lines = [f"Utterances: {len(ids)}\n", "## Overall (means)\n",
             "| system | PESQ | STOI | SI-SDR (dB) | composite A |", "|---|---|---|---|---|"]
    for n, v in vals.items():
        lines.append(f"| {n} | {v['pesq'].mean():.3f} | {v['stoi'].mean():.4f} | "
                     f"{v['si_sdr'].mean():.2f} | {v['composite'].mean():.4f} |")

    lines += ["", f"## Difference vs. {args.baseline} (paired bootstrap 95% CI over utterances)\n",
              "| system | metric | mean diff | 95% CI |", "|---|---|---|---|"]
    base = vals[args.baseline]
    for n, v in vals.items():
        if n == args.baseline:
            continue
        for m in METRICS:
            d, lo, hi = paired_bootstrap(v[m] - base[m])
            lines.append(f"| {n} | {m} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")

    if args.meta:
        with args.meta.open() as f:
            meta = {row["id"]: row for row in csv.DictReader(f)}
        missing = [i for i in ids if i not in meta]
        if missing:
            raise ValueError(f"{len(missing)} ids missing from --meta, e.g. {missing[:3]}")
        edges = [float(x) for x in args.snr_edges.split(",")]
        groupings = [("input SNR (dB)", [snr_group(float(meta[i]["snr_db"]), edges) for i in ids])]
        if args.group_col:
            groupings.append((args.group_col, [meta[i][args.group_col] for i in ids]))
        for title, groups in groupings:
            groups = np.array(groups)
            lines += ["", f"## Composite A by {title}\n",
                      "| group | n | " + " | ".join(vals) + " |", "|---|---|" + "---|" * len(vals)]
            for g in sorted(set(groups)):
                sel = groups == g
                cells = " | ".join(f"{v['composite'][sel].mean():.4f}" for v in vals.values())
                lines.append(f"| {g} | {sel.sum()} | {cells} |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
