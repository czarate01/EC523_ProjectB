"""Exact public aggregation plus the staff metric-backend boundary."""

from __future__ import annotations

import numpy as np

EPSILON = 1e-8


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    y = np.asarray(reference, dtype=np.float64)
    z = np.asarray(estimate, dtype=np.float64)
    if y.ndim != 1 or z.shape != y.shape or y.size == 0:
        raise ValueError("SI-SDR expects equal, non-empty 1D arrays")
    if not np.isfinite(y).all() or not np.isfinite(z).all():
        raise ValueError("signals must be finite")
    y = y - y.mean()
    z = z - z.mean()
    energy = float(np.dot(y, y))
    if energy <= EPSILON:
        raise ValueError("silent reference is not scoreable")
    alpha = float(np.dot(z, y)) / energy
    target = alpha * y
    residual = z - target
    return float(10.0 * np.log10((np.dot(target, target) + EPSILON) /
                                 (np.dot(residual, residual) + EPSILON)))


def normalize_pesq(value: float) -> float:
    return float(np.clip((value - 1.0) / 3.5, 0.0, 1.0))


def normalize_stoi(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def normalize_si_sdr(value: float) -> float:
    return float((np.clip(value, -10.0, 30.0) + 10.0) / 40.0)


def quality(pesq_values, stoi_values, si_sdr_values) -> dict[str, float]:
    p = np.asarray(pesq_values, dtype=np.float64)
    s = np.asarray(stoi_values, dtype=np.float64)
    r = np.asarray(si_sdr_values, dtype=np.float64)
    if not (p.ndim == s.ndim == r.ndim == 1 and len(p) == len(s) == len(r) and len(p)):
        raise ValueError("metric arrays must be non-empty 1D arrays of equal length")
    if not (np.isfinite(p).all() and np.isfinite(s).all() and np.isfinite(r).all()):
        raise ValueError("metric arrays must be finite")
    pn = float(np.mean(np.clip((p - 1.0) / 3.5, 0.0, 1.0)))
    sn = float(np.mean(np.clip(s, 0.0, 1.0)))
    rn = float(np.mean((np.clip(r, -10.0, 30.0) + 10.0) / 40.0))
    return {
        "pesq": float(np.mean(p)), "stoi": float(np.mean(s)),
        "si_sdr": float(np.mean(r)), "p_norm": pn, "s_norm": sn,
        "r_norm": rn, "quality": 0.40 * pn + 0.30 * sn + 0.30 * rn,
        "utterances": int(len(p)),
    }


def leaderboard(noisy_quality: float, clean_quality: float) -> float:
    if not (0.0 <= noisy_quality <= 1.0 and 0.0 <= clean_quality <= 1.0):
        raise ValueError("qualities must lie in [0,1]")
    return 100.0 * (0.85 * noisy_quality + 0.15 * clean_quality)

