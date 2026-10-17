from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 16_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wav_paths(root: Path) -> list[Path]:
    paths = sorted(p for p in root.rglob("*.wav") if p.is_file())
    if not paths:
        raise ValueError(f"no WAV files found under {root}")
    return paths


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    rate, audio = wavfile.read(path)
    if audio.ndim != 1:
        raise ValueError(f"{path}: expected mono WAV, got shape {audio.shape}")
    if audio.dtype == np.int16:
        x = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.float32:
        x = audio
    else:
        raise ValueError(f"{path}: expected int16 or float32 PCM, got {audio.dtype}")
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError(f"{path}: empty or non-finite audio")
    return int(rate), x.astype(np.float32, copy=False)


def write_wav(path: Path, rate: int, audio: np.ndarray) -> None:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("output must be a finite, non-empty mono array")
    if float(np.max(np.abs(x))) > 1.0:
        raise ValueError("output peak exceeds 1.0; runner will not clip or normalize")
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, rate, x)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

