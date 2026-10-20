"""Paired clean/noisy data loading for EC523 Project B.

The staff manifest format is not known yet. `load_pairs` accepts either
  * a manifest (.csv/.tsv/.json/.jsonl) with one column whose name contains
    "clean" and one whose name contains "noisy" (paths absolute or relative
    to the manifest's folder), plus an optional "id" column, or
  * two folders, clean_dir and noisy_dir, with matching relative file names.
Adjust `load_pairs` once you see the real manifest.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset

SAMPLE_RATE = 16_000


def read_audio(path) -> np.ndarray:
    rate, audio = wavfile.read(path)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {rate}")
    if audio.ndim != 1:
        raise ValueError(f"{path}: expected mono, got shape {audio.shape}")
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.float32:
        return audio
    raise ValueError(f"{path}: unsupported dtype {audio.dtype}")


def _read_rows(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
        return data if isinstance(data, list) else list(data.values())
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",\t;")
    return list(csv.DictReader(text.splitlines(), dialect=dialect))


def _find_key(keys, word):
    hits = [k for k in keys if word in k.lower()]
    if len(hits) != 1:
        raise ValueError(f"manifest needs exactly one column containing '{word}', found {hits}")
    return hits[0]


def load_pairs(manifest=None, clean_dir=None, noisy_dir=None) -> list[dict]:
    """Return a list of {"id", "clean", "noisy"} dicts."""
    pairs = []
    if manifest:
        manifest = Path(manifest)
        rows = _read_rows(manifest)
        ck, nk = _find_key(rows[0], "clean"), _find_key(rows[0], "noisy")
        ik = next((k for k in rows[0] if k.lower() in ("id", "utt_id", "utterance_id")), None)
        for row in rows:
            clean, noisy = Path(row[ck]), Path(row[nk])
            clean = clean if clean.is_absolute() else manifest.parent / clean
            noisy = noisy if noisy.is_absolute() else manifest.parent / noisy
            pairs.append({"id": row[ik] if ik else noisy.stem, "clean": clean, "noisy": noisy})
    elif clean_dir and noisy_dir:
        clean_dir, noisy_dir = Path(clean_dir), Path(noisy_dir)
        for noisy in sorted(noisy_dir.rglob("*.wav")):
            rel = noisy.relative_to(noisy_dir)
            clean = clean_dir / rel
            if not clean.exists():
                raise FileNotFoundError(f"no clean file for {rel}")
            pairs.append({"id": rel.as_posix(), "clean": clean, "noisy": noisy})
    else:
        raise ValueError("give a manifest or both clean_dir and noisy_dir")
    if not pairs:
        raise ValueError("no pairs found")
    return pairs


class PairedCrops(Dataset):
    """Same-index crops of clean/noisy pairs.

    Training: a random crop of `crop_seconds` (zero-padded if shorter).
    `clean_prob` > 0 replaces the noisy input with the clean signal for
    that fraction of examples (the clean-preservation experiment).
    """

    def __init__(self, pairs, crop_seconds=4.0, random_crop=True, clean_prob=0.0):
        self.pairs = pairs
        self.n = int(round(crop_seconds * SAMPLE_RATE))
        self.random_crop, self.clean_prob = random_crop, clean_prob

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        item = self.pairs[i]
        clean, noisy = read_audio(item["clean"]), read_audio(item["noisy"])
        if len(clean) != len(noisy):
            raise ValueError(f"{item['id']}: clean/noisy lengths differ")
        n = self.n
        if len(clean) > n:
            start = int(torch.randint(0, len(clean) - n + 1, ())) if self.random_crop else 0
            clean, noisy = clean[start:start + n], noisy[start:start + n]
        else:
            clean = np.pad(clean, (0, n - len(clean)))
            noisy = np.pad(noisy, (0, n - len(noisy)))
        if self.clean_prob > 0 and float(torch.rand(())) < self.clean_prob:
            noisy = clean.copy()
        return torch.from_numpy(noisy.copy()), torch.from_numpy(clean.copy())


class FullUtterances(Dataset):
    """Whole utterances for validation (use batch_size=1)."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        item = self.pairs[i]
        clean, noisy = read_audio(item["clean"]), read_audio(item["noisy"])
        if len(clean) != len(noisy):
            raise ValueError(f"{item['id']}: clean/noisy lengths differ")
        return torch.from_numpy(noisy.copy()), torch.from_numpy(clean.copy())
