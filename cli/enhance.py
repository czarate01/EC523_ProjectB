#!/usr/bin/env python3
"""Staff-owned inference runner for Project B."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from common import SAMPLE_RATE, read_wav, sha256, wav_paths, write_json, write_wav


def load_submission(source: Path):
    spec = importlib.util.spec_from_file_location("student_submission", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=Path("submission/model.py"))
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.output.resolve() == args.input.resolve():
        raise ValueError("input and output directories must differ")
    module = load_submission(args.source)
    model = module.build_enhancer(SAMPLE_RATE)
    state = load_file(str(args.checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    records = []
    total_audio = 0.0
    started = time.perf_counter()
    for src in wav_paths(args.input):
        rel = src.relative_to(args.input)
        if ".." in rel.parts:
            raise ValueError("unsafe relative path")
        rate, x = read_wav(src)
        if rate != SAMPLE_RATE:
            raise ValueError(f"{src}: expected {SAMPLE_RATE} Hz, got {rate}")
        tensor = torch.from_numpy(x).view(1, -1).to(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            y = model.enhance(tensor, SAMPLE_RATE)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        if not isinstance(y, torch.Tensor) or tuple(y.shape) != tuple(tensor.shape):
            raise ValueError(f"{rel}: model must return shape {tuple(tensor.shape)}")
        out = y.detach().to("cpu", torch.float32).numpy().reshape(-1)
        dst = args.output / rel
        write_wav(dst, SAMPLE_RATE, out)
        duration = len(x) / SAMPLE_RATE
        total_audio += duration
        records.append({
            "id": rel.as_posix(), "samples": len(x), "seconds": duration,
            "input_sha256": sha256(src), "output_sha256": sha256(dst),
            "peak": float(np.max(np.abs(out))), "dc": float(np.mean(out)),
            "elapsed_seconds": elapsed,
        })

    wall = time.perf_counter() - started
    report = {
        "sample_rate": SAMPLE_RATE,
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_sha256": sha256(args.source),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device), "audio_seconds": total_audio,
        "wall_seconds": wall, "rtf": wall / total_audio,
        "files": records,
    }
    write_json(args.output / "run.json", report)
    print(json.dumps({k: report[k] for k in ("parameters", "rtf", "audio_seconds")}, indent=2))


if __name__ == "__main__":
    main()

