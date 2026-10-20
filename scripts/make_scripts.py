#!/usr/bin/env python3
"""Make a speaker-held-out train/dev split of the course training data.

Reads the clean/noisy folders and the dataset's log file
(one line per utterance: "wavfile noise SNR"), then writes
    <out>/train.csv and <out>/dev.csv   columns: id, clean, noisy, speaker, noise, snr_db
and, with --link-dir, folders of symlinks
    <link-dir>/clean/*.wav and <link-dir>/noisy/*.wav   (dev speakers only)
so the official enhance.py / evaluate.py can run on the dev set.
Nothing is written into the shared course folder.

    python scripts/make_split.py \
        --root /projectnb/ec523kb/projects/DeNoise/16khz \
        --log  /projectnb/ec523kb/projects/DeNoise/log_trainset_28spk.txt \
        --dev-speakers p226,p287 --out meta --link-dir dev
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="folder holding the clean_/noisy_ subfolders")
    ap.add_argument("--clean-sub", default="clean_trainset_28spk_wav")
    ap.add_argument("--noisy-sub", default="noisy_trainset_28spk_wav")
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--dev-speakers", default="p226,p287")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--link-dir", type=Path)
    args = ap.parse_args()

    clean_dir, noisy_dir = args.root / args.clean_sub, args.root / args.noisy_sub
    dev_speakers = set(args.dev_speakers.split(","))

    info = {}
    for line in args.log.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0] if parts[0].endswith(".wav") else parts[0] + ".wav"
        info[name] = (parts[1], parts[-1])

    rows, problems = {"train": [], "dev": []}, []
    for clean in sorted(clean_dir.glob("*.wav")):
        name = clean.name
        noisy = noisy_dir / name
        if not noisy.exists():
            problems.append(f"{name}: no noisy file")
            continue
        if name not in info:
            problems.append(f"{name}: not in log file")
            continue
        speaker = name.split("_")[0]
        noise, snr = info[name]
        split = "dev" if speaker in dev_speakers else "train"
        rows[split].append({"id": name, "clean": str(clean.resolve()), "noisy": str(noisy.resolve()),
                            "speaker": speaker, "noise": noise, "snr_db": snr})
    extra_noisy = {p.name for p in noisy_dir.glob("*.wav")} - {p.name for p in clean_dir.glob("*.wav")}
    problems += [f"{n}: no clean file" for n in sorted(extra_noisy)]

    missing = dev_speakers - {r["speaker"] for r in rows["dev"]}
    if missing:
        raise SystemExit(f"dev speakers not found in the data: {sorted(missing)}")

    args.out.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        with (args.out / f"{split}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "clean", "noisy", "speaker", "noise", "snr_db"])
            writer.writeheader()
            writer.writerows(items)

    if args.link_dir:
        for kind in ("clean", "noisy"):
            d = args.link_dir / kind
            d.mkdir(parents=True, exist_ok=True)
            for r in rows["dev"]:
                link = d / r["id"]
                if not link.exists():
                    os.symlink(r[kind], link)

    for split, items in rows.items():
        spk = {r["speaker"] for r in items}
        print(f"{split}: {len(items)} utterances, {len(spk)} speakers")
        print("   SNR counts:", dict(sorted(Counter(r["snr_db"] for r in items).items(),
                                          key=lambda kv: float(kv[0]))))
        print("   noise counts:", dict(sorted(Counter(r["noise"] for r in items).items())))
    print(f"problems: {len(problems)}")
    for p in problems[:20]:
        print("  ", p)


if __name__ == "__main__":
    main()