#!/usr/bin/env python3
"""Train DCCRN on paired clean/noisy speech (EC523 Project B).

Baseline (paper DCCRN-E):
    python train.py --train-manifest TRAIN --dev-manifest DEV --out runs/dccrn_e

Experiments (one change each):
    --clean-prob 0.1      clean -> clean pairs (clean-speech preservation)
    --stft-weight 1.0     add a multi-resolution STFT magnitude loss
    --mask-bound 2.0      let the E-mode mask amplify up to 2x
    --masking-mode R      real/imag mask instead of the complex E mask
    --clstm               complex LSTM (DCCRN-CL style)

Sanity check (should reach a high SI-SNR on a handful of crops):
    python train.py --train-manifest TRAIN --overfit 10 --epochs 300 --out runs/overfit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "cli"))
from submission.model import CONFIG, DCCRN, pad_to_hop  # noqa: E402

from data import FullUtterances, PairedCrops, load_pairs  # noqa: E402


# ---------------------------------------------------------------- losses ---

def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-utterance SI-SNR in dB, [B, T] -> [B]. Means removed, as in the
    course's SI-SDR metric."""
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    scale = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    target = scale * ref
    noise = est - target
    return 10 * torch.log10(target.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps) + eps)


def multi_res_stft_loss(est, ref, sizes=((512, 128), (1024, 256), (2048, 512))):
    """Spectral convergence + log-magnitude L1, averaged over resolutions."""
    total = 0.0
    for n_fft, hop in sizes:
        win = torch.hann_window(n_fft, device=est.device)
        e = torch.stft(est, n_fft, hop, window=win, return_complex=True).abs().clamp(min=1e-7)
        r = torch.stft(ref, n_fft, hop, window=win, return_complex=True).abs().clamp(min=1e-7)
        sc = torch.linalg.norm(r - e) / torch.linalg.norm(r)
        mag = (torch.log(r) - torch.log(e)).abs().mean()
        total = total + sc + mag
    return total / len(sizes)


def compute_loss(est, clean, stft_weight):
    snr = si_snr(est, clean).mean()
    loss = -snr
    spec = torch.zeros((), device=est.device)
    if stft_weight > 0:
        spec = multi_res_stft_loss(est, clean)
        loss = loss + stft_weight * spec
    return loss, snr.detach(), spec.detach()


# --------------------------------------------------------------- helpers ---

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def model_config(args) -> dict:
    cfg = dict(CONFIG)
    cfg.update(masking_mode=args.masking_mode, use_clstm=args.clstm,
               mask_bound=args.mask_bound)
    if args.clstm:  # paper DCCRN-CL channel plan
        cfg["kernel_num"] = (32, 64, 128, 256, 256, 256)
    cfg["kernel_num"] = tuple(cfg["kernel_num"])
    return cfg


@torch.no_grad()
def evaluate(model, loader, device, stft_weight):
    model.eval()
    losses, snrs = [], []
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        length = noisy.shape[-1]
        est, _ = model(pad_to_hop(noisy, model.win_inc))
        loss, snr, _ = compute_loss(est[..., :length], clean, stft_weight)
        losses.append(loss.item())
        snrs.append(snr.item())
    return float(np.mean(losses)), float(np.mean(snrs))


def save_weights(model, path: Path) -> None:
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path))


# ------------------------------------------------------------------ main ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-manifest"); ap.add_argument("--train-clean"); ap.add_argument("--train-noisy")
    ap.add_argument("--dev-manifest"); ap.add_argument("--dev-clean"); ap.add_argument("--dev-noisy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=float, default=4.0, help="training crop length in seconds")
    ap.add_argument("--patience", type=int, default=10, help="early stopping, in epochs")
    ap.add_argument("--lr-patience", type=int, default=0,
                    help="epochs without dev improvement before halving the LR (paper: 0)")
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-dev", type=int, default=0, help="use only the first N dev utterances (0 = all)")
    ap.add_argument("--overfit", type=int, default=0, help="train and validate on the first N fixed crops")
    # experiment knobs
    ap.add_argument("--clean-prob", type=float, default=0.0)
    ap.add_argument("--stft-weight", type=float, default=0.0)
    ap.add_argument("--mask-bound", type=float, default=1.0)
    ap.add_argument("--masking-mode", default="E", choices=["E", "C", "R"])
    ap.add_argument("--clstm", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    train_pairs = load_pairs(args.train_manifest, args.train_clean, args.train_noisy)
    if args.overfit:
        train_pairs = train_pairs[: args.overfit]
        train_set = PairedCrops(train_pairs, args.crop, random_crop=False)
        dev_loader = DataLoader(PairedCrops(train_pairs, args.crop, random_crop=False), batch_size=1)
    else:
        train_set = PairedCrops(train_pairs, args.crop, random_crop=True, clean_prob=args.clean_prob)
        dev_pairs = load_pairs(args.dev_manifest, args.dev_clean, args.dev_noisy)
        if args.max_dev:
            dev_pairs = dev_pairs[: args.max_dev]
        dev_loader = DataLoader(FullUtterances(dev_pairs), batch_size=1, num_workers=min(args.workers, 2))
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=min(args.batch_size, len(train_set)), shuffle=True,
                              drop_last=not args.overfit, num_workers=args.workers, generator=gen,
                              pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)

    cfg = model_config(args)
    model = DCCRN(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # paper: "the learning rate ... will decay 0.5 when the validation loss goes up"
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=args.lr_patience, min_lr=1e-5)

    start_epoch, best, bad_epochs, gpu_seconds = 0, math.inf, 0, 0.0
    last = args.out / "last.pt"
    if last.exists():  # resume after an SCC time limit
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch, best, bad_epochs = state["epoch"] + 1, state["best"], state["bad_epochs"]
        gpu_seconds = state["gpu_seconds"]
        torch.set_rng_state(state["rng"])
        print(f"resumed at epoch {start_epoch}")
    else:
        (args.out / "config.json").write_text(json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()}, "model_config": cfg,
             "parameters": n_params, "torch": torch.__version__,
             "train_utterances": len(train_pairs)}, indent=2))
    print(f"parameters: {n_params:,}  train utterances: {len(train_pairs)}  device: {device}")

    log_path = args.out / "log.csv"
    new_log = not log_path.exists()
    log = log_path.open("a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["epoch", "train_loss", "train_si_snr", "train_stft", "dev_loss",
                         "dev_si_snr", "lr", "epoch_seconds", "peak_mem_gib"])

    for epoch in range(start_epoch, args.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        model.train()
        tl, ts, tp = [], [], []
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            est, _ = model(noisy)
            loss, snr, spec = compute_loss(est, clean, args.stft_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tl.append(loss.item()); ts.append(snr.item()); tp.append(spec.item())

        dev_loss, dev_snr = evaluate(model, dev_loader, device, args.stft_weight)
        if not args.overfit:  # overfit check: plain training, no LR decay or early stop
            sched.step(dev_loss)
        seconds = time.time() - t0
        gpu_seconds += seconds
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        lr = opt.param_groups[0]["lr"]
        writer.writerow([epoch, np.mean(tl), np.mean(ts), np.mean(tp), dev_loss, dev_snr, lr,
                         round(seconds, 1), round(peak, 3)])
        log.flush()
        print(f"epoch {epoch:3d}  train {np.mean(tl):8.3f} (SI-SNR {np.mean(ts):6.2f} dB)  "
              f"dev {dev_loss:8.3f} (SI-SNR {dev_snr:6.2f} dB)  lr {lr:.1e}  "
              f"{seconds:6.1f}s  peak {peak:.2f} GiB", flush=True)

        if dev_loss < best:
            best, bad_epochs = dev_loss, 0
            save_weights(model, args.out / "best.safetensors")  # model-selection rule: best dev loss
        else:
            bad_epochs += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "bad_epochs": bad_epochs,
                    "gpu_seconds": gpu_seconds, "rng": torch.get_rng_state()}, last)
        if bad_epochs >= args.patience and not args.overfit:
            print(f"early stop: no dev improvement for {args.patience} epochs")
            break

    log.close()
    print(f"best dev loss {best:.3f}; total training time {gpu_seconds / 3600:.2f} GPU-hours")


if __name__ == "__main__":
    main()
