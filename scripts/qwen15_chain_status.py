#!/usr/bin/env python3
"""Qwen2.5-1.5B tuning/heldout gate status.

This makes the Qwen chain explicit for future agents:
  train sweep -> heldout_eval.py -> bracket/interior LR selection -> downstream rows.
Training logs do not contain heldout loss, so completed `retrained/base` directories are
not enough to launch the next phase.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("/mnt/ssd-2/lucia")
TUNING = ROOT / "paper_runs/tuning"
LOGS = TUNING / "_logs"
HELDOUT = ROOT / "datasets_local/heldout_4k_qwen_v2.hf"
SIZES = ("4k", "8k", "16k", "32k", "64k", "128k", "256k")
LRS = ("1e-05", "2e-05", "5e-05", "0.0001", "0.0002", "0.0005")
LR_VALUE = {
    "1e-05": 1e-5,
    "2e-05": 2e-5,
    "5e-05": 5e-5,
    "0.0001": 1e-4,
    "0.0002": 2e-4,
    "0.0005": 5e-4,
}


def completed_base(size: str, lr: str) -> Path:
    return TUNING / f"tune_qwen15b_{size}_bs256_lr{lr}_s42/retrained/base"


def eval_losses() -> dict[str, float]:
    losses: dict[str, float] = {}
    pat = re.compile(r"(tune_qwen15b_[^/\s]+/retrained/base)\s+([0-9]+\.[0-9]+)")
    for log in sorted(LOGS.glob("qwen15b_heldout*.log")):
        text = log.read_text(errors="ignore")
        for rel, val in pat.findall(text):
            run = rel.split("/", 1)[0]
            losses[run] = float(val)
    return losses


def main() -> None:
    losses = eval_losses()
    print(f"heldout: {HELDOUT}")
    print("\nSweep status:")
    for size in SIZES:
        measured = []
        for lr in LRS:
            run = f"tune_qwen15b_{size}_bs256_lr{lr}_s42"
            base = completed_base(size, lr)
            state = "measured" if run in losses else "base" if base.is_dir() else "missing/running"
            suffix = f" loss={losses[run]:.4f}" if run in losses else ""
            print(f"  {size:>4} {lr:>7}: {state}{suffix}")
            if run in losses:
                measured.append((LR_VALUE[lr], lr, losses[run]))
        if len(measured) >= 3:
            measured.sort()
            best_i = min(range(len(measured)), key=lambda i: measured[i][2])
            best = measured[best_i]
            bracketed = 0 < best_i < len(measured) - 1
            verdict = "INTERIOR: launchable" if bracketed else "ENDPOINT: extend one octave first"
            print(f"       best lr={best[1]} heldout={best[2]:.4f} -> {verdict}")
        else:
            print("       blocked: run heldout_eval.py for every completed base first")
    print("\nCanonical eval command:")
    print("  PYTHONNOUSERSITE=1 /mnt/ssd-2/lucia/envs/paper/bin/python -s -P scripts/heldout_eval.py \\")
    print(f"    --heldout {HELDOUT} --device cuda:0 --batch_size 8 MODEL_DIR...")


if __name__ == "__main__":
    main()
