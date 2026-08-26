"""Generate a runnable training config + commands for one tuning.csv row.

Usage:
    python scripts/gen_tuning_run.py tune_adamw_8k_lr0.0002 [--seed 42] [--nproc 2]

Reads the row from tuning.csv, writes a bergson `train` config under
/mnt/ssd-2/lucia/paper_runs/tuning/<run_id>_s<seed>/, mirrors it to
<repo>/configs/tuning/<run_id>_s<seed>.yaml (commit that copy with the claim), and prints
the exact commands to run. The config encodes every control from CONTROLS.md; the row
supplies what varies. One seed per point (see DECISIONS.md, tuning procedure step 4).

After the held-out number is recorded, delete <run_path>/checkpoints — training checkpoints
are large and tuning runs never reuse them. (The keep-checkpoints control applies to
experiment runs, not tuning runs.)
"""

import argparse
import csv
import math
import os
from pathlib import Path

import yaml

import sys
# `python -P` keeps a bergson checkout's cwd off sys.path, but it also drops
# this script's own directory -- put just that one entry back for the import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_config  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BERGSON = "/mnt/ssd-1/lucia/bergson-main-paper-429"
# Datasets are gitignored, so they live only in the original checkout -- deriving
# DATA from BERGSON breaks every config the moment BERGSON is repointed.
DATA = "/mnt/ssd-1/lucia/bergson-damping/runs/ekfac_vs_n/datasets"
LONDON = "/mnt/ssd-2/lucia/datasets_local"


def _dataset_for(run_id: str, n: int) -> str:
    """Corpus for this run. Keyed on the run id, not just the document count.

    A london sweep and a smollm2 sweep at the same n_docs are different corpora
    and the same filename pattern cannot serve both. Building the path from n
    alone gave london runs train_<n>k.hf -- smollm2 -- and nothing anywhere in
    the run reports the corpus, so it trains happily on the wrong data and the
    heldout number lands in the london table looking ordinary.
    """
    if "london" in run_id:
        path = f"{LONDON}/london_{n // 1000}k.hf"
    else:
        path = f"{DATA}/train_{n // 1000}k.hf"
    if not os.path.isdir(path):
        raise SystemExit(
            f"refusing: {run_id} resolves to {path}, which does not exist. "
            "A tuning run that trains on the wrong corpus is invisible in its "
            "own logs, so this fails here rather than later."
        )
    return path

PYTHON = "/home/lucia/envs/paper/bin/python"
RUNS = "/mnt/ssd-2/lucia/paper_runs/tuning"
# gpt2-family dropout knobs, pinned off (the control): configuring 0.0 beats relying on eval mode.
DROPOUT_OFF = "resid_pdrop=0.0,attn_pdrop=0.0,embd_pdrop=0.0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nproc", type=int, default=2)
    args = ap.parse_args()

    with open(REPO / "tuning.csv", newline="") as f:
        rows = {r["run_id"]: r for r in csv.DictReader(f)}
    row = rows.get(args.run_id)
    assert row, f"{args.run_id} not in tuning.csv"
    assert row["status"] != "blocked", f"{args.run_id} is blocked: {row['notes']}"
    assert row["model"] in ("gpt2", "gpt2-medium", "gpt2-large"), row["model"]

    bs, n, ep = int(row["batch_size"]), int(row["n_docs"]), int(row["num_epochs"])
    lr = float(row["lr"])
    ga = max(1, bs // (16 * args.nproc))
    run_path = f"{RUNS}/{args.run_id}_s{args.seed}"

    cfg = {
        "run_path": run_path,
        "steps": [{"train": {
            "run_path": run_path,
            "overwrite": True,
            "model": row["model"],
            "model_kwargs": DROPOUT_OFF,
            "precision": "fp32",
            "use_tf32_matmuls": False,
            "distributed": {"nnode": 1, "nproc_per_node": args.nproc},
            "data": {"dataset": _dataset_for(args.run_id, n),
                     "split": "train", "chunk_length": 0},
            "lr_schedule": {"lr": lr, "lr_scheduler_type": "polynomial",
                            "lr_start": 1e-6, "lr_end": lr / 10, "warmup_steps": 0.25},
            "batch_size": bs,
            "grad_accum_steps": ga,
            "num_epochs": ep,
            "seed": args.seed,
            "optimizer": row["optimizer"],
            "adam_beta1": 0.95, "adam_beta2": 0.975,
            "adam_eps": 1e-8, "eps_root": float(row["eps_root"]),
            "weight_decay": float(row["weight_decay"]),
            **({"max_grad_norm": float(row["max_grad_norm"])} if row["max_grad_norm"] else {}),
            "loss_reduction": "mean",
            "train_mode": False,
            "save_models": True,
            "save_optimizer_state": "none",
            # Train-only sweep: keep no trajectory. The default save_mode "sqrt"
            # wrote ~50 GB of checkpoints per 1000-step run; "interval" with an
            # out-of-range interval saves only the final state.
            "save_mode": "interval",
            "save_interval": 10**9,
        }}],
    }

    # logit_scale exists only on bergson feat/logit-scale (PR #433). Emitting it
    # unconditionally breaks EVERY row on older bergson: a no-op *value* does not
    # help when the *field* is unknown -- simple_parsing rejects the whole config
    # with "Couldn't instantiate class Train using init args". Emit only when it
    # actually deviates from the 1.0 control.
    scale = float(row["logit_scale"])
    if scale != 1.0:
        cfg["steps"][0]["train"]["logit_scale"] = scale

    # Launch from the tracked copy under configs/: a run directory is
    # disposable and has twice been swept out from under its own config.
    mirror = run_config.save(cfg, run_path, "tune.yaml")
    cfg_path = mirror

    steps = math.ceil(n * ep / bs)
    print(f"wrote {cfg_path}   ({steps} steps; ga={ga} at nproc={args.nproc})")
    print("tracked under configs/ — commit it with the claim")
    print("\n# 1. train:")
    print(f"cd /tmp && PYTHONNOUSERSITE=1 PYTHONPATH={BERGSON} "
          f"{PYTHON} -s -P -m bergson {cfg_path}")
    print("\n# 2. held-out loss (fill heldout_loss in tuning.csv via the builder):")
    print(f"PYTHONNOUSERSITE=1 {PYTHON} -s -P {REPO}/scripts/heldout_eval.py {run_path}/model")
    print("\n# 3. free the space (tuning runs never reuse checkpoints):")
    print(f"rm -rf {run_path}/checkpoints")


if __name__ == "__main__":
    main()
