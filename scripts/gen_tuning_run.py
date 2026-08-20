"""Generate a runnable training config + commands for one tuning.csv row.

Usage:
    python scripts/gen_tuning_run.py tune_adamw_8k_lr0.0002 [--seed 42] [--nproc 2]

Reads the row from tuning.csv, writes a bergson `train` config under
/mnt/ssd-2/lucia/paper_runs/tuning/<run_id>_s<seed>/, and prints the exact commands to run.
The config encodes every control from CONTROLS.md; the row supplies what varies. Groups whose
runs are 63 steps or fewer need two seeds (42 and 43) — run the command once per seed and
record the mean held-out loss in the row.

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

REPO = Path(__file__).resolve().parent.parent
BERGSON = "/mnt/ssd-1/lucia/bergson-damping"
DATA = f"{BERGSON}/runs/ekfac_vs_n/datasets"
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
            "data": {"dataset": f"{DATA}/train_{n // 1000}k.hf",
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
        }}],
    }

    os.makedirs(run_path, exist_ok=True)
    cfg_path = f"{run_path}/tune.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    steps = math.ceil(n * ep / bs)
    two_seed = steps <= 63
    print(f"wrote {cfg_path}   ({steps} steps; ga={ga} at nproc={args.nproc})")
    print(f"\n# 1. train ({'run for --seed 42 AND 43, record the MEAN' if two_seed else 'single seed'}):")
    print(f"PYTHONPATH={BERGSON} bergson {cfg_path}")
    print("\n# 2. held-out loss (fill heldout_loss in tuning.csv via the builder):")
    print(f"python {REPO}/scripts/heldout_eval.py {run_path}/model")
    print("\n# 3. free the space (tuning runs never reuse checkpoints):")
    print(f"rm -rf {run_path}/checkpoints")


if __name__ == "__main__":
    main()
