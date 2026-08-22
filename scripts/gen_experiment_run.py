"""Generate a runnable bank+MAGIC config for one experiments.csv planned row.

Usage:
    python scripts/gen_experiment_run.py plan_adam_eps1e17_8k_bs256 [--nproc 4]

Reads the row, writes a bergson `magic` pipeline config (base training with
checkpoints kept, 100 leave-1%-out retrains, per-query MAGIC over query_20) to
/mnt/ssd-2/lucia/paper_runs/experiments/<run_id>/, mirrors it to
<repo>/configs/experiments/, and prints the exact commands. Every control from
CONTROLS.md is encoded; the row supplies what varies (incl. its tuned lr).

IMPORTANT execution rules (see NODES.md):
- Run from a directory that is NOT a bergson checkout, with `python -P`:
  a bergson repo as cwd silently shadows PYTHONPATH and you will run the
  wrong code version.
- Record nproc: bit-exact reuse of the bank later requires the same world size.
- Disk: a run writes ~28 GB of checkpoints plus ~0.5 GB per retrained model.
  Check `df` on the target volume first.
"""

import argparse
import csv
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
BERGSON = "/mnt/ssd-1/lucia/bergson-main-paper-429"
# Datasets are gitignored, so they exist ONLY in the original checkout -- a fresh
# worktree has the code but not the data. Deriving DATA from BERGSON therefore
# breaks every generated config the moment BERGSON is repointed at a new pinned
# worktree. Code path and data path are independent; pin the data explicitly.
DATA = "/mnt/ssd-1/lucia/bergson-damping/runs/ekfac_vs_n/datasets"
RUNS = "/mnt/ssd-2/lucia/paper_runs/experiments"
DROPOUT_OFF = "resid_pdrop=0.0,attn_pdrop=0.0,embd_pdrop=0.0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--nproc", type=int, default=4)
    args = ap.parse_args()

    with open(REPO / "experiments.csv", newline="") as f:
        rows = {r["run_id"]: r for r in csv.DictReader(f)}
    row = rows.get(args.run_id)
    assert row, f"{args.run_id} not in experiments.csv"
    # planned rows, and partial rows whose bank metrics are missing (e.g. the
    # rows reset by the D15 strike, which keep ms but need a fresh MAGIC bank):
    # every generated bank runs per-query MAGIC + subsets + validation.
    assert row["status"] in ("planned", "partial") and not row["magic_lds"], (
        f"{args.run_id}: status={row['status']}, magic_lds={row['magic_lds']!r} - "
        "this row already has a MAGIC bank")
    assert row["model"] in ("gpt2", "gpt2-medium", "gpt2-large"), (
        f"model {row['model']} needs its own generator (arch rows are blocked)")
    assert row["dataset"] == "smollm2", row["dataset"]

    n = int(row["n_docs"])
    lr = float(row["lr"])
    run_path = f"{RUNS}/{args.run_id}"

    cfg = {"run_path": run_path, "steps": [{"magic": {
        "run_path": run_path,
        "overwrite": False,
        "resume": True,
        "model": row["model"],
        "model_kwargs": DROPOUT_OFF,
        "precision": "fp32",
        "use_tf32_matmuls": False,
        "seed": int(row["seed"]),
        "cleanup_ckpts": False,
        "distributed": {"nnode": 1, "nproc_per_node": args.nproc},
        "data": {"dataset": f"{DATA}/train_{n // 1000}k.hf",
                 "split": "train", "chunk_length": 0},
        "query": {"dataset": f"{DATA}/query_20.hf",
                  "split": "train", "chunk_length": 0},
        "batch_size": int(row["batch_size"]),
        "grad_accum_steps": max(1, int(row["batch_size"]) // (16 * args.nproc)),
        "num_epochs": int(row["num_epochs"]),
        "lr_schedule": {"lr": lr, "lr_scheduler_type": "polynomial",
                        "lr_start": 1e-6, "lr_end": lr / 10, "warmup_steps": 0.25},
        "optimizer": "adamw" if row["optimizer"] == "adamw" else row["optimizer"],
        "adam_beta1": 0.95, "adam_beta2": 0.975,
        "adam_eps": 1e-8, "eps_root": float(row["eps_root"]),
        "weight_decay": float(row["weight_decay"]),
        **({"max_grad_norm": float(row["max_grad_norm"])} if row["max_grad_norm"] else {}),
        "loss_reduction": "mean",
        "train_mode": False,
        # MagicConfig defaults skip_validation=True: without this override the
        # run scores MAGIC and exits WITHOUT building the retrain bank - caught
        # after the first completed bank had no ground truth.
        "skip_validation": False,
        "num_subsets": int(row["n_subsets"]),
        "subset_fraction": float(row["subset_fraction"]),
        "query_method": "none",
        "save_models": True,
        # AdamW only: persists the final second moments for TrackStar-Adam /
        # SOURCE-Adam (D8). The saver raises on muon state, so muon rows skip it.
        "save_optimizer_state": "last" if row["optimizer"] == "adamw" else "none",
        # log-spaced checkpoints (the MAGIC paper's mode): replay-compatible,
        # denser near the end (what ckptavg wants), and keeps long-step banks
        # (bs16/bs32: 2000/1000 steps) at ~15-20 checkpoints instead of the
        # sqrt mode's 30-45 — the difference between 12 banks fitting on
        # ssd-2 and not.
        "save_mode": "log",
    }}]}

    os.makedirs(run_path, exist_ok=True)
    cfg_path = f"{run_path}/experiment.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    mirror = REPO / "configs" / "experiments" / f"{args.run_id}.yaml"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    with open(mirror, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"wrote {cfg_path} (lr={lr:g}, nproc={args.nproc} — record nproc in the row notes)")
    print(f"mirrored to {mirror} — commit it with the claim")
    print("\n# Canonical invocation (pinned env; -s -P + NOUSERSITE close the shadowing traps):")
    print(f"cd /tmp && CUDA_VISIBLE_DEVICES=<gpus> MASTER_PORT=<unique> PYTHONNOUSERSITE=1 \\\n"
          f"  PYTHONPATH={BERGSON} \\\n"
          f"  /mnt/ssd-2/lucia/envs/paper/bin/python -s -P -m bergson {cfg_path}")
    print("\n# The magic step trains the base (checkpoints kept), builds the retrain")
    print("# bank, and runs per-query MAGIC + validation. Resumable at every stage.")


if __name__ == "__main__":
    main()
