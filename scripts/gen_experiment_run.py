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
BERGSON = "/mnt/ssd-1/lucia/bergson-damping"
DATA = f"{BERGSON}/runs/ekfac_vs_n/datasets"
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
    assert row["status"] == "planned", f"{args.run_id} status={row['status']}"
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
        "num_subsets": int(row["n_subsets"]),
        "subset_fraction": float(row["subset_fraction"]),
        "query_method": "none",
        "save_models": True,
        "save_optimizer_state": "last",
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
    print("\n# Run from OUTSIDE any bergson checkout (cwd shadows PYTHONPATH otherwise):")
    print(f"cd /tmp && PYTHONPATH={BERGSON} python -P -m bergson {cfg_path}")
    print("\n# The magic step trains the base (checkpoints kept), builds the retrain")
    print("# bank, and runs per-query MAGIC + validation. Resumable at every stage.")


if __name__ == "__main__":
    main()
