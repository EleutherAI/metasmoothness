#!/usr/bin/env python3
"""Split a bank's remaining retrains into slice configs for parallel processes.

    python scripts/slice_bank.py <run_id> --start 72 --slices 2

Writes ``slice_<a>_<b>.yaml`` next to the run's ``experiment.yaml`` (identical
except ``subset_start``/``subset_stop``) and prints one canonical launch
command per slice. Each slice process retrains the base (cheap, bit-exact at
the same nproc), resumes MAGIC scoring from ``per_query/`` (all queries must
already be scored), retrains only its subset range and writes
``validation_<a>_<b>.csv``; ``scripts/magic_lds.py <run_dir>`` merges the
slices. See NODES.md "Sharding a bank's retrains".
"""
import argparse
from pathlib import Path

import yaml

import sys
# `python -P` keeps a bergson checkout's cwd off sys.path, but it also drops
# this script's own directory -- put just that one entry back for the import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_config  # noqa: E402

EXPERIMENTS = Path("/mnt/ssd-2/lucia/paper_runs/experiments")
BERGSON = "/mnt/ssd-1/lucia/bergson-main-paper-429"
PYTHON = "/home/lucia/envs/paper/bin/python"


def magic_block(cfg: dict) -> dict:
    """Return the magic step's config dict (gen_experiment_run.py's layout)."""
    return next(step["magic"] for step in cfg["steps"] if "magic" in step)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--start", type=int, default=0, help="first subset not yet retrained")
    ap.add_argument("--slices", type=int, default=2)
    ap.add_argument("--port", type=int, default=29900, help="first MASTER_PORT; +2 per slice")
    args = ap.parse_args()

    run = EXPERIMENTS / args.run_id
    cfg = run_config.load(run)
    magic = magic_block(cfg)
    n = magic["num_subsets"]
    nproc = magic["distributed"]["nproc_per_node"]
    queries = len(list((run / "per_query").glob("q*.pt")))
    print(f"# {args.run_id}: {queries} queries scored; slicing subsets {args.start}..{n}")
    print("# Launch ONE AT A TIME: wait for 'Validating' in the previous slice's log before")
    print("# starting the next - every slice resumes from and re-saves the last checkpoint,")
    print("# and two doing so at once corrupt it (PytorchStreamReader ... miniz error).")

    edges = [args.start + round(i * (n - args.start) / args.slices) for i in range(args.slices + 1)]
    for k, (a, b) in enumerate(zip(edges, edges[1:])):
        sliced = run_config.load(run)
        block = magic_block(sliced)
        block["subset_start"], block["subset_stop"] = a, b
        path = run / f"slice_{a}_{b}.yaml"
        path.write_text(yaml.safe_dump(sliced, sort_keys=False))
        print(
            f"cd /tmp && CUDA_VISIBLE_DEVICES=<{nproc} gpus> MASTER_PORT={args.port + 2 * k} "
            f"PYTHONNOUSERSITE=1 PYTHONPATH={BERGSON} {PYTHON} -s -P -m bergson {path} "
            f"> {run}/slice_{a}_{b}.log 2>&1"
        )


if __name__ == "__main__":
    main()
