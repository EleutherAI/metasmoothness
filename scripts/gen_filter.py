"""Generate a tail-filter validation config for one run and one score source.

The tail-filter estimator (bergson PR #430, merged) removes, per query, the
`filter_fraction` slice of documents that a scorer ranks most influential,
retrains once, and measures that query's loss change against the unablated
baseline. The matched control is the run's own leave-k-out bank, whose subsets
are random removals of the same size -- so `filter_fraction` MUST match the
bank's `subset_fraction` or bergson refuses.

One retrain per query, so cost is (queries x rows x score sources) retrains.

    python gen_filter.py <run_id> --source magic|ekfac [--fraction 0.01] [--nproc 2]
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import yaml

BERGSON = "/mnt/ssd-1/lucia/bergson-filter"
sys.path.insert(0, BERGSON)
from bergson.cli.commands import Validate  # noqa: E402

EXP = ["/mnt/ssd-2/lucia/paper_runs/experiments",
       "/mnt/ssd-1/lucia/paper_runs/experiments"]

ap = argparse.ArgumentParser()
ap.add_argument("run_id")
ap.add_argument("--source", choices=["magic", "ekfac"], required=True)
ap.add_argument("--fraction", type=float, default=None)
ap.add_argument("--nproc", type=int, default=2)
ap.add_argument("--method", default="filter-proponents")
args = ap.parse_args()

root = None
for base in EXP:
    if (Path(base) / args.run_id).is_dir():
        root = Path(base) / args.run_id
        break
if root is None:
    sys.exit(f"run dir not found: {args.run_id}")

exp = yaml.safe_load((root / "experiment.yaml").read_text())
magic = next(s["magic"] for s in exp["steps"] if "magic" in s)

scores = root / ("scores" if args.source == "magic" else "ekfac_scores/scores")
if not scores.exists():
    sys.exit(f"no {args.source} scores at {scores}")

# The random control comes from the bank's own subsets, so the filter must
# remove the same number of documents they do.
fraction = args.fraction if args.fraction is not None else magic.get("subset_fraction")
if not fraction:
    sys.exit("run has no subset_fraction; pass --fraction explicitly")

valid = {f.name for f in dataclasses.fields(Validate)}
skip = {"run_path", "num_subsets", "skip_validation", "save_models", "save_mode",
        "save_optimizer_state", "cleanup_ckpts", "resume", "overwrite",
        "double_backward_batch_size", "train_mode", "scores", "method",
        "filter_fraction", "retrained_dir"}
cfg = {k: v for k, v in magic.items() if k in valid and k not in skip}

out_dir = root / f"filter_{args.method.replace('filter-', '')}_{args.source}"
# Reuse the row's existing leave-k-out bank as the random control instead of
# retraining num_subsets fresh randoms: load_bank_losses reads the bank's 100
# retrained models directly, which drops this from ~120 retrains per row to 20
# AND makes the comparator the bank's own randoms rather than a new draw.
# bergson asserts the bank removes the same number of docs as the filter.
retrained = root / "retrained"
if not retrained.is_dir():
    sys.exit(f"no bank at {retrained}; the random control needs it")

cfg.update(
    run_path=str(out_dir),
    scores=str(scores),
    method=args.method,
    subset_fraction=float(fraction),
    retrained_dir=str(retrained),
)
cfg["distributed"] = dict(cfg.get("distributed") or {}, nproc_per_node=args.nproc, nnode=1)

dropped = sorted(set(magic) - set(cfg) - skip)
# bergson refuses to start when run_path already exists, so keep the config
# OUTSIDE the run directory and let bergson create it.
path = root / f"{out_dir.name}.yaml"
path.write_text(yaml.safe_dump({"steps": [{"validate": cfg}]}, sort_keys=False))
print(f"wrote {path}")
print(f"  source={args.source} scores={scores}")
print(f"  method={args.method} subset_fraction={fraction} nproc={args.nproc}")
print(f"  random control: bank at {retrained}")
print(f"  queries={(cfg.get('query') or {}).get('dataset','?').split('/')[-1]}")
if dropped:
    print(f"  not accepted by Validate, omitted: {dropped}")
