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

# `python -P` keeps a bergson checkout's cwd off sys.path, but it also drops
# this script's own directory -- put just that one entry back for the import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_config  # noqa: E402

EXP = ["/mnt/ssd-2/lucia/paper_runs/experiments",
       "/mnt/ssd-1/lucia/paper_runs/experiments"]

ap = argparse.ArgumentParser()
ap.add_argument("run_id")
ap.add_argument("--source", choices=["magic", "ekfac"], required=True)
ap.add_argument("--fraction", type=float, default=None)
ap.add_argument("--random-n", type=int, default=3,
    help="random retrains for the control; 3 unless the delta collapses")
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

exp = run_config.load(root)
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
# save_mode is forced to interval with a stride nothing reaches, so each query's
# retrain keeps only its final state. The previous choice was bergson's "sqrt"
# default, kept so the trajectory would stay replayable by MAGIC -- but that is
# 27-118 GiB per run, one trajectory per query, and /mnt/ssd-2 is a 25 TiB quota
# shared with other tenants sitting at 24.7 TiB. It hit the limit twice; the
# second time thirteen runs stalled mid-flight holding 714 GiB and produced
# nothing. A filter run never replays MAGIC: it retrains once per query and the
# deliverable is that query's final loss. Same pattern, and same reason, as
# gen_tuning_run.py. (bergson-filter predates save_mode "final", PR #441, which
# would say this more directly.) Drop these two keys to restore a replayable
# trajectory for a row that needs one.
skip = {"run_path", "num_subsets", "skip_validation", "save_models", "save_mode",
        "save_optimizer_state", "cleanup_ckpts", "resume", "overwrite",
        "double_backward_batch_size", "train_mode", "scores", "method",
        "filter_fraction", "retrained_dir"}
cfg = {k: v for k, v in magic.items() if k in valid and k not in skip}
# Random-removal control size: THREE (Lucia, 2026-08-25). Measured from every
# completed run, random_sd is ~0.001-0.002 nats against a typical filter delta
# of 0.05-0.09, so at k=3 the control contributes SE = sd/sqrt(3) ~ 0.0007,
# around 1% of the effect. Twenty was costing 17 extra retrains per row for
# nothing, which at 256k is about 39 GPU-hours thrown away per point.
if args.random_n:
    cfg["num_subsets"] = args.random_n
cfg["save_mode"] = "interval"
cfg["save_interval"] = 10**9

out_dir = root / f"filter_{args.method.replace('filter-', '')}_{args.source}"
# Reuse the row's existing leave-k-out bank as the random control instead of
# retraining num_subsets fresh randoms: load_bank_losses reads the bank's 100
# retrained models directly, which drops this from ~120 retrains per row to 20
# AND makes the comparator the bank's own randoms rather than a new draw.
# bergson asserts the bank removes the same number of docs as the filter.
# retrained_dir is the RUN directory, not the retrained/ subdir: bergson's
# load_and_validate_subsets_match asserts (d / "retrained" / "base").exists(),
# so it appends "retrained" itself. Passing the subdir makes it look for
# retrained/retrained/base and fail AFTER all 20 per-query retrains are done.
retrained = root / "retrained"
if not (retrained / "base").is_dir():
    sys.exit(f"no bank base at {retrained / 'base'}; the random control needs it")

cfg.update(
    run_path=str(out_dir),
    scores=str(scores),
    method=args.method,
    subset_fraction=float(fraction),
    retrained_dir=str(root),
)
cfg["distributed"] = dict(cfg.get("distributed") or {}, nproc_per_node=args.nproc, nnode=1)

# The dataset directory on ssd-1 has been unlistable fleet-wide since a copy got
# stuck in uninterruptible sleep inside it, and on the node holding that stuck
# request every access to it hangs unkillably -- which is what left six A100s on
# marisa-0 idle: training there was fine, but each filter job needed query_20.hf
# and blocked. Rewrite any dataset that has been mirrored to ssd-2. Only the
# mirror is stat-ed; the ssd-1 path is left alone rather than probed, since
# probing it is the thing that hangs.
MIRROR = Path("/mnt/ssd-2/lucia/datasets_local")


def _mirrored(entry):
    """Repoint one {'dataset': ...} mapping at the mirror, if the mirror has it."""
    if not isinstance(entry, dict) or not entry.get("dataset"):
        return
    local = MIRROR / Path(entry["dataset"]).name
    if local.is_dir():
        entry["dataset"] = str(local)


for _key in ("data", "query"):
    _mirrored(cfg.get(_key))

dropped = sorted(set(magic) - set(cfg) - skip)
# bergson refuses to start when run_path already exists, so keep the config
# OUTSIDE the run directory and let bergson create it.
path = root / f"{out_dir.name}.yaml"
path.write_text(yaml.safe_dump({"steps": [{"validate": cfg}]}, sort_keys=False))
print(f"wrote {path}")
print(f"  source={args.source} scores={scores}")
print(f"  method={args.method} subset_fraction={fraction} nproc={args.nproc}")
print(f"  random control: bank at {retrained} (passed as run dir {root})")
print(f"  queries={(cfg.get('query') or {}).get('dataset','?').split('/')[-1]}")
if dropped:
    print(f"  not accepted by Validate, omitted: {dropped}")
