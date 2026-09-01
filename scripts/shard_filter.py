#!/usr/bin/env python3
"""Split a row's filter_proponents_<source>.yaml into per-query shards.

    python scripts/shard_filter.py <run_id> --source ekfac --shards 4

bergson has no query_start/query_stop: `subset_start`/`subset_stop` exist but are
only read by validate_scores (the bank path), not by tail_filter_retrain. The
query set is an ordinary dataset path though, so slicing THAT shards the work
with no code change -- Lucia's suggestion, and it is the whole trick.

Each shard gets its own run_path so the CSVs cannot collide, and its own 3 random
control retrains: num_subsets=0 is rejected unless a bank is supplied, so the
control cannot be shared across shards. That is a real difference from a serial
run, not just bookkeeping -- serially the SAME 3 random models are evaluated
against all 20 queries, whereas sharded each group of 5 queries gets its own 3.
Per-query deltas stay valid; only the shared variance between queries changes.

Cost: shards*(queries/shards + 3) retrains total, but wall-clock is one shard.
For 20 queries in 4 shards: 32 retrains total, 8 per shard vs 23 serial.
"""
import argparse, copy, os
from pathlib import Path
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("run_id")
ap.add_argument("--source", choices=["magic", "ekfac", "bm25"], default="ekfac")
ap.add_argument("--prefix", default="filter_proponents",
                help="filter config/run prefix; use filter_top40 for fixed-40 runs")
ap.add_argument("--score-slice-prefix", default="",
                help="score slice directory prefix. Defaults to scores for magic/ekfac "
                     "and bm25_scores for bm25.")
ap.add_argument("--shards", type=int, default=4)
ap.add_argument("--queries", type=int, default=20)
ap.add_argument("--mirror", default="/mnt/ssd-2/lucia/datasets_local")
ap.add_argument("--controls", choices=["shared", "per-shard"], default="shared",
                help="shared (default): every shard READS one control bank and "
                     "retrains none, so the row costs 3 control retrains total. "
                     "per-shard: each shard retrains its own 3, costing 3*shards.")
ap.add_argument("--bank", default="",
                help="control bank for --controls shared (default "
                     "<run>/bank_from_filter). Build it with gen_bank.py.")
ap.add_argument("--force", action="store_true",
                help="proceed even when the plan exceeds queries+controls retrains")
args = ap.parse_args()

root = None
for base in ("/mnt/ssd-2/lucia/paper_runs/experiments",
             "/mnt/ssd-1/lucia/paper_runs/experiments"):
    if (Path(base) / args.run_id).is_dir():
        root = Path(base) / args.run_id
        break
if root is None:
    raise SystemExit("run dir not found: %s" % args.run_id)

src = root / ("%s_%s.yaml" % (args.prefix, args.source))
if not src.is_file():
    raise SystemExit("generate the unsharded config first: %s" % src)

doc = yaml.safe_load(open(src))
step = doc["steps"][0]["validate"]
if args.queries % args.shards:
    raise SystemExit("%d queries does not divide into %d shards" % (args.queries, args.shards))
per = args.queries // args.shards

made = []
for i in range(args.shards):
    a, b = i * per, (i + 1) * per
    qds = Path(args.mirror) / ("query_%d_q%d_%d.hf" % (args.queries, a, b))
    if not qds.is_dir():
        raise SystemExit("missing query slice %s -- create it first" % qds)
    s = copy.deepcopy(step)
    s["query"] = dict(s.get("query", {}))
    s["query"]["dataset"] = str(qds)
    out_dir = root / ("%s_%s_q%d_%d" % (args.prefix, args.source, a, b))
    s["run_path"] = str(out_dir)

    # Point the shard at its OWN score slice. This used to be left at the full
    # 20-column file, which dies inside validate_scores AFTER the shard has
    # trained -- so each failure costs a whole retrain.
    score_prefix = args.score_slice_prefix or ("bm25_scores" if args.source == "bm25" else "scores")
    sl = root / ("%s_q%d_%d" % (score_prefix, a, b))
    if (sl / "info.json").is_file():
        s["scores"] = str(sl)
    elif "scores" in s:
        raise SystemExit(
            "missing score slice %s -- run scripts/shard_scores.py first, or this "
            "shard trains a full retrain and then dies with 'scores has N query "
            "columns but the query dataset has %d documents'" % (sl, b - a))

    # Controls: read ONE shared bank instead of retraining 3 per shard.
    if args.controls == "shared":
        bank = Path(args.bank or str(root / "bank_from_filter"))
        if bank.is_dir():
            s["retrained_dir"] = str(bank)
        else:
            s.pop("retrained_dir", None)

    # --- shard config assembled, now write it ---
    out = root / ("%s_%s_q%d_%d.yaml" % (args.prefix, args.source, a, b))
    with open(out, "w") as f:
        yaml.safe_dump({"steps": [{"validate": s}], "run_path": str(out_dir)},
                       f, sort_keys=False)
    made.append((out, qds, s.get("num_subsets")))

print("wrote %d shard configs for %s (%s)" % (len(made), args.run_id, args.source))
for out, qds, ns in made:
    print("  %s  queries=%s  random_control=%s" % (out.name, qds.name, ns))
print()

# --- cost guard -------------------------------------------------------------
# The row needs `queries` retrains plus `n_ctrl` controls. Sharding must not
# multiply the controls; if the plan does, stop rather than warn.
n_ctrl = int(step.get("num_subsets", 3) or 0)
# Each shard also retrains an unfiltered BASELINE before its query loop -- the
# same-conditions reference. Like the controls it is identical across shards, and
# like the controls a bank supplies it (load_bank_losses returns bank_base), so
# --controls shared removes both. Measured on the 128k row: 6 retrains per shard
# (1 baseline + 2 queries + 3 controls) where the row needs 24 in total.
minimum = 1 + args.queries + n_ctrl
planned = (args.shards * (1 + args.queries // args.shards + n_ctrl)
           if args.controls == "per-shard" else args.queries)
bank = args.bank or str(root / "bank_from_filter")

print("retrain budget: %d planned, %d minimum (1 baseline + %d queries + %d controls)"
      % (planned, minimum, args.queries, n_ctrl))

if args.controls == "shared":
    print("controls: shared, read from %s" % bank)
    if not Path(bank).is_dir():
        print("WARNING: %s does not exist yet. Build it before the shards finish:"
              % bank)
        print("  python scripts/gen_bank.py %s --num-subsets %d --subset-fraction %s"
              % (args.run_id, n_ctrl, step.get("subset_fraction", "<frac>")))
elif planned > minimum and not args.force:
    raise SystemExit(
        "REFUSING: --controls per-shard makes this %d retrains against a minimum "
        "of %d.\n"
        "The baseline and the 3 controls are models for the WHOLE row -- scoring "
        "them against every query is forward passes, not training. Per-shard, each "
        "shard repeats BOTH.\n"
        "Build the bank once and let all shards read it:\n"
        "  python scripts/gen_bank.py %s --num-subsets %d --subset-fraction %s\n"
        "  python scripts/shard_filter.py %s --controls shared\n"
        "Use --force only if you truly want %d independent control sets."
        % (planned, minimum, args.run_id, n_ctrl,
           step.get("subset_fraction", "<frac>"), args.run_id, args.shards))
