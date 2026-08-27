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
ap.add_argument("--source", choices=["magic", "ekfac"], default="ekfac")
ap.add_argument("--shards", type=int, default=4)
ap.add_argument("--queries", type=int, default=20)
ap.add_argument("--mirror", default="/mnt/ssd-2/lucia/datasets_local")
args = ap.parse_args()

root = None
for base in ("/mnt/ssd-2/lucia/paper_runs/experiments",
             "/mnt/ssd-1/lucia/paper_runs/experiments"):
    if (Path(base) / args.run_id).is_dir():
        root = Path(base) / args.run_id
        break
if root is None:
    raise SystemExit("run dir not found: %s" % args.run_id)

src = root / ("filter_proponents_%s.yaml" % args.source)
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
    out_dir = root / ("filter_proponents_%s_q%d_%d" % (args.source, a, b))
    s["run_path"] = str(out_dir)
    out = root / ("filter_proponents_%s_q%d_%d.yaml" % (args.source, a, b))
    with open(out, "w") as f:
        yaml.safe_dump({"steps": [{"validate": s}], "run_path": str(out_dir)},
                       f, sort_keys=False)
    made.append((out, qds, s.get("num_subsets")))

print("wrote %d shard configs for %s (%s)" % (len(made), args.run_id, args.source))
for out, qds, ns in made:
    print("  %s  queries=%s  random_control=%s" % (out.name, qds.name, ns))
print()
print("total retrains %d (%d per shard) vs %d serial"
      % (args.shards * (per + 3), per + 3, args.queries + 3))
