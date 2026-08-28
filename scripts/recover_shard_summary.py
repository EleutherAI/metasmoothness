#!/usr/bin/env python3
"""Rebuild a SHARD's filter_summary.csv from its per-query CSV plus the bank.

recover_filter_summary.py handles the unsharded case. A shard needs one extra
thing: its filter_proponents.csv carries LOCAL query indices 0..k-1, while the
bank's validation csv is keyed by GLOBAL query index. Shard q<a>_<b> holds global
queries a..b-1, so the bank lookup for local i must use a+i.

The summary is written back with LOCAL indices, exactly as a healthy shard writes
it, so scripts/merge_filter_shards.py performs the local->global remap as usual and
nothing downstream needs to know this shard was recovered.

    python recover_shard_summary.py <run_id> --source magic --shard q7_14
"""
import argparse
import csv
import os
import statistics
import sys

ap = argparse.ArgumentParser()
ap.add_argument("run_id")
ap.add_argument("--source", default="magic")
ap.add_argument("--shard", required=True, help="e.g. q7_14")
ap.add_argument("--prefix", default="filter_proponents")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

lo, hi = (int(x) for x in a.shard.lstrip("q").split("_"))

ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments", "/mnt/ssd-1/lucia/paper_runs/experiments"]
root = next((r for r in ROOTS if os.path.isdir(os.path.join(r, a.run_id))), None)
if root is None:
    sys.exit(f"run dir not found: {a.run_id}")
run = os.path.join(root, a.run_id)
fdir = os.path.join(run, f"{a.prefix}_{a.source}_{a.shard}")
prop = os.path.join(fdir, "filter_proponents.csv")
out = os.path.join(fdir, "filter_summary.csv")
if not os.path.isfile(prop):
    sys.exit(f"no filter_proponents.csv at {prop}")
if os.path.exists(out) and not a.force:
    sys.exit(f"{out} already exists (use --force)")

bank = None
for cand in ("bank_from_filter/validation_merged.csv", "bank_from_filter/validation.csv",
             "validation_merged.csv", "validation.csv"):
    p = os.path.join(run, cand)
    if os.path.isfile(p):
        bank = p
        break
if bank is None:
    sys.exit(f"no bank validation csv under {run}")
print(f"  bank: {bank}")

rnd = {}
for r in csv.DictReader(open(bank)):
    rnd.setdefault(int(float(r["query"])), []).append(float(r["diff"]))

rows = []
for r in csv.DictReader(open(prop)):
    local = int(float(r["query"]))
    g = lo + local
    if g not in rnd:
        sys.exit(f"global query {g} absent from the bank -- refusing a partial summary")
    ctrl = rnd[g]
    fc = float(r["loss_change"])
    # rank 1 means the filtered removal hurt more than every random removal.
    rank = 1 + sum(1 for c in ctrl if c >= fc)
    rows.append({
        "query": local,
        "n_removed": r.get("n_removed", ""),
        "filter_change": fc,
        "random_mean": statistics.fmean(ctrl),
        "random_sd": statistics.pstdev(ctrl) if len(ctrl) > 1 else 0.0,
        "random_n": len(ctrl),
        "rank": rank,
    })

if len(rows) != hi - lo:
    print(f"  WARNING: {len(rows)} rows for a shard spanning {hi - lo} queries")

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"  wrote {out}: {len(rows)} rows (local queries 0..{len(rows) - 1} = global {lo}..{lo + len(rows) - 1})")
print(f"  mean filter_change {statistics.fmean(r['filter_change'] for r in rows):.6f}")
print(f"  mean random_mean   {statistics.fmean(r['random_mean'] for r in rows):.6f}")
