#!/usr/bin/env python3
"""Rebuild a filter_summary.csv from a crashed filter run plus its bank.

    python scripts/recover_filter_summary.py <run_id> [--source ekfac]

A tail-filter run writes filter_proponents.csv incrementally (one row per query)
but filter_summary.csv only at the very end, after it loads the random control.
If it dies in between -- e.g. because it looked for the bank at <run>/retrained
while the bank actually lives at <run>/bank_from_filter/retrained -- every query
retrain is on disk but nothing downstream can see it.

filter_proponents.csv's `loss_change` is bit-identical to filter_summary.csv's
`filter_change` (verified across 20 queries on a run that produced both), so the
summary can be reconstructed exactly, taking the random control from the bank's
validation_merged.csv instead of from fresh retrains.
"""
import argparse, csv, os, statistics, sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("run_id")
ap.add_argument("--source", default="ekfac")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments", "/mnt/ssd-1/lucia/paper_runs/experiments"]
root = next((r for r in ROOTS if os.path.isdir(os.path.join(r, a.run_id))), None)
if root is None:
    sys.exit("run dir not found: %s" % a.run_id)
run = os.path.join(root, a.run_id)
fdir = os.path.join(run, "filter_proponents_%s" % a.source)
prop = os.path.join(fdir, "filter_proponents.csv")
out = os.path.join(fdir, "filter_summary.csv")
if not os.path.isfile(prop):
    sys.exit("no filter_proponents.csv at %s" % prop)
if os.path.exists(out) and not a.force:
    sys.exit("%s already exists (use --force)" % out)

bank = None
# Two bank layouts exist. Rows built through gen_bank keep theirs under
# bank_from_filter/; rows whose magic step wrote the bank in place keep it at the
# run root (the london and gpt2medium rows do). Checking only the first made a
# fully recoverable run look like it had no ground truth at all.
for cand in ("bank_from_filter/validation_merged.csv", "bank_from_filter/validation.csv",
             "validation_merged.csv", "validation.csv"):
    p = os.path.join(run, cand)
    if os.path.isfile(p):
        bank = p
        break
if bank is None:
    sys.exit("no bank validation csv under %s" % run)

rnd = defaultdict(list)
for r in csv.DictReader(open(bank)):
    rnd[int(float(r["query"]))].append(float(r["diff"]))

rows = []
for r in csv.DictReader(open(prop)):
    q = int(float(r["query"]))
    if q not in rnd:
        sys.exit("query %d absent from the bank -- refusing a partial summary" % q)
    ctrl = rnd[q]
    fc = float(r["loss_change"])
    rows.append({
        "query": q,
        "n_removed": r.get("n_removed", ""),
        "filter_change": fc,
        "random_mean": statistics.fmean(ctrl),
        "random_sd": statistics.pstdev(ctrl) if len(ctrl) > 1 else 0.0,
        "random_n": len(ctrl),
        "rank": 1 + sum(1 for c in ctrl if c >= fc),
    })
rows.sort(key=lambda r: r["query"])

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

d = [r["filter_change"] - r["random_mean"] for r in rows]
print("  wrote %s" % out)
print("  queries=%d  control subsets/query=%d" % (len(rows), rows[0]["random_n"]))
print("  mean delta = %.5f   rank-1 queries: %d/%d"
      % (statistics.fmean(d), sum(1 for r in rows if r["rank"] == 1), len(rows)))
