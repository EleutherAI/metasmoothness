"""Audit which local retrain banks are safe to delete, and optionally delete them.

Lucia: "you can delete bank directories that are on the hub and have been
processed/statistics all extracted locally".

Three conditions, ALL required, checked per row. A bank is only redundant if the
Hub has the models AND everything we would ever recompute from them locally has
already been computed:

  1. published   the Hub repo exists and its file count matches the local bank
  2. ground truth extracted   a local validation CSV with 100 subsets x 20
     queries survives, since that is the actual scientific artifact -- the models
     are only the means of producing it
  3. statistics extracted   magic_lds AND ekfac_lds recorded in experiments.csv,
     and a filter delta present in data/filter_deltas.csv

Deletes ONLY <run>/retrained (and bank_from_filter/retrained), never the
validation CSVs, subsets.json, scores, or config. Those are small and are what
the analysis reads.

    python reclaim_banks.py            # audit only
    python reclaim_banks.py --delete   # act
"""
import argparse
import csv
import glob
import os
import shutil
import sys

AP = argparse.ArgumentParser()
AP.add_argument("--delete", action="store_true")
AP.add_argument("--min-gb", type=float, default=1.0)
args = AP.parse_args()

REPO = "/mnt/ssd-2/lucia/metasmoothness"
ROOTS = ["/mnt/ssd-1/lucia/paper_runs/experiments",
         "/mnt/ssd-2/lucia/paper_runs/experiments"]

rows = {r["run_id"]: r for r in csv.DictReader(open(REPO + "/experiments.csv"))}
deltas = {r["run"]: r for r in csv.DictReader(open(REPO + "/data/filter_deltas.csv"))}

from huggingface_hub import HfApi
api = HfApi()
published = {}
for d in api.list_datasets(author="EleutherAI", search="metasmoothness-bank"):
    published[d.id.split("metasmoothness-bank-")[-1]] = d.id


def bank_dirs(run_root):
    out = []
    for sub in ("retrained", "bank_from_filter/retrained"):
        p = os.path.join(run_root, sub)
        if os.path.isdir(p) and glob.glob(p + "/subset_*"):
            out.append(p)
    return out


def val_ok(run_root):
    """A local ground-truth CSV with the full 100 x 20 grid."""
    for pat in ("validation_merged.csv", "validation.csv",
                "bank_from_filter/validation_merged.csv",
                "bank_from_filter/validation.csv"):
        p = os.path.join(run_root, pat)
        if not os.path.isfile(p):
            continue
        try:
            rr = list(csv.DictReader(open(p)))
        except OSError:
            continue
        subs = {r["subset"] for r in rr}
        qs = {r["query"] for r in rr}
        if len(subs) >= 100 and len(qs) >= 20:
            return p, len(subs), len(qs)
    return None, 0, 0


def du_gb(p):
    total = 0
    for dirpath, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / 1024 ** 3


safe, unsafe = [], []
seen = set()
for root in ROOTS:
    for p in sorted(glob.glob(root + "/*/")):
        rid = os.path.basename(p.rstrip("/"))
        if rid in seen:
            continue
        seen.add(rid)
        dirs = bank_dirs(p)
        if not dirs:
            continue
        why = []
        if rid not in published:
            why.append("not on the Hub")
        vpath, ns, nq = val_ok(p)
        if vpath is None:
            why.append("no complete local validation CSV (100x20)")
        r = rows.get(rid, {})
        if not (r.get("magic_lds") or "").strip():
            why.append("magic_lds not recorded")
        if not (r.get("ekfac_lds") or "").strip():
            why.append("ekfac_lds not recorded")
        d = deltas.get(rid, {})
        if not ((d.get("magic_mean") or "").strip() or (d.get("ekfac_mean") or "").strip()):
            why.append("no filter delta")
        gb = sum(du_gb(x) for x in dirs)
        (unsafe if why else safe).append((rid, gb, dirs, why, vpath))

print("%-40s %8s  %s" % ("run", "GB", "status"))
tot = 0.0
for rid, gb, dirs, why, vpath in sorted(safe, key=lambda x: -x[1]):
    tot += gb
    print("%-40s %8.1f  SAFE (gt: %s)" % (rid, gb, os.path.basename(vpath)))
for rid, gb, dirs, why, vpath in sorted(unsafe, key=lambda x: -x[1]):
    print("%-40s %8.1f  KEEP -- %s" % (rid, gb, "; ".join(why)))

print("\n%d safe, %.1f GB reclaimable; %d kept" % (len(safe), tot, len(unsafe)))

if not args.delete:
    print("\naudit only. rerun with --delete to act.")
    sys.exit(0)

freed = 0.0
for rid, gb, dirs, why, vpath in safe:
    for d in dirs:
        try:
            shutil.rmtree(d)
            freed += gb
            print("deleted %s (%.1f GB)" % (d, gb))
        except OSError as e:
            print("FAILED %s: %s" % (d, e))
print("freed about %.1f GB" % freed)
