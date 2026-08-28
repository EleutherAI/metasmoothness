"""Copy filter deltas from data/filter_deltas.csv into experiments.csv.

filter_deltas.py computes the deltas and writes data/filter_deltas.csv, but the
figures read experiments.csv, and nothing connects the two -- experiments.csv is
maintained by hand. So a finished measurement can sit on disk while its figure
still shows the "retraining" placeholder, which is exactly what happened to the
Muon 128k point: the number existed and the plot did not show it.

Run this after filter_deltas.py and before scaling_plot_mpl.py.

    python sync_filter_deltas.py [--force]

Without --force, only empty cells are filled; a value already in experiments.csv
is left alone and reported as a disagreement if it differs by more than --tol.
"""
import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path("/mnt/ssd-2/lucia/metasmoothness")
EXP, DEL = ROOT / "experiments.csv", ROOT / "data/filter_deltas.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true", help="overwrite existing values")
# Relative, not absolute. The CIs are 10k bootstraps whose draws shift with which
# rows are present, so bounds routinely differ in the 4th significant figure --
# reporting that is noise. Only a gap this large means the two disagree.
ap.add_argument("--tol", type=float, default=0.01,
                help="relative gap that counts as a real disagreement")
a = ap.parse_args()

src = {r["run"]: r for r in csv.DictReader(open(DEL))}
rows = list(csv.DictReader(open(EXP)))
cols = list(rows[0].keys())

# (experiments.csv column, filter_deltas.csv column)
PAIRS = [(f"filter_{m}_{lo}", f"{m}_{hi}")
         for m in ("random", "ekfac", "magic")
         for lo, hi in (("delta", "mean"), ("lo", "lo"), ("hi", "hi"))]

filled, clashes = [], []
for r in rows:
    s = src.get(r["run_id"])
    if not s:
        continue
    for dst_col, src_col in PAIRS:
        v = (s.get(src_col) or "").strip()
        if not v or dst_col not in cols:
            continue
        cur = (r[dst_col] or "").strip()
        if cur and not a.force:
            if abs(float(cur) - float(v)) > a.tol * max(abs(float(v)), 1e-9):
                clashes.append((r["run_id"], dst_col, cur, v))
            continue
        r[dst_col] = f"{float(v):.5f}"
        filled.append((r["run_id"], dst_col))
    if (s.get("n_queries") or "").strip() and not (r["filter_n_queries"] or "").strip():
        r["filter_n_queries"] = s["n_queries"]
        filled.append((r["run_id"], "filter_n_queries"))

if filled:
    shutil.copy(EXP, EXP.with_suffix(".csv.bak"))
    with open(EXP, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

by_run = {}
for run, col in filled:
    by_run.setdefault(run, []).append(col.replace("filter_", ""))
for run, c in sorted(by_run.items()):
    print(f"  {run}: {', '.join(c)}")
print(f"  filled {len(filled)} cell(s) across {len(by_run)} row(s)")
for run, col, cur, v in clashes:
    print(f"  DISAGREES {run} {col}: experiments.csv {cur} vs filter_deltas.csv {v}")
