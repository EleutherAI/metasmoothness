"""Mean query-loss delta under each removal condition, with CIs over queries.

No derived score. Three quantities in the same units (nats of query loss), each
the mean over queries of that query's loss change when 1% of the training data
is removed, differing only in HOW the 1% is chosen:

    random   -- the row's own leave-k-out bank subsets (the matched control)
    EK-FAC   -- top 1% by EK-FAC attribution
    MAGIC    -- top 1% by MAGIC attribution

CI is a 10k bootstrap resampling QUERIES, the unit CONTROLS pairs over, seed 0.
Writes data/filter_deltas.csv.
"""
import csv
import glob
import os
from collections import defaultdict

import numpy as np

ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments",
         "/mnt/ssd-1/lucia/paper_runs/experiments"]
OUT = "/mnt/ssd-2/lucia/metasmoothness/data/filter_deltas.csv"
rng = np.random.default_rng(0)


def ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(10_000, len(x)))
    b = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


runs = defaultdict(dict)
seen = set()
for root in ROOTS:
    for summ in sorted(glob.glob(f"{root}/*/filter_*_*/filter_summary.csv")):
        d = os.path.dirname(summ)
        run = os.path.basename(os.path.dirname(d))
        src = os.path.basename(d).split("_")[-1]
        if (run, src) in seen:
            continue
        seen.add((run, src))
        runs[run][src] = list(csv.DictReader(open(summ)))

out_rows = []
print(f"{'run':28s} {'n':>3s} {'random Δ':>22s} {'EK-FAC Δ':>22s} {'MAGIC Δ':>22s}")
for run in sorted(runs):
    src = runs[run]
    cells, rec = [], {"run": run}
    # The random control is identical for both scorers (same bank), so take it once.
    any_src = next(iter(src.values()))
    rnd = [float(r["random_mean"]) for r in any_src]
    rec["n_queries"] = len(rnd)
    for label, vals in (("random", rnd),
                        ("ekfac", [float(r["filter_change"]) for r in src["ekfac"]]
                         if "ekfac" in src else None),
                        ("magic", [float(r["filter_change"]) for r in src["magic"]]
                         if "magic" in src else None)):
        if vals is None:
            cells.append(f"{'-':>22s}")
            continue
        m, lo, hi = ci(vals)
        rec[f"{label}_mean"], rec[f"{label}_lo"], rec[f"{label}_hi"] = m, lo, hi
        cells.append(f"{m:8.5f} [{lo:7.5f},{hi:7.5f}]")
    out_rows.append(rec)
    print(f"{run[:28]:28s} {len(rnd):3d} " + " ".join(cells))

if out_rows:
    keys = sorted({k for r in out_rows for k in r})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "n_queries"] +
                           [k for k in keys if k not in ("run", "n_queries")])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {OUT}")
