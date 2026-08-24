"""How does the LDS confidence interval shrink with the number of queries?

MAGIC LDS is the mean of per-query Spearman correlations, so the query count
drives the CI directly. If the interval is already inside the D6 threshold at
fewer than 20 queries, later rows can be scored more cheaply -- MAGIC costs one
reverse pass PER QUERY, so this is the single biggest cost lever in the grid.

For each row this resamples QUERIES (the unit CONTROLS pairs over) at
n = 5, 10, 15, 20 and reports the bootstrap CI half-width of the mean.
"""
import csv
import glob
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

D6 = 0.06  # DECISIONS D6 escalation threshold on the half-width
NS = [5, 10, 15, 20]
rng = np.random.default_rng(0)


def per_query_rho(run_dir):
    merged = sorted(glob.glob(f"{run_dir}/**/validation_merged.csv", recursive=True))
    path = merged[0] if merged else f"{run_dir}/validation.csv"
    if not os.path.exists(path):
        return None
    diffs, sums = defaultdict(list), defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            q = int(r["query"])
            diffs[q].append(float(r["diff"]))
            sums[q].append(float(r["score_sum"]))
    if not diffs:
        return None
    n_subsets = len(diffs[sorted(diffs)[0]])
    return (np.array([spearmanr(diffs[q], sums[q]).statistic for q in sorted(diffs)]),
            n_subsets)


rows = []
for base in ("/mnt/ssd-2/lucia/paper_runs/experiments",
             "/mnt/ssd-1/lucia/paper_runs/experiments"):
    for d in sorted(glob.glob(f"{base}/*")):
        rid = os.path.basename(d)
        if any(r[0] == rid for r in rows):
            continue
        got = per_query_rho(d)
        if got is None:
            continue
        rho, n_subsets = got
        if len(rho) < 20:
            continue
        # A partial bank still yields per-query rho, so the subset count MUST be
        # shown -- otherwise an unfinished row reads as a finished measurement.
        rows.append((rid, rho, n_subsets))

print(f"{'run':32s} {'subs':>5s} {'mean':>7s} " + " ".join(f"{'n=' + str(n):>9s}" for n in NS))
print(f"{'':32s} {'':>5s} {'':>7s} " + " ".join(f"{'halfwidth':>9s}" for _ in NS))
enough = defaultdict(int)
for rid, rho, n_subsets in rows:
    cells = []
    for n in NS:
        idx = rng.integers(0, len(rho), size=(5000, n))
        boot = rho[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        hw = (hi - lo) / 2
        cells.append(f"{hw:9.4f}")
        if hw < D6:
            enough[n] += 1
    flag = "" if n_subsets >= 100 else "  <- PARTIAL BANK"
    print(f"{rid[:32]:32s} {n_subsets:5d} {rho.mean():7.4f} "
          + " ".join(cells) + flag)

print(f"\nrows whose half-width is under the D6 threshold of {D6}:")
for n in NS:
    print(f"  n={n:2d}: {enough[n]}/{len(rows)}")
