"""Paired optimizer contrast: per-query differences of two rows' Spearman arrays.

CONTROLS.md pairs optimizer contrasts over queries, and the subset draws are
seeded identically across optimizers so the arms are comparable query by query.
That paired statistic is far tighter than differencing two independent row
intervals -- the anchor's +0.0863 carries [+0.0670, +0.1052], half-width 0.019,
against single-row half-widths of 0.01-0.05.

Bootstrap resamples QUERIES (the paired unit), not subsets, 10k times, seed 0.
"""

import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


def per_query_rho(run_dir):
    # A sharded bank's rows live in validation_merged.csv, written by
    # magic_lds.py after it merges the slices and asserts each subset appears
    # exactly once. Reading validation.csv there would silently use only the
    # pre-shard prefix -- 22 subsets instead of 100 for muon bs32.
    import os
    path = f"{run_dir}/validation_merged.csv"
    if not os.path.isfile(path):
        path = f"{run_dir}/validation.csv"
    diffs, scores = defaultdict(list), defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            q = int(r["query"])
            diffs[q].append(float(r["diff"]))
            scores[q].append(float(r["score_sum"]))
    return np.array([spearmanr(diffs[q], scores[q]).statistic for q in sorted(diffs)])


a_dir, b_dir, a_name, b_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
a, b = per_query_rho(a_dir), per_query_rho(b_dir)
assert len(a) == len(b), f"query counts differ: {len(a)} vs {len(b)}"

d = a - b
rng = np.random.default_rng(0)
idx = rng.integers(0, len(d), size=(10_000, len(d)))
boot = d[idx].mean(axis=1)
lo, hi = np.percentile(boot, [2.5, 97.5])

print(f"{a_name}  mean per-query rho = {a.mean():.4f}")
print(f"{b_name}  mean per-query rho = {b.mean():.4f}")
print(f"paired {a_name} - {b_name} = {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
      f"half-width {(hi-lo)/2:.4f}")
print(f"query wins for {a_name}: {(d > 0).sum()}/{len(d)}")
