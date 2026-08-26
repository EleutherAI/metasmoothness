#!/usr/bin/env python3
"""Does a scorer's tail-filter power track its LDS, measured per QUERY?

The row-level version of this question tops out at thirteen points -- the number
of rows that have an LDS, still have their bank, and are not cut -- and at n=13
a Spearman interval is about +-0.45. It cannot answer the question no matter how
much GPU time goes into it.

Both quantities exist per query, though:

  per-query LDS     magic_lds() already computes one Spearman per query, over
                    the 100 subsets, and simply does not save it
  per-query delta   filter_summary.csv stores that query's loss change under
                    the scorer's top-1% removal, and the mean of the bank's
                    random removals for the same query

So each row contributes ~20 paired points instead of 1. The CI must still
bootstrap over ROWS, not queries: queries inside a row share a bank, a training
run and a corpus, so resampling queries would treat 260 correlated observations
as independent and report an interval several times too tight.

Needs no GPU and no models -- per-query LDS reads scores and validation.csv, so
it works for banks already released to the Hub.

    python filter_vs_lds_perquery.py
"""
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/mnt/ssd-2/lucia/metasmoothness/scripts")
from axes import is_cut  # noqa: E402
from magic_lds import magic_lds, merge_slices  # noqa: E402

EXP = "/mnt/ssd-2/lucia/metasmoothness/experiments.csv"
rows = {r["run_id"]: r for r in csv.DictReader(open(EXP))}


def run_root(rid):
    for b in ("/mnt/ssd-2", "/mnt/ssd-1"):
        p = f"{b}/lucia/paper_runs/experiments/{rid}"
        if os.path.isdir(p):
            return Path(p)
    return None


def per_query_lds(root):
    """One Spearman per query, over subsets. None if the bank is unscoreable."""
    # A bank built through validate(method=lds) lives in <run>/bank_from_filter,
    # not at the run root. Try both, nearest first. Without this the 2000-step
    # rows -- the only ones above 250 steps -- were dropped as unscoreable.
    last = None
    for cand in (root, root / "bank_from_filter"):
        if not cand.is_dir():
            continue
        try:
            pre = cand / "validation_merged.csv"
            csv_path = pre if pre.is_file() else merge_slices(cand)
            _, _, _, per_q, _ = magic_lds(csv_path, n_boot=1, seed=0)
            return per_q
        except Exception as e:                              # noqa: BLE001
            last = e
    print(f"  {root.name}: cannot score ({type(last).__name__ if last else 'no bank'})",
          file=sys.stderr)
    return None


def collect(src):
    """(row, query, lds_q, delta_q) for every query of every usable row."""
    out = []
    for rid in sorted(rows):
        if is_cut(rid):
            continue
        root = run_root(rid)
        if root is None:
            continue
        summ = root / f"filter_proponents_{src}" / "filter_summary.csv"
        if not summ.exists():
            continue
        lds_q = per_query_lds(root)
        if lds_q is None:
            continue
        for rec in csv.DictReader(open(summ)):
            q = int(rec["query"])
            if q >= len(lds_q):
                continue
            delta = float(rec["filter_change"]) - float(rec["random_mean"])
            out.append((rid, q, float(lds_q[q]), delta))
    return out


def cluster_bootstrap(pairs, n_boot=10000, seed=0):
    """Resample ROWS with replacement; queries travel with their row."""
    by_row = {}
    for rid, q, x, y in pairs:
        by_row.setdefault(rid, []).append((x, y))
    keys = list(by_row)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        xs, ys = [], []
        for i in pick:
            for x, y in by_row[keys[i]]:
                xs.append(x)
                ys.append(y)
        r = spearmanr(xs, ys).statistic
        if r == r:
            boots.append(r)
    return np.quantile(boots, [0.025, 0.975])


for src in ("magic", "ekfac"):
    pairs = collect(src)
    if len(pairs) < 20:
        print(f"\n{src.upper()}: only {len(pairs)} paired queries -- too few")
        continue
    xs = [p[2] for p in pairs]
    ys = [p[3] for p in pairs]
    rho = spearmanr(xs, ys).statistic
    lo, hi = cluster_bootstrap(pairs)
    n_rows = len({p[0] for p in pairs})
    print(f"\n{src.upper()}: per-query filter delta vs per-query LDS")
    print(f"  rho = {rho:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"(bootstrap over {n_rows} rows)")
    print(f"  {len(pairs)} paired queries from {n_rows} rows")
