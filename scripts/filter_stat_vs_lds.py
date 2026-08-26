"""Correlate LDS against TWO filter statistics, not just the mean delta.

filter_vs_lds_perquery.py pairs LDS with the mean per-query loss change under
top-1% removal. That is one summary of what the filter measured. There is a
second already on disk, from filter_query_spearman.py: for each run, the Spearman
over queries between the attribution mass a scorer removed and the loss change it
actually caused.

The two ask different things of the same retrains:

    delta  -- HOW MUCH damage removing the scorer's top 1% does, on average
    rho_q  -- whether the scorer knew WHICH queries it would hurt

A scorer can score well on one and badly on the other, so a null result on the
delta version does not settle the question. This runs both against the same LDS
values so they can be read side by side.

Bootstrap over ROWS, not queries: rows are the independent unit here, since every
query inside a row shares a bank, a training run and a scorer.

Needs no GPU. Reads experiments.csv, data/filter_deltas.csv and
data/filter_query_spearman.csv.
"""
import csv
import os

import numpy as np
from scipy.stats import spearmanr

REPO = "/mnt/ssd-2/lucia/metasmoothness"
rng = np.random.default_rng(0)

rows = {r["run_id"]: r for r in csv.DictReader(open(REPO + "/experiments.csv"))}
deltas = {r["run"]: r for r in csv.DictReader(open(REPO + "/data/filter_deltas.csv"))}
perq = {}
p = REPO + "/data/filter_query_spearman.csv"
if os.path.isfile(p):
    for r in csv.DictReader(open(p)):
        perq[(r["run"], r["scorer"])] = r


def boot(x, y, n_boot=10000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y).statistic
    n = len(x)
    if n < 4:
        return rho, float("nan"), float("nan"), n
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(x[i])) < 2 or len(set(y[i])) < 2:
            continue
        r = spearmanr(x[i], y[i]).statistic
        if np.isfinite(r):
            out.append(r)
    if len(out) < 100:
        return rho, float("nan"), float("nan"), n
    return rho, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), n


print("%-7s %-26s %8s %20s %5s" % ("scorer", "filter statistic", "rho", "95% CI", "n"))
for sc, ldskey in (("magic", "magic_lds"), ("ekfac", "ekfac_lds")):
    # statistic 1: mean loss change under the scorer's top-1% removal, minus the
    # bank's own random-removal control at the same size
    xs, ys = [], []
    for rid, r in rows.items():
        lds = (r.get(ldskey) or "").strip()
        d = deltas.get(rid, {})
        dv = (d.get(sc + "_mean") or "").strip()
        rv = (d.get("random_mean") or "").strip()
        if lds and dv and rv:
            xs.append(float(lds))
            ys.append(float(dv) - float(rv))
    rho, lo, hi, n = boot(xs, ys)
    print("%-7s %-26s %+8.3f  [%+.3f, %+.3f] %5d"
          % (sc, "mean delta - random", rho, lo, hi, n))

    # statistic 2: per-query Spearman between removed attribution mass and the
    # loss change it caused
    xs2, ys2 = [], []
    for rid, r in rows.items():
        lds = (r.get(ldskey) or "").strip()
        q = perq.get((rid, sc), {})
        qv = (q.get("rho") or "").strip()
        if lds and qv:
            xs2.append(float(lds))
            ys2.append(float(qv))
    if len(xs2) >= 4:
        rho2, lo2, hi2, n2 = boot(xs2, ys2)
        print("%-7s %-26s %+8.3f  [%+.3f, %+.3f] %5d"
              % (sc, "per-query rho", rho2, lo2, hi2, n2))
    else:
        print("%-7s %-26s %8s %20s %5d" % (sc, "per-query rho", "-", "too few", len(xs2)))
