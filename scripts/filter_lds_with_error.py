"""Correlate filter delta against LDS, propagating BOTH measurement errors.

filter_stat_vs_lds.py bootstraps over rows and treats each row's delta and LDS as
exact. They are not. Both carry their own bootstrap CIs, and the delta's is wide:
the median CI spans 0.49 x the delta itself, and on one row it spans 9.57 x.

Measurement error in x and y attenuates a correlation -- the observed rho is
biased toward zero relative to the true one, and an interval that ignores it is
too narrow. So a headline of +0.792 [+0.522, +0.945] is answering "how much would
this move if I had different ROWS", not "how much would it move if I measured the
same rows again".

This resamples rows AND redraws each row's delta and LDS from their own intervals,
treating the reported CI as +-1.96 sigma. Reported alongside the exact-value
version so the cost of the noise is visible.

Needs no GPU.
"""
import csv
import os

import numpy as np
from scipy.stats import spearmanr

REPO = "/mnt/ssd-2/lucia/metasmoothness"
rng = np.random.default_rng(0)

rows = {r["run_id"]: r for r in csv.DictReader(open(REPO + "/experiments.csv"))}
deltas = {r["run"]: r for r in csv.DictReader(open(REPO + "/data/filter_deltas.csv"))}


def sigma(lo, hi):
    """CI half-width -> sigma, treating the interval as +-1.96 sigma."""
    try:
        return max(0.0, (float(hi) - float(lo)) / 3.92)
    except (TypeError, ValueError):
        return 0.0


def collect(sc, ldskey):
    """(lds, lds_sd, delta, delta_sd) per row, delta net of the random control."""
    out = []
    for rid, r in rows.items():
        lds = (r.get(ldskey) or "").strip()
        d = deltas.get(rid, {})
        dv = (d.get(sc + "_mean") or "").strip()
        rv = (d.get("random_mean") or "").strip()
        if not (lds and dv and rv):
            continue
        lds_sd = sigma(r.get(ldskey.replace("_lds", "_ci_lo")),
                       r.get(ldskey.replace("_lds", "_ci_hi")))
        # the delta and the random control each carry error; they are measured on
        # the same retrains so this is a conservative independent-error treatment
        d_sd = sigma(d.get(sc + "_lo"), d.get(sc + "_hi"))
        r_sd = sigma(d.get("random_lo"), d.get("random_hi"))
        out.append((float(lds), lds_sd, float(dv) - float(rv),
                    (d_sd ** 2 + r_sd ** 2) ** 0.5))
    return out


print("%-7s %-34s %8s %20s %4s" % ("scorer", "treatment", "rho", "95% CI", "n"))
for sc, key in (("magic", "magic_lds"), ("ekfac", "ekfac_lds")):
    rec = collect(sc, key)
    if len(rec) < 5:
        print("%-7s %-34s %8s" % (sc, "too few rows", "-"))
        continue
    x0 = np.array([a for a, _, _, _ in rec])
    xs = np.array([b for _, b, _, _ in rec])
    y0 = np.array([c for _, _, c, _ in rec])
    ys = np.array([d for _, _, _, d in rec])
    n = len(rec)

    point = spearmanr(x0, y0).statistic

    def run(with_noise):
        out = []
        for _ in range(10000):
            i = rng.integers(0, n, n)
            xx, yy = x0[i], y0[i]
            if with_noise:
                xx = xx + rng.normal(0, 1, n) * xs[i]
                yy = yy + rng.normal(0, 1, n) * ys[i]
            if len(set(xx)) < 2 or len(set(yy)) < 2:
                continue
            r = spearmanr(xx, yy).statistic
            if np.isfinite(r):
                out.append(r)
        return np.percentile(out, 2.5), np.percentile(out, 97.5)

    lo1, hi1 = run(False)
    lo2, hi2 = run(True)
    print("%-7s %-34s %+8.3f  [%+.3f, %+.3f] %4d"
          % (sc, "rows only (exact values)", point, lo1, hi1, n))
    print("%-7s %-34s %+8.3f  [%+.3f, %+.3f] %4d"
          % (sc, "rows + measurement error", point, lo2, hi2, n))
