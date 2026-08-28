"""How many rows would it take to pin down the EK-FAC correlation?

The EK-FAC rho has read out four times as points were added and landed on both
sides of significance every time. Two explanations look identical from a single
reading: the sample is simply too small, or the relationship is weak enough that
no reachable n resolves it. They are distinguishable, and the answer decides
whether more filter deltas are worth the fleet time.

Method: resample the observed (LDS, delta) pairs with replacement to size n, and
for each n report how often the 95% bootstrap CI excludes zero. That is the
probability a study of that size would call the correlation non-zero, given the
effect actually present in the data. MAGIC is included as the contrast -- if the
procedure says MAGIC is already resolved and EK-FAC is not, the difference is the
effect size and not the method.
"""
import csv

import numpy as np
from scipy.stats import spearmanr

ROOT = "/mnt/ssd-2/lucia/metasmoothness"
rows = list(csv.DictReader(open(ROOT + "/experiments.csv")))
rng = np.random.default_rng(0)


def pairs(scorer):
    out = []
    for r in rows:
        a, b = (r.get(f"{scorer}_lds") or "").strip(), (r.get(f"filter_{scorer}_delta") or "").strip()
        if a and b:
            out.append((float(a), float(b)))
    return np.array(out)


def ci_excludes_zero(x, y, nboot=600):
    idx = rng.integers(0, len(x), size=(nboot, len(x)))
    v = [spearmanr(x[i], y[i]).statistic for i in idx]
    v = [t for t in v if np.isfinite(t)]
    if len(v) < 50:
        return False
    lo, hi = np.percentile(v, [2.5, 97.5])
    return lo > 0 or hi < 0


for scorer in ("ekfac", "magic"):
    P = pairs(scorer)
    obs = spearmanr(P[:, 0], P[:, 1]).statistic
    print(f"  {scorer.upper()}  observed rho {obs:+.3f} on n={len(P)}")
    for n in (30, 40, 60, 100, 150):
        hits = 0
        trials = 120
        for _ in range(trials):
            s = P[rng.integers(0, len(P), size=n)]
            hits += ci_excludes_zero(s[:, 0], s[:, 1])
        print(f"     n={n:4d}   CI excludes zero in {100*hits/trials:5.1f}% of studies")
