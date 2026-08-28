"""What the london rows would do to the EK-FAC correlation if included.

They are NOT in experiments.csv and are held out on purpose: they use
london_query_20.hf while every other row uses query_20.hf. Each row's LDS and
delta are internally consistent, so the PAIR is meaningful, but the delta's
magnitude is not on the same scale as the rest, and Spearman across rows assumes
the pairs are comparable. This prints both numbers so the choice is explicit.
"""
import csv

import numpy as np
from scipy.stats import spearmanr

ROOT = "/mnt/ssd-2/lucia/metasmoothness"
rows = list(csv.DictReader(open(ROOT + "/experiments.csv")))
rng = np.random.default_rng(0)

base = []
for r in rows:
    lds, d = (r.get("ekfac_lds") or "").strip(), (r.get("filter_ekfac_delta") or "").strip()
    if lds and d:
        base.append((float(lds), float(d)))

LONDON = [(0.3156, 0.04886, "london16k_bs256_muon"),
          (0.3165, 0.04979, "london16k_bs256_adamw")]


def boot(pts, n=10000):
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    rho = spearmanr(x, y).statistic
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = [spearmanr(x[i], y[i]).statistic for i in idx]
    vals = [v for v in vals if np.isfinite(v)]
    return rho, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


r0, lo0, hi0 = boot(base)
print("  without london   n=%2d  rho=%+.3f [%+.3f, %+.3f]" % (len(base), r0, lo0, hi0))
withl = base + [(a, b) for a, b, _ in LONDON]
r1, lo1, hi1 = boot(withl)
print("  with london      n=%2d  rho=%+.3f [%+.3f, %+.3f]" % (len(withl), r1, lo1, hi1))
d = np.array([p[1] for p in base])
print("  delta range over the other rows: %.4f - %.4f (median %.4f)" % (d.min(), d.max(), np.median(d)))
print("  the two lowest LDS rows on the fleet, and their deltas:")
