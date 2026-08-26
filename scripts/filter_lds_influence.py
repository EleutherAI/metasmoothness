"""Leave-one-out influence on the filter-delta / LDS correlation.

+0.792 over 24 rows is only as good as its spread of rows. If one or two points
sit far from the rest -- and this grid has some, scale0.25 and gpt2-medium both
have near-zero MAGIC LDS -- a rank correlation can be carried almost entirely by
them, and the bootstrap CI will not say so, because resampling rows keeps
redrawing the same influential points.

This drops each row in turn and reports how far rho moves. A relationship that
survives every deletion is one worth quoting; one that collapses when a single
row leaves is a statement about that row.

Needs no GPU.
"""
import csv

import numpy as np
from scipy.stats import spearmanr

REPO = "/mnt/ssd-2/lucia/metasmoothness"
rows = {r["run_id"]: r for r in csv.DictReader(open(REPO + "/experiments.csv"))}
deltas = {r["run"]: r for r in csv.DictReader(open(REPO + "/data/filter_deltas.csv"))}

for sc, key in (("magic", "magic_lds"), ("ekfac", "ekfac_lds")):
    rec = []
    for rid, r in rows.items():
        lds = (r.get(key) or "").strip()
        d = deltas.get(rid, {})
        dv = (d.get(sc + "_mean") or "").strip()
        rv = (d.get("random_mean") or "").strip()
        if lds and dv and rv:
            rec.append((rid, float(lds), float(dv) - float(rv)))
    if len(rec) < 6:
        continue
    ids = [a for a, _, _ in rec]
    x = np.array([b for _, b, _ in rec])
    y = np.array([c for _, _, c in rec])
    full = spearmanr(x, y).statistic

    infl = []
    for i in range(len(rec)):
        m = np.ones(len(rec), bool)
        m[i] = False
        r = spearmanr(x[m], y[m]).statistic
        infl.append((r - full, ids[i], r))
    infl.sort(key=lambda t: t[0])

    print("\n%s: rho = %+.3f over %d rows" % (sc.upper(), full, len(rec)))
    print("  most influential rows (drop -> new rho):")
    for d, rid, r in infl[:3]:
        print("    %-38s %+.3f  ->  %+.3f" % (rid, d, r))
    for d, rid, r in infl[-3:]:
        print("    %-38s %+.3f  ->  %+.3f" % (rid, d, r))
    lo = min(t[2] for t in infl)
    hi = max(t[2] for t in infl)
    print("  leave-one-out range: %+.3f to %+.3f (full %+.3f)" % (lo, hi, full))
    # how many rows must be dropped, worst case, to push rho below 0.3
    order = sorted(range(len(rec)), key=lambda i: -y[i])
    print("  rows with |LDS| < 0.2 (the low-signal end): %s"
          % ", ".join(ids[i] for i in range(len(rec)) if abs(x[i]) < 0.2) or "none")
