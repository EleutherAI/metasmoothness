#!/usr/bin/env python3
"""How precise is the filter-delta vs LDS correlation now, and how precise can it get?

Bootstraps a CI on Spearman rho at the current n, then reports how many rows
could ever contribute, so the ceiling is visible rather than assumed.
"""
import csv
import glob
import os
import random
import sys

sys.path.insert(0, "/mnt/ssd-2/lucia/metasmoothness/scripts")
from axes import is_cut  # noqa: E402

EXP = "/mnt/ssd-2/lucia/metasmoothness/experiments.csv"
DELTAS = "/mnt/ssd-2/lucia/metasmoothness/data/filter_deltas.csv"


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


exp = {r["run_id"]: r for r in csv.DictReader(open(EXP))}
deltas = {r["run"]: r for r in csv.DictReader(open(DELTAS))}
rng = random.Random(0)

for scorer, lds_col, dcol in (("MAGIC", "magic_lds", "magic"), ("EK-FAC", "ekfac_lds", "ekfac")):
    pairs = []
    for rid, d in deltas.items():
        r = exp.get(rid)
        if not r or not r.get(lds_col) or not d.get(f"{dcol}_mean"):
            continue
        pairs.append((float(r[lds_col]), float(d[f"{dcol}_mean"])))

    n = len(pairs)
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    boots = []
    for _ in range(10000):
        samp = [pairs[rng.randrange(n)] for _ in range(n)]
        b = spearman([p[0] for p in samp], [p[1] for p in samp])
        if b == b:
            boots.append(b)
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]

    # how many rows could ever contribute: has this scorer's LDS, has a bank,
    # and is not cut
    ceiling = 0
    for rid, r in exp.items():
        if is_cut(rid) or not r.get(lds_col):
            continue
        root = next((f"/mnt/ssd-{b}/lucia/paper_runs/experiments/{rid}"
                     for b in (2, 1)
                     if os.path.isdir(f"/mnt/ssd-{b}/lucia/paper_runs/experiments/{rid}")), None)
        if root and len(glob.glob(os.path.join(root, "retrained", "subset_*"))) >= 100:
            ceiling += 1

    print(f"{scorer:<7} rho = {rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"n = {n}   ceiling = {ceiling} rows (half-width {(hi - lo) / 2:.2f})")
