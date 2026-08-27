#!/usr/bin/env python3
"""LDS vs filter-delta correlation, split by optimiser step count.

    python scripts/corr_by_steps.py

The standing question is whether the LDS<->filter-delta relationship holds at
higher step counts, not just on the 125-step rows that dominate the grid. This
splits the same rows by `steps` and reports each scorer separately, with the
random-control delta subtracted (a row's filter delta is only meaningful against
its own control).
"""
import csv, os, random, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "experiments.csv"))))


def f(r, k):
    v = (r.get(k) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def boot_ci(xs, ys, n=10000, seed=0):
    rnd = random.Random(seed)
    idx = range(len(xs))
    vals = []
    for _ in range(n):
        s = [rnd.choice(idx) for _ in idx]
        if len({xs[i] for i in s}) < 3:
            continue
        vals.append(spearman([xs[i] for i in s], [ys[i] for i in s]))
    vals.sort()
    if not vals:
        return (float("nan"), float("nan"))
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


print("  %-7s %-14s %5s %8s %20s" % ("scorer", "steps", "n", "rho", "95% CI"))
for scorer in ("magic", "ekfac"):
    pts = []
    for r in rows:
        lds = f(r, scorer + "_lds")
        d = f(r, "filter_%s_delta" % scorer)
        rnd_d = f(r, "filter_random_delta")
        st = f(r, "steps")
        if lds is None or d is None or st is None:
            continue
        pts.append((int(st), lds, d - (rnd_d or 0.0)))
    if not pts:
        continue
    groups = [("all", pts),
              ("<=125", [p for p in pts if p[0] <= 125]),
              (">125", [p for p in pts if p[0] > 125])]
    for name, g in groups:
        if len(g) < 4:
            print("  %-7s %-14s %5d   (too few to correlate)" % (scorer, name, len(g)))
            continue
        xs = [p[1] for p in g]
        ys = [p[2] for p in g]
        rho = spearman(xs, ys)
        lo, hi = boot_ci(xs, ys)
        print("  %-7s %-14s %5d %+8.3f   [%+.3f, %+.3f]" % (scorer, name, len(g), rho, lo, hi))
    steps_seen = sorted({p[0] for p in pts})
    print("       step counts present: %s" % steps_seen)
