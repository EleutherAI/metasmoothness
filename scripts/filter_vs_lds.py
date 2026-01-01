#!/usr/bin/env python3
"""Pair each scorer's tail-filter delta against its own LDS, across rows.

The question: does filter power track LDS? Both claim to measure how well a
scorer identifies influential data, but LDS correlates predicted against
measured loss changes over random subsets, while the filter delta measures what
actually happens when you remove the documents the scorer ranks top.

Each scorer is paired with ITS OWN LDS -- MAGIC delta against MAGIC LDS, EK-FAC
delta against EK-FAC LDS -- so this is two separate questions, not a comparison
between scorers.

Reports Spearman (rank) rather than Pearson: the deltas are not on a common
scale across rows. Each row has its own loss scale, visible in its random-removal
control, which spans 4e-05 to 4e-04 -- an order of magnitude -- so a raw delta
from one row is not comparable in magnitude to another's. Rank order survives
that; absolute size does not.

    python scripts/filter_vs_lds.py
"""
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def spearman(xs, ys):
    """Spearman rho with average ranks for ties. n is small; keep it explicit."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    exp = {r["run_id"]: r for r in csv.DictReader(open(REPO / "experiments.csv"))}
    deltas = {r["run"]: r for r in csv.DictReader(open(REPO / "data/filter_deltas.csv"))}

    for scorer, lds_col, d_col in (("MAGIC", "magic_lds", "magic"),
                                   ("EK-FAC", "ekfac_lds", "ekfac")):
        pairs = []
        for rid, d in deltas.items():
            row = exp.get(rid)
            if not row:
                continue
            lds, delta = row.get(lds_col, ""), d.get(f"{d_col}_mean", "")
            rand = d.get("random_mean", "")
            if lds in ("", "None") or delta in ("", "None"):
                continue
            pairs.append((rid, float(lds), float(delta), float(rand or "nan")))

        pairs.sort(key=lambda p: p[1])
        print(f"\n=== {scorer}: filter delta vs {scorer} LDS ===")
        print(f"{'run':<32}{'LDS':>8}{'delta':>10}{'random':>10}")
        for rid, lds, delta, rand in pairs:
            print(f"{rid:<32}{lds:>8.4f}{delta:>10.5f}{rand:>10.5f}")
        if len(pairs) >= 3:
            rho = spearman([p[1] for p in pairs], [p[2] for p in pairs])
            print(f"  Spearman rho = {rho:+.3f}  (n = {len(pairs)})")
        else:
            print(f"  n = {len(pairs)} -- too few rows to rank")


if __name__ == "__main__":
    main()
