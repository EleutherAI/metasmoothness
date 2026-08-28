"""Scatter the proponent-filter delta against the LDS, one panel per scorer.

The question is whether the cheap measurement (filter delta: retrain with the top
1% removed, read the query-loss change) tracks the expensive one (LDS: Spearman
against a 100-retrain bank). Points are rows; colour is step count, because the
interesting part is that the two scorers behave differently as training lengthens.

Spearman rho and its bootstrap CI are printed per panel, and again for the rows
above 125 steps, which is where the EK-FAC relationship falls apart.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

ROOT = Path("/mnt/ssd-2/lucia/metasmoothness")
rows = list(csv.DictReader(open(ROOT / "experiments.csv")))
rng = np.random.default_rng(0)


def boot(x, y, n=10000):
    """Bootstrap the correlation by resampling ROWS, the unit of replication."""
    if len(x) < 4:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = [spearmanr(x[i], y[i]).statistic for i in idx]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def series(scorer):
    out = []
    for r in rows:
        lds, delta, steps = (r.get(f"{scorer}_lds") or "").strip(), \
                            (r.get(f"filter_{scorer}_delta") or "").strip(), \
                            (r.get("steps") or "").strip()
        if lds and delta and steps:
            out.append((float(lds), float(delta), int(steps)))
    return out


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), constrained_layout=True)
for ax, (scorer, label) in zip(axes, [("magic", "MAGIC"), ("ekfac", "EK-FAC")]):
    pts = series(scorer)
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    s = np.array([p[2] for p in pts])
    rho = spearmanr(x, y).statistic
    lo, hi = boot(x, y)
    hi_mask = s > 125
    rho_hi = spearmanr(x[hi_mask], y[hi_mask]).statistic
    lo_hi, hi_hi = boot(x[hi_mask], y[hi_mask])

    sc = ax.scatter(x, y, c=s, norm=LogNorm(), cmap="viridis",
                    s=55, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_title(f"{label}   $\\rho$ = {rho:+.3f} [{lo:+.2f}, {hi:+.2f}]  (n = {len(x)})",
                 fontsize=11)
    ax.text(0.03, 0.96, f">125 steps:  $\\rho$ = {rho_hi:+.3f} [{lo_hi:+.2f}, {hi_hi:+.2f}]"
                        f"   (n = {hi_mask.sum()})",
            transform=ax.transAxes, va="top", fontsize=9, color="#444")
    ax.set_xlabel(f"{label} LDS  (Spearman vs a 100-retrain bank)")
    ax.grid(alpha=0.25, zorder=0)
    print(f"  {label:7s} n={len(x):3d}  rho={rho:+.3f} [{lo:+.3f},{hi:+.3f}]"
          f"   >125 steps n={hi_mask.sum():3d} rho={rho_hi:+.3f} [{lo_hi:+.3f},{hi_hi:+.3f}]")
    print(f"          LDS spans {x.min():.3f}-{x.max():.3f}, "
          f"delta spans {y.min():.3f}-{y.max():.3f}")

axes[0].set_ylabel("Proponent-filter $\\Delta$ (change in query loss)")
fig.colorbar(sc, ax=axes, label="training steps", pad=0.01)
out = ROOT / "figures" / "filter_vs_lds.png"
fig.savefig(out, dpi=160)
print(f"  wrote {out}")
