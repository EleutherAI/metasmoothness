"""Scatter the proponent-filter delta against the LDS, one panel per scorer.

The question is whether the cheap measurement (filter delta: retrain with the top
1% removed, read the query-loss change) tracks the expensive one (LDS: Spearman
against a 100-retrain bank). Points are rows; colour is step count, because the
interesting part is that the two scorers behave differently as training lengthens.

A second row scatters metasmoothness against the LDS for the same rows. Muon rows
are drawn as triangles in both rows: their metasmoothness matches or beats the
AdamW rows while their MAGIC LDS sits lower, so metasmoothness bounds
attributability without determining it.

Spearman rho and its bootstrap CI are printed per panel, and again for the rows
above 125 steps, which is where the EK-FAC relationship falls apart.
"""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

import absolute_losses as absl

ROOT = Path("/mnt/ssd-2/lucia/metasmoothness")
FIGURES = Path(os.environ.get("FIGURES_DIR") or ROOT / "figures")
rows = list(csv.DictReader(open(ROOT / "experiments.csv")))
import absolute_losses as absl  # noqa: E402
rows = [r for r in rows if r["run_id"] not in absl.EXCLUDE_RUNS]  # logit-scale variants excluded (2026-09-06)
rng = np.random.default_rng(0)


def filter_delta(r, scorer):
    """The row's filter_<scorer>_delta, or under QLD_STAT=median the median
    over queries recomputed from its merged summary (experiments.csv holds only
    the mean); '' when the row has no delta, the csv mean if no summary."""
    yv = (r.get(f"filter_{scorer}_delta") or "").strip()
    if yv and absl.STAT != "mean":
        p = absl.qld_from_summary(r["run_id"], f"filter_proponents_{scorer}")
        if p is not None:
            return str(p[0])
        print(f"  warning: {r['run_id']}/filter_proponents_{scorer}: no complete summary, "
              f"{absl.STAT} point falls back to the experiments.csv mean")
    return yv


def boot(x, y, n=10000):
    """Bootstrap the correlation by resampling ROWS, the unit of replication."""
    if len(x) < 4:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = [spearmanr(x[i], y[i]).statistic for i in idx]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def series(scorer, y_field=None, require_ms=False):
    out = []
    for r in rows:
        # Keep one model and one loss: drop the gpt2-medium and logit-scale rows.
        if r.get("model") != "gpt2" or float(r.get("logit_scale") or 1.0) != 1.0:
            continue
        lds, steps = (r.get(f"{scorer}_lds") or "").strip(), \
                     (r.get("steps") or "").strip()
        yv = (r.get(y_field) or "").strip() if y_field else filter_delta(r, scorer)
        ms = (r.get("metasmoothness") or "").strip()
        if require_ms and not ms:
            continue
        if lds and yv and steps:
            muon = "muon" in (r.get("optimizer") or "").lower()
            out.append((float(lds), float(yv), int(steps),
                        float(ms) if ms else float("nan"), muon))
    return out


def split_scatter(ax, x, y, s, muon):
    """Steps-coloured scatter with Muon rows as triangles."""
    sc = None
    for mask, marker in [(~muon, "o"), (muon, "^")]:
        if mask.any():
            sc = ax.scatter(x[mask], y[mask], c=s[mask], norm=LogNorm(),
                            cmap="viridis", s=55, marker=marker,
                            edgecolor="white", linewidth=0.6, zorder=3)
    return sc


fig, grid = plt.subplots(2, 2, figsize=(11.5, 9.4), constrained_layout=True)
axes = grid[0]
for ax, (scorer, label) in zip(axes, [("magic", "MAGIC"), ("ekfac", "EK-FAC")]):
    pts = series(scorer)
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    s = np.array([p[2] for p in pts])
    muon = np.array([p[4] for p in pts])
    rho = spearmanr(x, y).statistic
    lo, hi = boot(x, y)
    hi_mask = s > 125
    rho_hi = spearmanr(x[hi_mask], y[hi_mask]).statistic
    lo_hi, hi_hi = boot(x[hi_mask], y[hi_mask])

    sc = split_scatter(ax, x, y, s, muon)
    ax.set_title(f"{label}   $\\rho$ = {rho:+.3f} [{lo:+.2f}, {hi:+.2f}]  (n = {len(x)})",
                 fontsize=11)
    ax.text(0.03, 0.96, f">125 steps:  $\\rho$ = {rho_hi:+.3f} [{lo_hi:+.2f}, {hi_hi:+.2f}]"
                        f"   (n = {hi_mask.sum()})",
            transform=ax.transAxes, va="top", fontsize=9, color="#444")
    ax.set_xlabel(f"{label} LDS (100 retrains)")
    ax.grid(alpha=0.25, zorder=0)
    print(f"  {label:7s} n={len(x):3d}  rho={rho:+.3f} [{lo:+.3f},{hi:+.3f}]"
          f"   >125 steps n={hi_mask.sum():3d} rho={rho_hi:+.3f} [{lo_hi:+.3f},{hi_hi:+.3f}]")
    print(f"          LDS spans {x.min():.3f}-{x.max():.3f}, "
          f"delta spans {y.min():.3f}-{y.max():.3f}")

axes[0].set_ylabel(absl.label("Change in query loss"))
lo_x = min(ax.get_xlim()[0] for ax in axes)
hi_x = max(ax.get_xlim()[1] for ax in axes)
for ax in axes:
    ax.set_xlim(lo_x, hi_x)

# Second row: metasmoothness of the same runs, sharing the top row's LDS x-axis
# per column so a run can be traced vertically between rows. Muon as triangles.
for ax, (scorer, label) in zip(grid[1], [("magic", "MAGIC"), ("ekfac", "EK-FAC")]):
    pts = series(scorer, y_field=f"{scorer}_lds", require_ms=True)
    lds = np.array([p[0] for p in pts])
    s = np.array([p[2] for p in pts])
    ms = np.array([p[3] for p in pts])
    muon = np.array([p[4] for p in pts])
    rho = spearmanr(ms, lds).statistic
    lo, hi = boot(ms, lds)
    sc = split_scatter(ax, lds, ms, s, muon)
    ax.set_title(f"{label}   $\\rho$ = {rho:+.3f} [{lo:+.2f}, {hi:+.2f}]  (n = {len(ms)})",
                 fontsize=11)
    ax.set_xlabel(f"{label} LDS (100 retrains)")
    ax.set_xlim(lo_x, hi_x)
    ax.grid(alpha=0.25, zorder=0)
    print(f"  {label:7s} ms row  n={len(ms):3d}  rho(ms, LDS)={rho:+.3f} [{lo:+.3f},{hi:+.3f}]"
          f"   muon rows: {int(muon.sum())}")
grid[1][0].set_ylabel("Metasmoothness")
lo_y = min(ax.get_ylim()[0] for ax in grid[1])
hi_y = max(ax.get_ylim()[1] for ax in grid[1])
for ax in grid[1]:
    ax.set_ylim(lo_y, hi_y)

from matplotlib.lines import Line2D
grid[1][0].legend(handles=[
    Line2D([], [], marker="o", linestyle="", color="#777", label="AdamW"),
    Line2D([], [], marker="^", linestyle="", color="#777", label="Muon")],
    frameon=False, loc="upper left", fontsize=9)

fig.colorbar(sc, ax=grid, label="Training steps", pad=0.01)
out = FIGURES / "filter_vs_lds.pdf"
fig.savefig(out)
print(f"  wrote {out}")
