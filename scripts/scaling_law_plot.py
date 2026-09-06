"""Log-linear vs power law for QLD-vs-N, on log-log axes.

Panels: (a) EK-FAC top-1%, held-out queries -- the uncontaminated row; (b) the
same row with the in-distribution query set, whose 64k verbatim-overlap step is
visible as a kink no smooth law fits. Fitted curves come from
filter_scaling_law.py so the figure cannot drift from the table.

    python scripts/scaling_law_plot.py     ->  figures/filter_scaling_law.pdf

Not part of scripts/make_figures.py (the paper's figure set); run by hand.
"""
import collections
import csv
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import filter_scaling_law as fsl

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "figures", "filter_scaling_law.pdf")
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#888888"
PANELS = [(("ekfac", "top1%", "held-out"), "(a) held-out queries"),
          (("ekfac", "top1%", "in-dist"), "(b) in-distribution queries")]

cells = fsl.load(fsl.SRC)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, (key, title) in zip(axes, PANELS):
    series = cells[key]
    ns = sorted(series)
    mean = np.array([series[n].mean() for n in ns])
    err = np.array([1.96 * st.stdev(series[n]) / math.sqrt(len(series[n]))
                    for n in ns])
    # Individual per-query observations, as Hoffmann fit individual runs. A
    # log-space fit to them targets the GEOMETRIC mean over queries, which for
    # this right-skewed distribution sits below the arithmetic mean plotted
    # here, so the fit carries the smearing correction.
    logn, y = fsl.obs_arrays(series, ns, means=False)
    grid = np.logspace(np.log10(ns[0]) - 0.15, np.log10(ns[-1]) + 0.15, 200)

    ax.errorbar(ns, mean, yerr=err, fmt="o", color="k", ms=5, lw=1.2,
                capsize=3, zorder=5, label="observed (95% CI over 20 queries)")
    pl, par_l = fsl.fit_loglin(logn, y)
    pp, par_p = fsl.fit_kaplan_huber_smear(logn, y)
    vl = pl(grid)
    ax.plot(grid[vl > 0], vl[vl > 0], color=ORANGE, lw=1.8,
            label=r"log-linear  $\delta=a+b\log_2 N$ (OLS)")
    # where the log-linear fit is non-positive it cannot be drawn on log axes at
    # all; mark the crossing rather than silently clipping the curve.
    zc = fsl.zero_crossing(pl)
    if zc is not None and math.isfinite(zc):
        ax.axvline(zc, color=ORANGE, ls=":", lw=1.2)
        ax.text(zc, mean[0] * 0.42, r" $\delta\!\leq\!0$ below here",
                color=ORANGE, fontsize=8, va="center")
    ax.plot(grid, pp(grid), color=BLUE, lw=1.8,
            label=r"power law  $\delta=AN^{\alpha}$, $\alpha=%.2f$ "
                  r"(Huber/log, smeared)" % par_p["alpha"])
    if key[2] == "in-dist":
        ax.axvline(64000, color=GREY, ls="--", lw=1.0)
        ax.text(64000, mean[0] * 0.42, " query text enters the pool",
                color=GREY, fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Pin the y range to the data. The log-linear fit dives to zero just below
    # the smallest rung; letting it set the limits would squash every point into
    # the top third of the panel to show a curve whose only content is that it
    # leaves the plot.
    ax.set_ylim(mean[0] * 0.3, mean[-1] * 2.2)
    ax.set_xticks(ns)
    ax.set_xticklabels(["%dk" % (n // 1000) for n in ns], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("training documents $N$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("QLD (nats): filter_change $-$ random_mean")
fig.suptitle("EK-FAC proponent filter, top 1% removed: log-linear vs power law "
             "in $N$", fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=170)
print("wrote", OUT)
