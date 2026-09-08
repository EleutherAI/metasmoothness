"""Qwen2.5-1.5B proponent-filter scaling, same shape as figures/filter_scaling.pdf:
(a) top x% of documents removed (1%, 5%, 10%), (b) top-k documents removed (40, 200, 400 as available).
QLD = filter_change - random_mean per query, 95% CI over the 20 queries; x = training tokens (2 epochs x N docs x 512).
Writes figures/filter_scaling_qwen.pdf, its companion filter_scaling_qwen_absolute.pdf (unfiltered / random-control /
filtered mean query loss per condition, same points and CI; scripts/absolute_losses.py) and
filter_scaling_qwen_relative.pdf (mean_q (filtered - random) / random in percent; absolute_losses.relative_point)."""
import csv, math, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import absolute_losses as absl

E = "/mnt/ssd-2/lucia/paper_runs/experiments"
FIG = os.environ.get("FIGURES_DIR") or "/mnt/ssd-2/lucia/metasmoothness/figures"
OUT = f"{FIG}/filter_scaling_qwen.pdf"
OUT_ABS = f"{FIG}/filter_scaling_qwen_absolute.pdf"
NS = [4000, 8000, 16000, 32000, 64000, 128000]
BLUE, GREEN, ORANGE, PURPLE = "#2a78d6", "#1baf7a", "#eb6834", "#8e5bd6"
PCT = [("Top 1%", "filter_proponents_ekfac", BLUE), ("Top 5%", "filter_prop5pct_ekfac", GREEN), ("Top 10%", "filter_prop10pct_ekfac", ORANGE)]
TOPK = [("Top 40", "filter_top40_ekfac", BLUE), ("Top 200", "filter_top200_ekfac", GREEN), ("Top 400", "filter_top400_ekfac", ORANGE)]
DODGE = (0.97, 1.0, 1.03)
tokens = lambda n: 2 * n * 512
run = lambda n: f"qwen15b_{n // 1000}k_bs256"


# A fixed-document run whose count equals a fraction run's count at that N is the same experiment (same selection,
# same number of docs removed): 200 docs = 5% of 4k, 400 docs = 10% of 4k = 5% of 8k. Read those runs instead of
# re-running them (n_removed in the fraction summaries: 4k 5%=200, 4k 10%=400, 8k 5%=400).
ALIAS = {(4000, "filter_top200_ekfac"): "filter_prop5pct_ekfac",
         (4000, "filter_top400_ekfac"): "filter_prop10pct_ekfac",
         (8000, "filter_top400_ekfac"): "filter_prop5pct_ekfac"}


def qld(n, d):
    d = ALIAS.get((n, d), d)
    p = f"{E}/{run(n)}/{d}/filter_summary.csv"
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if len(rows) < 20:
        return None
    q = [float(r["filter_change"]) - float(r["random_mean"]) for r in rows]
    if absl.STAT != "mean":  # (median, err_lo, err_hi): bootstrap of the median
        return absl.boot_ci(q)
    e = 1.96 * st.stdev(q) / math.sqrt(len(q))
    return st.mean(q), e, e


def draw(ax, series):
    for label, d, color in series:
        pts = [(tokens(n),) + qld(n, d) for n in NS if qld(n, d)]
        if not pts:
            continue
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[[p[2] for p in pts], [p[3] for p in pts]],
                    color=color, label=label, marker="o", markersize=5, linewidth=2, capsize=3, capthick=1.2)


def style(ax, title, legend=True):
    ax.set_xscale("log")
    ax.set_xticks([tokens(n) for n in NS])
    ax.set_xticklabels([f"{tokens(n) / 1e6:.0f}M" for n in NS])
    ax.minorticks_off()
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Number of training tokens")
    if legend:
        ax.legend(frameon=False, fontsize=9, loc="upper left")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
draw(ax1, PCT)
style(ax1, "(a) Top x% of documents removed")
ax1.set_ylabel(absl.label("Query loss difference"))
draw(ax2, TOPK)
style(ax2, "(b) Top k documents removed")
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
for n in NS:
    print(f"{n // 1000:>3}k:", "  ".join(f"{lab}={qld(n, d)[0]:+.4f}" for lab, d, _ in PCT + TOPK if qld(n, d)))


# Companion: absolute losses. Every condition filters the same rung model and
# query set, so the unfiltered loss is drawn once per panel (black, from the
# top-1% / top-40 result); each condition adds its random-control and filtered
# lines. A point needs the same complete (20-query, merged) summary as the QLD.
def absolute(n, d):
    d = ALIAS.get((n, d), d)
    return absl.point(run(n), d, min_queries=20, pool_summary=False)


fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
legends = []
for ax, series, title in ((axes[0], PCT, "(a) Top x% of documents removed"), (axes[1], TOPK, "(b) Top k documents removed")):
    absl.draw(ax, [tokens(n) for n in NS], [absolute(n, series[0][1]) for n in NS], None, series=("unfiltered",), neutral=True)
    for (label, d, color), dodge in zip(series, DODGE):
        pts = [absolute(n, d) for n in NS]
        absl.draw(ax, [tokens(n) * dodge for n in NS], pts, color, series=("random", "filtered"))
        for n, p in zip(NS, pts):
            if p:
                print(f"{title[:3]} {label:7s} N={n // 1000:>3}k n={p['n']:2d} unfiltered={p['unfiltered'][0]:.4f} "
                      f"random={p['random'][0]:.4f} filtered={p['filtered'][0]:.4f}")
    style(ax, title, legend=False)
    legends.append((ax, [*absl.handles(series=("unfiltered",), series_color=absl.NEUTRAL["unfiltered"]),
                         *absl.handles([(l, c) for l, _, c in series], series=("random", "filtered"))]))
axes[0].set_ylabel(absl.label("Mean query loss"))
absl.legend_above_each(fig, legends, 3)
if absl.WRITE_ABSOLUTE:
    fig.savefig(OUT_ABS)
    print("wrote", OUT_ABS)
else:
    print("not written (absolute_losses.WRITE_ABSOLUTE is False):", OUT_ABS)


# Companion: the filter's cost as a percent of its random control's loss,
# 1.96 x SEM over queries as the QLD panels. Same complete-summary requirement.
OUT_REL = f"{FIG}/filter_scaling_qwen_relative.pdf"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
for ax, series, title in ((ax1, PCT, "(a) Top x% of documents removed"), (ax2, TOPK, "(b) Top k documents removed")):
    for (label, d, color), dodge in zip(series, DODGE):
        pts = [(n, absl.relative_point(run(n), ALIAS.get((n, d), d), ci=absl.sem_ci, min_queries=20, pool_summary=False)) for n in NS]
        pts = [(n, p) for n, p in pts if p]
        if not pts:
            continue
        ax.errorbar([tokens(n) * dodge for n, _ in pts], [p[0] for _, p in pts],
                    yerr=[[p[1] for _, p in pts], [p[2] for _, p in pts]],
                    color=color, label=label, marker="o", markersize=5, linewidth=2, capsize=3, capthick=1.2)
        for n, p in pts:
            print(f"{title[:3]} {label:7s} N={n // 1000:>3}k relative={p[0]:5.2f}% "
                  + (f"+-{p[1]:.2f}" if absl.STAT == "mean" else f"[-{p[1]:.2f} +{p[2]:.2f}]"))
    style(ax, title)
ax1.set_ylabel(absl.label("Query loss increase (%)"))
fig.tight_layout()
if absl.WRITE_RELATIVE:
    fig.savefig(OUT_REL)
    print("wrote", OUT_REL)
else:
    print("not written (absolute_losses.WRITE_RELATIVE is False):", OUT_REL)
