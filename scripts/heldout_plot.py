"""GPT-2 (AdamW) EK-FAC proponent filter: in-distribution vs held-out queries, same shape as figures/filter_scaling.pdf.
(a) top 1% removed, (b) top 40 removed. QLD = filter_change - random_mean, 95% CI over the 20 queries.
Held-out summaries are read from <run>/<filter>_heldout/filter_summary.csv (merged) or pooled from *_heldout_q*_* shards.
Writes figures/filter_heldout.pdf, its companion filter_heldout_absolute.pdf (unfiltered / random-control / filtered
mean query loss per query set, same points and CI; scripts/absolute_losses.py) and filter_heldout_relative.pdf
(mean_q (filtered - random) / random in percent, per query set; absolute_losses.relative_point)."""
import csv, glob, math, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import absolute_losses as absl

E = "/mnt/ssd-2/lucia/paper_runs/experiments"
FIG = os.environ.get("FIGURES_DIR") or "/mnt/ssd-2/lucia/metasmoothness/figures"
OUT = f"{FIG}/filter_heldout.pdf"
OUT_ABS = f"{FIG}/filter_heldout_absolute.pdf"
RUNS = {4000: "plan_adam_eps1e17_4k_bs256", 8000: "plan_adam_eps1e17_8k_bs256", 16000: "sm_adamw_eps1e17_16k_bs256",
        32000: "plan_adam_eps1e17_32k_bs256", 64000: "plan_adam_eps1e17_64k_bs256", 128000: "plan_adam_eps1e17_128k_bs256",
        256000: "plan_adam_eps1e17_256k_bs256", 512000: "plan_adam_eps1e17_512k_bs256"}
BLUE, ORANGE = "#2a78d6", "#eb6834"
PANELS = (("(a) Top 1% of documents removed", "filter_proponents_ekfac"), ("(b) Top 40 documents removed", "filter_top40_ekfac"))
CONDITIONS = (("In-distribution queries", BLUE, 0.98), ("Held-out queries", ORANGE, 1.02))
tokens = lambda n: 2 * n * 512


def qld(path):
    if not os.path.exists(path):
        return []
    out = []
    for r in csv.DictReader(open(path)):
        try:
            a, b = float(r["filter_change"]), float(r["random_mean"])
            if a == a and b == b:
                out.append(a - b)
        except (KeyError, ValueError):
            pass
    return out


def heldout(run, f):
    q = qld(f"{E}/{run}/{f}_heldout/filter_summary.csv")
    if q:
        return q
    for d in sorted(glob.glob(f"{E}/{run}/{f}_heldout_q*_*")):
        if os.path.isdir(d) and ".nan" not in d and ".invalid" not in d and ".partial" not in d:
            q += qld(d + "/filter_summary.csv")
    return q if len(q) == 20 else []


def ci(q):
    """(stat, err_lo, err_hi): 1.96 x SEM of the mean, or under QLD_STAT=median
    the bootstrap interval of the median (absolute_losses.boot_ci)."""
    if absl.STAT != "mean":
        return absl.boot_ci(q)
    e = 1.96 * st.stdev(q) / math.sqrt(len(q))
    return st.mean(q), e, e


def subdir(n, f, label):
    """Which result dir feeds (rung, panel, condition); the held-out top-40 at 4k
    is the held-out 1% run (40 docs is 1% of 4k)."""
    if not label.startswith("Held"):
        return f
    if f == "filter_top40_ekfac" and n == 4000 and not heldout(RUNS[n], f):
        return "filter_proponents_ekfac_heldout"
    return f + "_heldout"


def style(ax, title):
    ax.set_xscale("log")
    ax.set_xticks([tokens(n) for n in RUNS])
    ax.set_xticklabels([f"{tokens(n) / 1e6:.0f}M" for n in RUNS])
    ax.minorticks_off()
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Number of training tokens")


fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
for ax, (title, f) in zip(axes, PANELS):
    for label, color, dodge in CONDITIONS:
        getter = (lambda run: heldout(run, f)) if label.startswith("Held") else (lambda run: qld(f"{E}/{run}/{f}/filter_summary.csv"))
        xs, ys, lo, hi = [], [], [], []
        for n, run in RUNS.items():
            q = getter(run)
            if f == "filter_top40_ekfac" and n == 4000 and not q and label.startswith("Held"):
                q = heldout(run, "filter_proponents_ekfac")  # 40 docs is 1% at 4k: same run
            if len(q) < 2:
                continue
            m, e, e_hi = ci(q)
            xs.append(tokens(n) * dodge); ys.append(m); lo.append(e); hi.append(e_hi)
            print(f"{title[:3]} {label[:8]:8s} N={n // 1000:>3}k n={len(q):2d} QLD={m:+.4f} "
                  + (f"+-{e:.4f}" if absl.STAT == "mean" else f"[-{e:.4f} +{e_hi:.4f}]"))
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, label=label, marker="o", markersize=5, linewidth=2, capsize=3, capthick=1.2)
    style(ax, title)
axes[0].set_ylabel(absl.label("Query loss difference"))
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)

# Companion: absolute losses. The two conditions are different query sets, so
# each carries its own unfiltered line; a held-out point needs all 20 queries
# (as above), an in-distribution one at least 2.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
for ax, (title, f) in zip(axes, PANELS):
    for label, color, dodge in CONDITIONS:
        pts = [absl.point(run, subdir(n, f, label), min_queries=20 if label.startswith("Held") else 2)
               for n, run in RUNS.items()]
        absl.draw(ax, [tokens(n) * dodge for n in RUNS], pts, color)
        for n, p in zip(RUNS, pts):
            if p:
                print(f"{title[:3]} {label[:8]:8s} N={n // 1000:>3}k n={p['n']:2d} unfiltered={p['unfiltered'][0]:.4f} "
                      f"random={p['random'][0]:.4f} filtered={p['filtered'][0]:.4f}")
    style(ax, title)
axes[0].set_ylabel(absl.label("Mean query loss"))
absl.legend_above(fig, absl.handles([(l, c) for l, c, _ in CONDITIONS]), 5)
if absl.WRITE_ABSOLUTE:
    fig.savefig(OUT_ABS)
    print("wrote", OUT_ABS)
else:
    print("not written (absolute_losses.WRITE_ABSOLUTE is False):", OUT_ABS)


# Companion: the filter's cost as a percent of its random control's loss,
# 1.96 x SEM over queries as the QLD panel above. A held-out point needs all 20
# queries, an in-distribution one at least 2, as for the absolute companion.
OUT_REL = f"{FIG}/filter_heldout_relative.pdf"

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
for ax, (title, f) in zip(axes, PANELS):
    for label, color, dodge in CONDITIONS:
        xs, ys, lo, hi = [], [], [], []
        for n, run in RUNS.items():
            p = absl.relative_point(run, subdir(n, f, label), ci=absl.sem_ci,
                                    min_queries=20 if label.startswith("Held") else 2)
            if p:
                xs.append(tokens(n) * dodge); ys.append(p[0]); lo.append(p[1]); hi.append(p[2])
                print(f"{title[:3]} {label[:8]:8s} N={n // 1000:>3}k relative={p[0]:5.2f}% "
                      + (f"+-{p[1]:.2f}" if absl.STAT == "mean" else f"[-{p[1]:.2f} +{p[2]:.2f}]"))
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, label=label, marker="o", markersize=5, linewidth=2, capsize=3, capthick=1.2)
    style(ax, title)
axes[0].set_ylabel(absl.label("Query loss increase (%)"))
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
if absl.WRITE_RELATIVE:
    fig.savefig(OUT_REL)
    print("wrote", OUT_REL)
else:
    print("not written (absolute_losses.WRITE_RELATIVE is False):", OUT_REL)
