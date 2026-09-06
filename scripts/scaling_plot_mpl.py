#!/usr/bin/env python3
"""Render the proponent-filter figures from experiments.csv with matplotlib.

    python scripts/scaling_plot_mpl.py    # write figures/filter_scaling.pdf (main,
                                          # AdamW, held-out queries: 1% filter +
                                          # fixed-40-document filter vs corpus
                                          # size) and the appendix
                                          # figures (Muon row: corpus scaling +
                                          # batch sweep, EK-FAC vs MAGIC, 16k
                                          # variants), each appendix figure beside
                                          # its *_absolute.pdf companion showing the
                                          # unfiltered / random-control / filtered
                                          # query losses the QLD is the difference of
                                          # (scripts/absolute_losses.py); the main
                                          # figure's companion comes from
                                          # scripts/plot_filter_absolute_losses.py

Run selection (bs256 rows, the lr 2e-4 re-run preferred at muon 4k) mirrors
scripts/scaling_plot.py; the fixed-40 deltas mirror scripts/top40_curve.py.
Regenerate whenever experiments.csv is rebuilt or a top-40 filter lands
(scripts/make_figures.py runs every figure script and checks the output set).
"""
import argparse
import csv
import math
import os
import pathlib
import statistics

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import absolute_losses as absl

ROOT = pathlib.Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", type=pathlib.Path,
                default=pathlib.Path(os.environ.get("FIGURES_DIR") or ROOT / "figures"))
args = ap.parse_args()

BLUE, ORANGE, AQUA, BM25, RANDOM = "#2a78d6", "#eb6834", "#1baf7a", "#5c5c5c", "#8f8f8f"
# The x dodge separates coincident error bars; multiplicative because x is log.
SERIES = [("AdamW", BLUE, 0.98, ("plan_adam_eps1e17_", "sm_adamw_eps1e17_")),
          ("Muon", ORANGE, 1.02, ("plan_muon_eps1e17_", "sm_muon_eps1e17_"))]
NS = [4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]
BATCHES = [16, 32, 64, 128, 256, 512]
ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments", "/mnt/ssd-1/lucia/paper_runs/experiments"]
TOP40_ROWS = [(4000, "plan_adam_eps1e17_4k_bs256"),
              (8000, "plan_adam_eps1e17_8k_bs256"),
              (16000, "sm_adamw_eps1e17_16k_bs256"),
              (32000, "plan_adam_eps1e17_32k_bs256"),
              (64000, "plan_adam_eps1e17_64k_bs256"),
              (128000, "plan_adam_eps1e17_128k_bs256"),
              (256000, "plan_adam_eps1e17_256k_bs256"),
              (512000, "plan_adam_eps1e17_512k_bs256")]
VARIANT_ROWS = [("Baseline (bs 256)", "sm_adamw_eps1e17_16k_bs256"),
                ("Weight decay 0.0", "plan_adam_eps1e17_16k_wd0.0"),
                ("Weight decay 0.1", "plan_adam_eps1e17_16k_wd0.1"),
                ("Grad clip 1.0", "plan_adam_eps1e17_16k_clip1.0"),
                ("4 epochs", "plan_adam_eps1e17_16k_ep4"),
                ("GPT-2 medium", "plan_adam_eps1e17_16k_gpt2-medium"),
                ]  # logit-scale variants excluded (absolute_losses.EXCLUDE_RUNS, 2026-09-06)
PREFER = ("plan_muon_eps1e17_4k_bs256_lr2e-4",)
# The Muon corpus-scaling series stops at 256k documents (2026-09-04: the
# 512k Muon filter was dropped from the plan).
SERIES_MAX_N = {"Muon": 256000}
# Tokens seen in training: 2 epochs over N docs of 512 tokens each.
tokens = lambda n: 2 * n * 512

rows = list(csv.DictReader(open(ROOT / "experiments.csv")))
by_id = {r["run_id"]: r for r in rows}


def pick_scaling(prefixes, n):
    for r in sorted(rows, key=lambda r: r["run_id"] not in PREFER):
        rid = r["run_id"]
        if not (rid.endswith("_bs256") or "_bs256_" in rid):
            continue
        if not rid.startswith(prefixes):
            continue
        try:
            if int(float(r["n_docs"])) != n:
                continue
        except (TypeError, ValueError):
            continue
        return r
    return None


def pick_batch(prefixes, bs):
    for r in rows:
        rid = r["run_id"]
        if rid.startswith(prefixes) and rid.endswith(f"16k_bs{bs}"):
            return r
    return None


def summary_delta_ci(run, subdir, *, subtract_random=False,
                     column="filter_change", boot=10000, min_rows=20):
    root = next((r for r in ROOTS if os.path.isdir(os.path.join(r, run))), None)
    path = os.path.join(root, run, subdir, "filter_summary.csv") if root else None
    if not (path and os.path.isfile(path)):
        return None
    d = []
    for row in csv.DictReader(open(path)):
        val = float(row[column])
        if subtract_random:
            val -= float(row["random_mean"])
        # A single NaN query (e.g. a diverged retrain) would turn the whole
        # point into NaN and drop it from the figure silently. Skip it and
        # let the remaining queries carry the mean.
        if not math.isfinite(val):
            continue
        d.append(val)
    if len(d) < min_rows:
        print(f"  skip {run}/{subdir}: {len(d)} finite queries < {min_rows}")
        return None
    return absl.boot_ci(d, boot)


def summary_scaling_points(prefixes, subdir, column="filter_change",
                           subtract_random=False):
    pts = []
    for n in NS:
        r = pick_scaling(prefixes, n)
        pts.append(summary_delta_ci(r["run_id"], subdir, column=column,
                                    subtract_random=subtract_random)
                   if r else None)
    return pts


def delta_ci(r, method="ekfac"):
    if r is None or not (r.get(f"filter_{method}_delta") or "").strip():
        return None
    if absl.STAT != "mean":
        # experiments.csv holds only the mean and its CI: recompute the point
        # from the merged summary, as summary_delta_ci does for BM25 / top-40.
        p = absl.qld_from_summary(r["run_id"], f"filter_proponents_{method}")
        if p is not None:
            return p
        print(f"  warning: {r['run_id']}/filter_proponents_{method}: no complete "
              f"summary, {absl.STAT} point falls back to the experiments.csv mean")
    d = float(r[f"filter_{method}_delta"])
    return (d, d - float(r[f"filter_{method}_lo"]),
            float(r[f"filter_{method}_hi"]) - d)


def draw(ax, xs, points, color, label=None):
    kept = [(x, p) for x, p in zip(xs, points) if p is not None]
    x, d, lo, hi = zip(*[(x, d, lo, hi) for x, (d, lo, hi) in kept])
    ax.errorbar(x, d, yerr=[lo, hi], color=color, label=label,
                marker="o", markersize=5, linewidth=2, capsize=3, capthick=1.2)
    return [x for x, p in zip(xs, points) if p is None]


def style(ax, xticks, xlabels, xlabel):
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks, xlabels)
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(absl.label("Query loss difference"))
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.09)


def outside_legend(ax, n):
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncols=n, borderaxespad=0)


def save(fig, name, layout=True):
    # legend_above already ran tight_layout with a rect reserving the legend
    # strip; a second bare tight_layout here would discard that rect.
    if layout:
        fig.tight_layout()
    out = args.outdir / name
    if name.endswith("_absolute.pdf") and not absl.WRITE_ABSOLUTE:
        print(f"not written (absolute_losses.WRITE_ABSOLUTE is False): {out}")
        return
    fig.savefig(out)
    print(f"wrote {out}")


def scaling_points(prefixes, method="ekfac"):
    return [delta_ci(pick_scaling(prefixes, n), method) for n in NS]


def absolute_point(r, subdir):
    """Unfiltered / random / filtered mean query loss of one row's filter result,
    bootstrap CI over queries exactly as summary_delta_ci draws the QLD."""
    return absl.point(r["run_id"], subdir, ci=absl.boot_ci) if r else None


def absolute_scaling_points(prefixes, subdir, ns=NS):
    return [absolute_point(pick_scaling(prefixes, n), subdir) for n in ns]


def style_absolute(ax, xticks, xlabels, xlabel):
    style(ax, xticks, xlabels, xlabel)
    ax.set_ylabel(absl.label("Mean query loss"))


args.outdir.mkdir(parents=True, exist_ok=True)
tok_ticks = [tokens(n) for n in NS]
tok_labels = [f"{tokens(n) / 1e6:.0f}M" for n in NS]

# Main figure: AdamW 1% filter beside the fixed-40-document filter, on the
# HELD-OUT query set (2026-09-05: switched from the in-distribution queries;
# the companions filter_scaling_absolute / _relative and every appendix figure
# still use the in-distribution set, and filter_heldout.pdf shows both). Both
# series are raw changes in query loss relative to the unfiltered run (no random
# subtraction), so proponent and random filters share an axis and QLD is the
# gap between the curves; the other figures keep plotting QLD itself.
# The held-out top-40 at 4k is the held-out 1% run: 40 docs is 1% of 4k.
HELDOUT_1PCT = "filter_proponents_ekfac_heldout"
HELDOUT_TOP40 = {n: ("filter_top40_ekfac_heldout" if n != 4000 else HELDOUT_1PCT)
                 for n, _ in TOP40_ROWS}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
name, color, _, prefixes = SERIES[0]
draw(ax1, tok_ticks,
     summary_scaling_points(prefixes, HELDOUT_1PCT),
     color, label="EK-FAC proponents")
draw(ax1, tok_ticks,
     summary_scaling_points(prefixes, HELDOUT_1PCT, column="random_mean"),
     RANDOM, label="Random filter")
style(ax1, tok_ticks, tok_labels, "Number of training tokens")
ax1.set_ylabel(absl.label("Change in query loss"))
ax1.set_title("(a) Top 1% of documents removed", fontsize=10)
ax1.legend(loc="upper left", frameon=False, fontsize=9)

top40_ticks = [tokens(n) for n, _ in TOP40_ROWS]
draw(ax2, top40_ticks,
     [summary_delta_ci(run, HELDOUT_TOP40[n]) for n, run in TOP40_ROWS],
     color)
draw(ax2, top40_ticks,
     [summary_delta_ci(run, HELDOUT_TOP40[n], column="random_mean")
      for n, run in TOP40_ROWS], RANDOM)
style(ax2, top40_ticks, [f"{t / 1e6:.0f}M" for t in top40_ticks],
      "Number of training tokens")
ax2.set_title("(b) Top 40 documents removed", fontsize=10)
ax2.set_ylabel(None)
main_ylim = ax1.get_ylim()
save(fig, "filter_scaling.pdf")

# Appendix figure: the Muon comparison on the batch-size sweep at 16k documents.
# The corpus-scaling panel that used to sit beside it was dropped on 2026-09-05
# in favour of a table (scripts/qld_lds_table.py -> tables/qld_lds_adamw_muon.tex);
# it had also inherited main_ylim from the main figure, which clipped it once the
# main figure moved to held-out queries.
muon_ns = [n for n in NS if n <= min(SERIES_MAX_N.values())]  # used by the *_absolute path
fig, ax2 = plt.subplots(figsize=(6, 3.8), dpi=200)
for name, color, dodge, prefixes in SERIES:
    points = [delta_ci(pick_batch(prefixes, b)) for b in BATCHES]
    draw(ax2, [b * dodge for b in BATCHES], points, color, label=name)
style(ax2, BATCHES, [str(b) for b in BATCHES], "Batch size")
ax2.set_title("Batch size (16k documents)", fontsize=10)
ax2.margins(y=0.6)  # headroom so the legend clears the bs16/bs32 error bars
ax2.legend(frameon=False, loc="upper left")
save(fig, "filter_muon_appendix.pdf")

# Companion: the same panels as absolute losses. Each optimizer has its own
# unfiltered model, so all three series are drawn per optimizer in its colour.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
for name, color, dodge, prefixes in SERIES:
    absl.draw(ax1, [tokens(n) * dodge for n in muon_ns],
              absolute_scaling_points(prefixes, "filter_proponents_ekfac", muon_ns), color)
    absl.draw(ax2, [b * dodge for b in BATCHES],
              [absolute_point(pick_batch(prefixes, b), "filter_proponents_ekfac")
               for b in BATCHES], color)
style_absolute(ax1, [tokens(n) for n in muon_ns],
               [f"{tokens(n) / 1e6:.0f}M" for n in muon_ns], "Number of training tokens")
style_absolute(ax2, BATCHES, [str(b) for b in BATCHES], "Batch size")
ax1.set_title("Corpus scaling", fontsize=10)
ax2.set_title("Batch size (16k documents)", fontsize=10)
ax2.set_ylabel(None)
absl.legend_above(fig, absl.handles([(name, color) for name, color, _, _ in SERIES]), 5)
save(fig, "filter_muon_appendix_absolute.pdf", layout=False)

# Appendix figure: EK-FAC vs MAGIC vs BM25 proponent filters, AdamW corpus
# scaling. Serial MAGIC scoring stops at 64k documents, so the 1% panel is
# truncated to that common range; the top-40 panel shows EK-FAC and BM25 over
# the full chain (add MAGIC when its top-40 filter runs land). Both panels
# plot QLD (random control subtracted).

# [heldout-64k] Held-out query set at 64k (plan_adam_eps1e17_64k_bs256, filter_*_<method>_heldout):
# hollow markers at the in-distribution point's x, same color/dodge, QLD with the
# random control subtracted. Only 64k has held-out rows for every method; a
# missing summary (e.g. MAGIC until its held-out filters land) is skipped.
HELDOUT_RUN = "plan_adam_eps1e17_64k_bs256"
HELDOUT_METHODS = [("ekfac", BLUE, 0.97), ("magic", AQUA, 1.0), ("bm25", BM25, 1.03)]


def draw_heldout(ax, x, subdir, color, dodge, relative=False):
    p = (absl.relative_point(HELDOUT_RUN, subdir) if relative
         else summary_delta_ci(HELDOUT_RUN, subdir, subtract_random=True))
    if not p:
        return False
    ax.errorbar([x * dodge], [p[0]], yerr=[[p[1]], [p[2]]], color=color, marker="o",
                fillstyle="none", markersize=6, linestyle="none", linewidth=2,
                capsize=3, capthick=1.2, markeredgewidth=1.6)
    return True


def heldout_handle():
    return Line2D([], [], linestyle="none", marker="o", fillstyle="none", color="0.3",
                  markersize=6, markeredgewidth=1.6, label="held-out queries (64k)")

CUT = NS.index(64000) + 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
prefixes = SERIES[0][3]
for method, color, dodge in [("ekfac", BLUE, 0.97), ("magic", AQUA, 1.0)]:
    label = {"ekfac": "EK-FAC", "magic": "MAGIC"}[method]
    draw(ax1, [t * dodge for t in tok_ticks[:CUT]],
         scaling_points(prefixes, method)[:CUT], color, label=label)
draw(ax1, [t * 1.03 for t in tok_ticks[:CUT]],
     summary_scaling_points(prefixes, "filter_proponents_bm25",
                            subtract_random=True)[:CUT],
     BM25, label="BM25")
for _m, _c, _d in HELDOUT_METHODS:  # [heldout-64k]
    draw_heldout(ax1, tokens(64000), f"filter_proponents_{_m}_heldout", _c, _d)
style(ax1, tok_ticks[:CUT], tok_labels[:CUT], "Number of training tokens")
ax1.set_title("(a) Top 1% of documents removed", fontsize=10)
ax1.legend(handles=[*ax1.get_legend_handles_labels()[0], heldout_handle()], loc="upper left", frameon=False, fontsize=9)

# Top-40 panel capped at 64k docs (66M tokens) so all three methods span the
# same range as panel (a) and the MAGIC series; 128k+ dropped.
T40 = [(n, run) for n, run in TOP40_ROWS if n <= 64000]
t40_ticks = [tokens(n) for n, _ in T40]
draw(ax2, [t * 0.97 for t in t40_ticks],
     [summary_delta_ci(run, "filter_top40_ekfac", subtract_random=True)
      for _, run in T40], BLUE)
draw(ax2, [t for t in t40_ticks],
     [summary_delta_ci(run, "filter_top40_magic", subtract_random=True)
      for _, run in T40], AQUA)
draw(ax2, [t * 1.03 for t in t40_ticks],
     [summary_delta_ci(run, "filter_top40_bm25", subtract_random=True)
      for _, run in T40], BM25)
for _m, _c, _d in HELDOUT_METHODS:  # [heldout-64k]
    draw_heldout(ax2, tokens(64000), f"filter_top40_{_m}_heldout", _c, _d)
style(ax2, t40_ticks, [f"{t / 1e6:.0f}M" for t in t40_ticks],
      "Number of training tokens")
ax2.set_title("(b) Top 40 documents removed", fontsize=10)
ax2.set_ylabel(None)
save(fig, "filter_method_appendix.pdf")

# Companion: absolute losses per method. The three methods filter the same
# row, so its unfiltered loss is drawn once (black); each method contributes
# its random-control and filtered lines, read from its own filter summary.
METHODS = [("EK-FAC", "ekfac", BLUE, 0.97), ("MAGIC", "magic", AQUA, 1.0), ("BM25", "bm25", BM25, 1.03)]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
for ax, ticks, subdir, pts in (
        (ax1, tok_ticks[:CUT], "filter_proponents_",
         lambda src: absolute_scaling_points(prefixes, "filter_proponents_" + src, NS[:CUT])),
        (ax2, t40_ticks, "filter_top40_",
         lambda src: [absl.point(run, "filter_top40_" + src, ci=absl.boot_ci) for _, run in T40])):
    absl.draw(ax, ticks, pts("ekfac"), None, series=("unfiltered",), neutral=True)
    for label, src, color, dodge in METHODS:
        absl.draw(ax, [t * dodge for t in ticks], pts(src), color, series=("random", "filtered"))
style_absolute(ax1, tok_ticks[:CUT], tok_labels[:CUT], "Number of training tokens")
style_absolute(ax2, t40_ticks, [f"{t / 1e6:.0f}M" for t in t40_ticks], "Number of training tokens")
ax1.set_title("(a) Top 1% of documents removed", fontsize=10)
ax2.set_title("(b) Top 40 documents removed", fontsize=10)
ax2.set_ylabel(None)
absl.legend_above(fig, [*absl.handles(series=("unfiltered",), series_color=absl.NEUTRAL["unfiltered"]),
                        *absl.handles([(l, c) for l, _, c, _ in METHODS], series=("random", "filtered"))], 6)
save(fig, "filter_method_appendix_absolute.pdf", layout=False)

# Appendix figure: training-setup variants at 16k documents, AdamW.
fig, ax = plt.subplots(figsize=(7, 3.8), dpi=200)
ys = range(len(VARIANT_ROWS))[::-1]
for y, (label, run) in zip(ys, VARIANT_ROWS):
    p = delta_ci(by_id[run])
    if p is None:
        continue
    d, lo, hi = p
    ax.errorbar([d], [y], xerr=[[lo], [hi]], color=BLUE, marker="o",
                markersize=5, linewidth=2, capsize=3, capthick=1.2)
ax.set_yticks(list(ys), [label for label, _ in VARIANT_ROWS])
ax.set_xlabel(absl.label("Query loss difference"))
ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
ax.set_axisbelow(True)
ax.margins(y=0.12)
save(fig, "filter_variants_appendix.pdf")

# Companion: the three absolute losses per variant, on a shared loss axis.
fig, ax = plt.subplots(figsize=(7, 3.8), dpi=200)
for y, (label, run) in zip(ys, VARIANT_ROWS):
    p = absl.point(run, "filter_proponents_ekfac", ci=absl.boot_ci)
    if p is None:
        continue
    for s in absl.SERIES:
        m, lo, hi = p[s]
        st = {k: v for k, v in absl.STYLE[s].items() if k != "linestyle"}
        ax.errorbar([m], [y], xerr=[[lo], [hi]], linestyle="", capsize=3, capthick=1.2,
                    color=absl.NEUTRAL.get(s, BLUE), **st)
ax.set_yticks(list(ys), [label for label, _ in VARIANT_ROWS])
ax.set_xlabel(absl.label("Mean query loss"))
ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
ax.set_axisbelow(True)
ax.margins(y=0.12)
ax.legend(handles=[Line2D([], [], linestyle="", color=absl.NEUTRAL.get(s, BLUE),
                          label=absl.LABEL[s], marker=absl.STYLE[s]["marker"])
                   for s in absl.SERIES],
          frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3,
          borderaxespad=0, fontsize=9)
save(fig, "filter_variants_appendix_absolute.pdf")


# ------------------------------------------------------------------ relative
if not absl.WRITE_RELATIVE:  # 2026-09-06: companions off by default (QLD_RELATIVE=1 re-enables)
    print("relative companions not written (absolute_losses.WRITE_RELATIVE is False)")
    raise SystemExit(0)
# Each QLD appendix figure again as a percent of the matched control's loss,
# mean_q (filtered - random) / random, bootstrap CI over queries as the
# QLD points (absolute_losses.relative_point).


def rel_points(rows, subdir):
    """[(x, point)] for [(x, run_id)]."""
    return [(x, absl.relative_point(run, subdir)) for x, run in rows]


def draw_rel(ax, rows, subdir, color, label=None, dodge=1.0):
    kept = [(x * dodge, p) for x, p in rel_points(rows, subdir) if p]
    if kept:
        ax.errorbar([x for x, _ in kept], [p[0] for _, p in kept],
                    yerr=[[p[1] for _, p in kept], [p[2] for _, p in kept]],
                    color=color, label=label, marker="o", markersize=5,
                    linewidth=2, capsize=3, capthick=1.2)


def style_rel(ax, xticks, xlabels, xlabel):
    style(ax, xticks, xlabels, xlabel)
    ax.set_ylabel(absl.label("Query loss increase (%)"))


def rows_for(prefixes, ns):
    return [(tokens(n), r["run_id"]) for n in ns
            if (r := pick_scaling(prefixes, n)) is not None]


ADAMW = SERIES[0][3]

# Muon appendix: the batch sweep, both optimizers (corpus scaling is the table).
fig, ax2 = plt.subplots(figsize=(6, 3.8), dpi=200)
for name, color, dodge, prefixes in SERIES:
    draw_rel(ax2, [(b, r["run_id"]) for b in BATCHES
                   if (r := pick_batch(prefixes, b)) is not None],
             "filter_proponents_ekfac", color, label=name, dodge=dodge)
style_rel(ax2, BATCHES, [str(b) for b in BATCHES], "Batch size")
ax2.set_title("Batch size (16k documents)", fontsize=10)
ax2.margins(y=0.6)
ax2.legend(frameon=False, loc="upper left")
save(fig, "filter_muon_appendix_relative.pdf")

# Method appendix: EK-FAC vs MAGIC vs BM25, same panels as the QLD figure.
# ADAMW, not `prefixes`: that name holds whatever the Muon loop left in it.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
pct_rows = rows_for(ADAMW, NS[:CUT])
for label, src, color, dodge in METHODS:
    draw_rel(ax1, pct_rows, "filter_proponents_" + src, color, label=label, dodge=dodge)
    draw_rel(ax2, [(tokens(n), run) for n, run in T40], "filter_top40_" + src,
             color, dodge=dodge)
for _m, _c, _d in HELDOUT_METHODS:  # [heldout-64k]
    draw_heldout(ax1, tokens(64000), f"filter_proponents_{_m}_heldout", _c, _d, relative=True)
    draw_heldout(ax2, tokens(64000), f"filter_top40_{_m}_heldout", _c, _d, relative=True)
style_rel(ax1, tok_ticks[:CUT], tok_labels[:CUT], "Number of training tokens")
style_rel(ax2, t40_ticks, [f"{t / 1e6:.0f}M" for t in t40_ticks], "Number of training tokens")
ax1.set_title("(a) Top 1% of documents removed", fontsize=10)
ax2.set_title("(b) Top 40 documents removed", fontsize=10)
ax2.set_ylabel(None)
ax1.legend(handles=[*ax1.get_legend_handles_labels()[0], heldout_handle()], loc="upper left", frameon=False, fontsize=9)
save(fig, "filter_method_appendix_relative.pdf")

# Variants appendix, one row per training-setup variant at 16k.
fig, ax = plt.subplots(figsize=(7, 3.8), dpi=200)
for y, (label, run) in zip(ys, VARIANT_ROWS):
    p = absl.relative_point(run, "filter_proponents_ekfac")
    if p is None:
        continue
    ax.errorbar([p[0]], [y], xerr=[[p[1]], [p[2]]], color=BLUE, marker="o",
                markersize=5, linewidth=2, capsize=3, capthick=1.2)
    print(f"  {label:20s} relative={p[0]:5.2f}% [-{p[1]:.2f} +{p[2]:.2f}]")
ax.set_yticks(list(ys), [label for label, _ in VARIANT_ROWS])
ax.set_xlabel(absl.label("Query loss increase (%)"))
ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
ax.set_axisbelow(True)
ax.margins(y=0.12)
save(fig, "filter_variants_appendix_relative.pdf")
