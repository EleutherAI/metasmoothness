#!/usr/bin/env python3
"""Render the proponent-filter figures from experiments.csv with matplotlib.

    python scripts/scaling_plot_mpl.py    # write figures/filter_scaling.png (main,
                                          # AdamW: corpus scaling + 16k batch sweep)
                                          # and figures/filter_scaling_appendix.png
                                          # (AdamW vs Muon corpus scaling)

Run selection (bs256 rows, the lr 2e-4 re-run preferred at muon 4k) mirrors
scripts/scaling_plot.py; regenerate whenever experiments.csv is rebuilt.
"""
import argparse
import csv
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", type=pathlib.Path, default=ROOT / "figures")
args = ap.parse_args()

# The x dodge separates coincident error bars; multiplicative because x is log.
SERIES = [("AdamW", "#2a78d6", 0.98, ("plan_adam_eps1e17_", "sm_adamw_eps1e17_")),
          ("Muon", "#eb6834", 1.02, ("plan_muon_eps1e17_", "sm_muon_eps1e17_"))]
NS = [4000, 8000, 16000, 32000, 64000, 128000]
BATCHES = [16, 32, 64, 128, 256, 512]
PREFER = ("plan_muon_eps1e17_4k_bs256_lr2e-4",)
# Tokens seen in training: 2 epochs over N docs of 512 tokens each.
tokens = lambda n: 2 * n * 512

rows = list(csv.DictReader(open(ROOT / "experiments.csv")))


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


def delta_ci(r):
    if r is None or not (r.get("filter_ekfac_delta") or "").strip():
        return None
    d = float(r["filter_ekfac_delta"])
    return d, d - float(r["filter_ekfac_lo"]), float(r["filter_ekfac_hi"]) - d


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
    ax.set_ylabel("Change in query loss")
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.09)


def scaling_points(prefixes):
    return [delta_ci(pick_scaling(prefixes, n)) for n in NS]


args.outdir.mkdir(parents=True, exist_ok=True)
tok_ticks = [tokens(n) for n in NS]
tok_labels = [f"{tokens(n) / 1e6:.0f}M" for n in NS]

# Main figure: AdamW corpus scaling beside the AdamW batch sweep at 16k docs.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200)
name, color, _, prefixes = SERIES[0]
draw(ax1, tok_ticks, scaling_points(prefixes), color)
style(ax1, tok_ticks, tok_labels, "Number of training tokens")

batch_points = [delta_ci(pick_batch(prefixes, b)) for b in BATCHES]
draw(ax2, BATCHES, batch_points, color)
style(ax2, BATCHES, [str(b) for b in BATCHES], "Batch size")

fig.tight_layout()
out = args.outdir / "filter_scaling.png"
fig.savefig(out)
print(f"wrote {out}")

# Appendix figure: AdamW vs Muon corpus scaling.
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
for name, color, dodge, prefixes in SERIES:
    missing = draw(ax, [t * dodge for t in tok_ticks], scaling_points(prefixes),
                   color, label=name)
    for x in missing:
        ax.annotate(f"{name}\nretraining", (x, 0), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7,
                    color="#9a988f")
style(ax, tok_ticks, tok_labels, "Number of training tokens")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
          ncols=len(SERIES), borderaxespad=0)

fig.tight_layout()
out = args.outdir / "filter_scaling_appendix.png"
fig.savefig(out)
print(f"wrote {out}")
