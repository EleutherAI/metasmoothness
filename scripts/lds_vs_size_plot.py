#!/usr/bin/env python3
"""LDS versus training-set size for the AdamW bs256 series (4k-32k): one row, two panels (MAGIC, EK-FAC).

    python scripts/lds_vs_size_plot.py [--outdir figures]   # -> figures/lds_vs_size.pdf

Inputs: experiments.csv columns magic_lds / magic_ci_lo / magic_ci_hi and ekfac_lds / ekfac_ci_lo / ekfac_ci_hi for the
rows of the methods table (TOP40_ROWS: plan_adam_eps1e17_{4k,8k,32k}_bs256 and sm_adamw_eps1e17_16k_bs256); 64k is
excluded because its LDS bank is not part of the paper's four tractable sizes. Same axis style as scaling_plot_mpl.py.
"""
import argparse
import csv
import os
import pathlib

import matplotlib.pyplot as plt

import absolute_losses as absl

ROOT = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path(os.environ.get("FIGURES_DIR") or ROOT / "figures"))
args = ap.parse_args()

RUNS = {4000: "plan_adam_eps1e17_4k_bs256", 8000: "plan_adam_eps1e17_8k_bs256",
        16000: "sm_adamw_eps1e17_16k_bs256", 32000: "plan_adam_eps1e17_32k_bs256"}
rows = {r["run_id"]: r for r in csv.DictReader(open(ROOT / "experiments.csv"))}
COLORS = getattr(absl, "METHOD_COLORS", {})
PANELS = [("magic", "(a) MAGIC", COLORS.get("magic", "#c0392b")), ("ekfac", "(b) EK-FAC", COLORS.get("ekfac", "#1f77b4"))]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200)
for ax, (m, title, color) in zip(axes, PANELS):
    xs, ys, lo, hi = [], [], [], []
    for n, run in RUNS.items():
        r = rows[run]
        v = (r.get(f"{m}_lds") or "").strip()
        if not v:
            print(f"  warning: {run} has no {m}_lds"); continue
        xs.append(n); ys.append(float(v)); lo.append(float(v) - float(r[f"{m}_ci_lo"])); hi.append(float(r[f"{m}_ci_hi"]) - float(v))
    ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker="o", markersize=5, linewidth=2, capsize=3, capthick=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(RUNS), [f"{n // 1000}k" for n in RUNS])
    ax.minorticks_off()
    ax.set_xlabel("Training documents")
    ax.set_ylabel("LDS (Spearman)")
    ax.set_ylim(0, 1)
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.09)
    ax.set_title(title, fontsize=10)
    for x, y, l in zip(xs, ys, lo):  # label sits just below the lower error-bar cap
        ax.annotate(f"{y:.2f}", (x, y - l), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=8, color=color)
fig.tight_layout()
out = args.outdir / "lds_vs_size.pdf"
args.outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(f"wrote {out}")
