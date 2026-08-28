#!/usr/bin/env python3
"""Render the proponent-filter scaling curve from experiments.csv with matplotlib.

    python scripts/scaling_plot_mpl.py                      # write figures/filter_scaling.png
    python scripts/scaling_plot_mpl.py --out other.png

Run selection (bs256 rows, the lr 2e-4 re-run preferred at muon 4k) mirrors
scripts/scaling_plot.py; regenerate whenever experiments.csv is rebuilt.
"""
import argparse
import csv
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=pathlib.Path, default=ROOT / "figures" / "filter_scaling.png")
args = ap.parse_args()

# The x dodge separates coincident error bars; multiplicative because x is log.
SERIES = [("AdamW", "#2a78d6", 0.98, ("plan_adam_eps1e17_", "sm_adamw_eps1e17_")),
          ("Muon", "#eb6834", 1.02, ("plan_muon_eps1e17_", "sm_muon_eps1e17_"))]
NS = [4000, 8000, 16000, 32000, 64000, 128000]
# Tokens seen in training: 2 epochs over N docs of 512 tokens each.
tokens = lambda n: 2 * n * 512
PREFER = ("plan_muon_eps1e17_4k_bs256_lr2e-4",)

rows = list(csv.DictReader(open(ROOT / "experiments.csv")))


def pick(prefixes, n):
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


fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)

for name, color, dodge, prefixes in SERIES:
    ns, deltas, lo_err, hi_err = [], [], [], []
    missing = []
    for n in NS:
        r = pick(prefixes, n)
        if r is None or not (r.get("filter_ekfac_delta") or "").strip():
            missing.append(n)
            continue
        d = float(r["filter_ekfac_delta"])
        ns.append(tokens(n))
        deltas.append(d)
        lo_err.append(d - float(r["filter_ekfac_lo"]))
        hi_err.append(float(r["filter_ekfac_hi"]) - d)
    xs = [n * dodge for n in ns]
    ax.errorbar(xs, deltas, yerr=[lo_err, hi_err], color=color, label=name,
                marker="o", markersize=5, linewidth=2, capsize=3, capthick=1.2)
    for n in missing:
        ax.annotate(f"{name}\nretraining", (tokens(n), 0), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7, color="#9a988f")

ax.set_xscale("log", base=2)
ax.set_xticks([tokens(n) for n in NS], [f"{tokens(n) / 1e6:.0f}M" for n in NS])
ax.minorticks_off()
ax.set_xlabel("Number of training tokens")
ax.set_ylabel("Change in query loss")
ax.grid(color="#e6e5e0", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
          ncols=len(SERIES), borderaxespad=0)
ax.margins(x=0.09)

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(args.out)
print(f"wrote {args.out}")
