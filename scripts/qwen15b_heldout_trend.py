"""Qwen2.5-1.5B: held-out loss of the selected-LR base model at every corpus rung.

    python scripts/qwen15b_heldout_trend.py    ->  figures/qwen15b_heldout_trend.pdf

y is the held-out cross-entropy on heldout_4k_qwen_v2 recorded in
data/qwen15b_heldout_v2.csv; each point is annotated with the learning rate the
rung's tuning sweep selected (lowest held-out loss among the measured rows of its
qwen_tuning.csv sweep group). The sweep's own minimum is printed beside the
recorded value so a re-evaluated rung cannot drift from its label unnoticed.

This is not a QLD figure (no filter is involved), so it has no *_absolute companion.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.environ.get("FIGURES_DIR") or os.path.join(ROOT, "figures"),
                   "qwen15b_heldout_trend.pdf")
BLUE = "#2a78d6"

heldout = {}
for r in csv.DictReader(open(os.path.join(ROOT, "data", "qwen15b_heldout_v2.csv"))):
    heldout[int(float(r["n_docs"]))] = float(r["heldout_ce"])

best = {}  # n_docs -> (loss, lr) of the sweep minimum
for r in csv.DictReader(open(os.path.join(ROOT, "qwen_tuning.csv"))):
    if r.get("status") != "measured" or not (r.get("heldout_loss") or "").strip():
        continue
    n, v = int(float(r["n_docs"])), float(r["heldout_loss"])
    if n not in best or v < best[n][0]:
        best[n] = (v, r["lr"])

ns = sorted(heldout)
ys = [heldout[n] for n in ns]
fig, ax = plt.subplots(figsize=(6, 3.5), dpi=200)
ax.plot(ns, ys, color=BLUE, marker="o", markersize=6, linewidth=2)
for n, y in zip(ns, ys):
    lr = best.get(n, (None, "?"))[1]
    ax.annotate(f"{float(lr):g}" if lr != "?" else "?", (n, y), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=8)
    swept = best.get(n, (float("nan"),))[0]
    print(f"{n // 1000:>4}k  heldout={y:.4f}  sweep min={swept:.4f} @ lr={lr}")
ax.set_xscale("log", base=2)
ax.set_xticks(ns, [f"{n // 1000}k" for n in ns])
ax.minorticks_off()
ax.set_xlabel("Qwen train documents")
ax.set_ylabel("Heldout CE loss")
ax.set_title("Qwen2.5-1.5B selected LR heldout trend", fontsize=10)
ax.grid(color="#e6e5e0", linewidth=0.8)
ax.set_axisbelow(True)
ax.margins(x=0.06, y=0.15)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
print("wrote", OUT)
