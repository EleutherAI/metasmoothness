#!/usr/bin/env python3
"""Appendix table: GPT-2 held-out EK-FAC QLD by training-set size for three filtering amounts (top 1%, top 40, top 400).

    python scripts/heldout_fixedk_table.py   # -> tables/heldout_fixedk.tex (complete table environment; needs booktabs)

Inputs: <run>/filter_proponents_ekfac_heldout/filter_summary.csv and <run>/filter_top40_ekfac_heldout/filter_summary.csv
(merged 20-query summaries; at 4k the top-40 run is the 1% run), and the unmerged 2-query top-400 shards
<run>/filter_top400_ekfac_heldout_q{a}_{a+2}/filter_summary.csv (in-job random controls). Cells are the mean over the 20
held-out queries of (filter_change - random_mean) with 95% percentile-bootstrap CIs (10k resamples); '--' when a cell is
incomplete. Rows in absolute_losses.EXCLUDE_RUNS are skipped.
"""
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import absolute_losses as absl  # noqa: E402
    EXCL = getattr(absl, "EXCLUDE_RUNS", set())
except Exception:  # matplotlib-less environments
    EXCL = {"plan_adam_eps1e17_16k_scale0.5", "plan_adam_eps1e17_16k_scale0.25"}

E = pathlib.Path("/mnt/ssd-2/lucia/paper_runs/experiments")
OUT = ROOT / "tables" / "heldout_fixedk.tex"
rng = np.random.default_rng(0)
BOOT = 10000
ADAM = {4000: "plan_adam_eps1e17_4k_bs256", 8000: "plan_adam_eps1e17_8k_bs256", 16000: "sm_adamw_eps1e17_16k_bs256",
        32000: "plan_adam_eps1e17_32k_bs256", 64000: "plan_adam_eps1e17_64k_bs256", 128000: "plan_adam_eps1e17_128k_bs256",
        256000: "plan_adam_eps1e17_256k_bs256", 512000: "plan_adam_eps1e17_512k_bs256"}
CAPTION = (r"Held-out EK-FAC filtering efficacy for GPT-2 by training-set size and filtering amount: QLD (nats) when removing "
           r"the top 1\%, the top 40, or the top 400 proponents of each query. Mean over $Q=20$ held-out queries with 95\% "
           r"bootstrap confidence intervals. At 4,000 documents the top 1\% is the top 40. The top-400 runs use three random "
           r"controls per pair of queries; the others use the shared random-control retrains of Figure~\ref{fig:filter_scaling}.")
LABEL = "tab:fixedk"


def perq(path):
    if not path.is_file():
        return None
    rows = sorted(csv.DictReader(open(path)), key=lambda r: int(float(r["query"])))
    d = np.array([float(r["filter_change"]) - float(r["random_mean"]) for r in rows])
    return d if len(d) == 20 else None


def perq_shards(run, prefix, width=2):
    out = {}
    for a in range(0, 20, width):
        p = E / run / f"{prefix}_q{a}_{a+width}" / "filter_summary.csv"
        if p.is_file():
            for i, r in enumerate(csv.DictReader(open(p))):
                out[a + i] = float(r["filter_change"]) - float(r["random_mean"])
    return np.array([out[q] for q in range(20)]) if len(out) == 20 else None


def cell(d, digits=3):
    if d is None:
        return "--"
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{d.mean():.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + CAPTION + "}", r"\label{" + LABEL + "}", r"\small",
         r"\begin{tabular}{@{}rrccc@{}}", r"\toprule",
         r"Documents & Tokens & \shortstack{QLD, top 1\% removed\\(nats)} & \shortstack{QLD, top 40 removed\\(nats)} & \shortstack{QLD, top 400 removed\\(nats)} \\",
         r"\midrule"]
for n, run in ADAM.items():
    if run in EXCL:
        continue
    p1 = perq(E / run / "filter_proponents_ekfac_heldout" / "filter_summary.csv")
    p40 = p1 if n == 4000 else perq(E / run / "filter_top40_ekfac_heldout" / "filter_summary.csv")
    p400 = perq_shards(run, "filter_top400_ekfac_heldout")
    lines.append(f"{n//1000}k & {round(n*1024/1e6)}M & {cell(p1)} & {cell(p40)} & {cell(p400)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwrote {OUT}")
