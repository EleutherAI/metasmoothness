#!/usr/bin/env python3
"""EK-FAC vs MAGIC vs BM25 at 4k-32k documents: the first four points of the methods
figure (filter_method_appendix.pdf) as a LaTeX table (full table environment).

    python scripts/qld_methods_table.py        # -> tables/qld_methods_4k_32k.tex

The QLD numbers are exactly the ones the figure draws (scaling_plot_mpl.py, method
appendix): mean over the 20 queries of (filter_change - random_mean), with a
10k-sample bootstrap 95% CI over queries. EK-FAC and MAGIC top-1% come from
experiments.csv (filter_<m>_delta and its CI, as the figure's scaling_points);
BM25 top-1% and every top-40 value come from the merged filter summaries
(<run>/filter_{proponents,top40}_<m>/filter_summary.csv), as the figure's
summary_delta_ci. LDS: EK-FAC and MAGIC from experiments.csv (as the Adam/Muon
table); BM25 is not recorded there and is computed here with scripts/ekfac_lds.py
on <run>/bm25_scores against the row's 100-subset bank (BM25 is higher-is-better,
the sign convention ekfac_lds.py assumes). Below 64k the in-distribution query set
is absent from every training corpus, so these are out-of-distribution queries;
the 64k row is the swapped-corpus run (plan_adam_eps1e17_64k_bs256_qswap: the 22 rows
containing the queries replaced by unseen documents), all three methods from its merged
filter summaries; LDS is not computed at 64k (no 100-subset bank), shown as --.

Per rung the method with the largest top-1% QLD is set in bold. Emitted as a
complete table environment (caption/label included); needs booktabs.
"""
import csv
import math
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import absolute_losses as absl  # noqa: E402  (boot_ci: the figure's bootstrap)

OUT = pathlib.Path(os.environ.get("TABLES_DIR") or ROOT / "tables") / "qld_methods_4k_32k.tex"
ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments", "/mnt/ssd-1/lucia/paper_runs/experiments"]
NS = [4000, 8000, 16000, 32000, 64000]
RUNS = {4000: "plan_adam_eps1e17_4k_bs256", 8000: "plan_adam_eps1e17_8k_bs256",
        16000: "sm_adamw_eps1e17_16k_bs256", 32000: "plan_adam_eps1e17_32k_bs256",
        64000: "plan_adam_eps1e17_64k_bs256_qswap"}  # 64k: swapped-corpus row, values from merged summaries, no LDS  # TOP40_ROWS in scaling_plot_mpl.py
METHODS = [("MAGIC", "magic"), ("EK-FAC", "ekfac"), ("BM25", "bm25")]
tokens = lambda n: 2 * n * 512
BOOT = 10000
CAPTION = (r"Mean effect of proponent filtering on query loss for BM25, EK-FAC, and MAGIC, with the linear "
           r"datamodelling score (LDS) of each method's scores on the same runs. Values are means with 95\% "
           r"confidence intervals, bootstrapped over $Q=20$ held-out queries. At 64,000 documents the training corpus has the "
           r"rows containing the queries replaced by unseen documents, and LDS is not computed.")
if absl.STAT == "median":  # QLD_STAT=median: same table with the per-query median (make_figures.py --stat median)
    CAPTION = CAPTION.replace("Mean effect", "Median effect").replace("Values are means", "Values are medians")
LABEL = "tab:proponent-filtering"

rows = list(csv.DictReader(open(ROOT / "experiments.csv", newline="")))
by_id = {r["run_id"]: r for r in rows}


def run_root(run):
    return next((x for x in ROOTS if os.path.isdir(os.path.join(x, run))), None)


def summary_delta_ci(run, subdir, min_rows=20):
    """Mirror of scaling_plot_mpl.summary_delta_ci(subtract_random=True)."""
    root = run_root(run)
    path = os.path.join(root, run, subdir, "filter_summary.csv") if root else None
    if not (path and os.path.isfile(path)):
        return None
    d = []
    for row in csv.DictReader(open(path)):
        val = float(row["filter_change"]) - float(row["random_mean"])
        if math.isfinite(val):
            d.append(val)
    if len(d) < min_rows:
        return None
    return absl.boot_ci(d, BOOT)


def delta_ci(run, method):
    """Mirror of scaling_plot_mpl.delta_ci: experiments.csv columns."""
    r = by_id.get(run)
    if r is None or not (r.get(f"filter_{method}_delta") or "").strip():
        return None
    if absl.STAT != "mean":  # the csv holds only the mean: recompute, as the figure does
        p = absl.qld_from_summary(run, f"filter_proponents_{method}")
        if p is not None:
            return p
        print(f"  warning: {run}/filter_proponents_{method}: no complete summary, "
              f"{absl.STAT} cell falls back to the experiments.csv mean")
    d = float(r[f"filter_{method}_delta"])
    return (d, d - float(r[f"filter_{method}_lo"]), float(r[f"filter_{method}_hi"]) - d)


def lds(run, method):
    """(value, lo, hi) or None."""
    r = by_id.get(run)
    if method in ("ekfac", "magic"):
        v = (r.get(f"{method}_lds") or "").strip()
        return (float(v), float(r[f"{method}_ci_lo"]), float(r[f"{method}_ci_hi"])) if v else None
    root = run_root(run)
    scores = os.path.join(root, run, "bm25_scores") if root else None
    if not (scores and os.path.isfile(os.path.join(scores, "scores.bin"))):
        return None
    out = subprocess.run([sys.executable, "-P", str(ROOT / "scripts" / "ekfac_lds.py"), "--scores", scores,
                          "--bank", os.path.join(root, run), "--n-boot", str(BOOT)],
                         capture_output=True, text=True, cwd="/tmp").stdout
    m = re.search(r"ekfac_lds\s+(-?[\d.]+)\s+\[(-?[\d.]+),\s*(-?[\d.]+)\]", out)
    return (float(m[1]), float(m[2]), float(m[3])) if m else None


def fmt(p, digits):
    """'value [lo, hi]' from (value, -delta, +delta) or '--'."""
    if p is None:
        return "--"
    d, lo, hi = p
    return f"{d:.{digits}f} [{d - lo:.{digits}f}, {d + hi:.{digits}f}]"


def fmt_abs(p, digits):
    """'value [lo, hi]' from (value, lo, hi) or '--'."""
    if p is None:
        return "--"
    return f"{p[0]:.{digits}f} [{p[1]:.{digits}f}, {p[2]:.{digits}f}]"


def bold(s):
    return r"\textbf{" + s + "}"


lines = [
    r"% generated by scripts/qld_methods_table.py from experiments.csv, the merged filter summaries and the LDS banks -- do not edit by hand",
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{" + CAPTION + "}",
    r"\label{" + LABEL + "}",
    r"\small",
    r"\begin{tabular}{@{}rrlccc@{}}",
    r"\toprule",
    r"Documents & Tokens & Method &",
    r"\shortstack{QLD, top 1\% removed\\(nats)} &",
    r"\shortstack{QLD, top 40 removed\\(nats)} &",
    r"\shortstack{LDS\\(Spearman)} \\",
    r"\midrule",
]
missing = []
for n in NS:
    run = RUNS[n]
    vals = {}
    for name, m in METHODS:
        pct = delta_ci(run, m) if (m in ("ekfac", "magic") and n < 64000) else summary_delta_ci(run, f"filter_proponents_{m}")
        top = summary_delta_ci(run, f"filter_top40_{m}")
        if n == 4000 and top is None:  # 40 docs is 1% of 4k: the figure reuses the 1% run
            top = pct
        vals[m] = (pct, top, lds(run, m) if n < 64000 else None)
        if pct is None or top is None or (vals[m][2] is None and n < 64000):
            missing.append((n, name, pct is None, top is None, vals[m][2] is None))
    best = max((m for m in vals if vals[m][0]), key=lambda m: vals[m][0][0], default=None)
    for i, (name, m) in enumerate(METHODS):
        pct, top, l = vals[m]
        cells = [name, fmt(pct, 3), fmt(top, 3), fmt_abs(l, 2)]
        if m == best:
            cells = [bold(c) for c in cells]
        docs = f"{n // 1000}k" if i == 0 else ""
        tok = f"{tokens(n) / 1e6:.0f}M" if i == 0 else ""
        lines.append(" & ".join([docs, tok, *cells]) + r" \\")
        print(f"  {n // 1000:>3}k {lines[-1]}")
    if n != NS[-1]:
        lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT}")
if missing:
    raise SystemExit(f"missing values: {missing}")
