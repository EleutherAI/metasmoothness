#!/usr/bin/env python3
"""Appendix table of every power-law fit quoted in the Results, with 95% CIs (query bootstrap) and R^2.

    python scripts/fits_table.py   # -> tables/fits.tex (complete table environment; needs booktabs)

Fits (all least squares in log space on the mean QLD over the 20 queries; CIs from 2,000 resamples of the queries):
  GPT-2, held-out EK-FAC (Figure 1): 1% series as a law in k (k = 0.01N; identical exponent to the law in N), top-40
    series in N (4k point = 1% run), joint law QLD = a k^alpha N^beta over the 1% and top-40 cells, and the same with
    beta fixed to 0.
  Qwen 1.5B, EK-FAC (Figure 2, series as drawn incl. qwen_scaling_plot aliases): 1%/5%/10% and top-40/200/400 in N,
    and the joint law over all cells.
Extrapolation column (GPT-2 1% series only): fit on N = 8k-256k, predict 512k; fit on 4k-256k, predict 512k.
"""
import csv
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
E = pathlib.Path("/mnt/ssd-2/lucia/paper_runs/experiments")
OUT = ROOT / "tables" / "fits.tex"
rng = np.random.default_rng(0)
NBOOT = 2000
ADAM = {4000: "plan_adam_eps1e17_4k_bs256", 8000: "plan_adam_eps1e17_8k_bs256", 16000: "sm_adamw_eps1e17_16k_bs256",
        32000: "plan_adam_eps1e17_32k_bs256", 64000: "plan_adam_eps1e17_64k_bs256", 128000: "plan_adam_eps1e17_128k_bs256",
        256000: "plan_adam_eps1e17_256k_bs256", 512000: "plan_adam_eps1e17_512k_bs256"}
QWEN = {4000: "qwen15b_4k_bs256", 8000: "qwen15b_8k_bs256", 16000: "qwen15b_16k_bs256", 32000: "qwen15b_32k_bs256", 64000: "qwen15b_64k_bs256"}
QWEN_ALIAS = {(4000, "filter_top200_ekfac"): "filter_prop5pct_ekfac", (4000, "filter_top400_ekfac"): "filter_prop10pct_ekfac",
              (8000, "filter_top400_ekfac"): "filter_prop5pct_ekfac"}
CAPTION_TMPL = (r"Power-law fits over QLD in Section~\ref{sec:results}, for the GPT-2 series of Figure~\ref{fig:filter_scaling} "
                r"and the Qwen series of Figure~\ref{fig:qwen}. Fits are least squares in log space on the mean QLD over $Q=20$ "
                r"queries, with 95\% confidence intervals from bootstrapping over queries. Single-variable fits are "
                r"$\mathrm{QLD} \propto x^{\alpha}$ in the stated variable; joint fits are $\mathrm{QLD} = a k^{\alpha} N^{\beta}$ "
                r"with $k$ the number of documents removed and $N$ the training-set size. Fitting the GPT-2 top-1\% series on "
                r"8,000--256,000 documents over-predicts the QLD at 512,000 documents by EX8; on 4,000--256,000 documents by EX4.")
LABEL = "tab:fits"


def perq(run, sub):
    p = E / run / sub / "filter_summary.csv"
    if not p.is_file():
        return None
    rows = sorted(csv.DictReader(open(p)), key=lambda r: int(float(r["query"])))
    d = np.array([float(r["filter_change"]) - float(r["random_mean"]) for r in rows])
    return d if len(d) == 20 else None


def fit1(xs, ys):
    x, y = np.log(xs), np.log(ys); A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None); pred = A @ np.array([a, b])
    return a, b, 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def boot1(xs, data):
    out = []
    for _ in range(NBOOT):
        idx = rng.integers(0, 20, 20); yb = np.array([d[idx].mean() for d in data])
        if (yb > 0).all():
            out.append(fit1(xs, yb)[0])
    return np.percentile(out, [2.5, 97.5])


def fitj(pts, ys, full):
    X = np.array([[math.log(k), math.log(n), 1.0] if full else [math.log(k), 1.0] for n, k, _ in pts]); y = np.log(ys)
    c, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ c
    return c, 1 - float((res ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def bootj(pts):
    out = []
    for _ in range(NBOOT):
        idx = rng.integers(0, 20, 20); yb = np.array([d[idx].mean() for *_, d in pts])
        if (yb > 0).all():
            out.append(fitj(pts, yb, True)[0])
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def ci(v, lo, hi, d=2):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


rows = []
# ---- GPT-2 held-out
ns, data = zip(*[(n, perq(r, "filter_proponents_ekfac_heldout")) for n, r in ADAM.items()]); ns = np.array(ns); ys = np.array([d.mean() for d in data])
ks = 0.01 * ns
a, _, r2 = fit1(ks, ys); lo, hi = boot1(ks, data)
def extrap(mask):
    a_, b_, _ = fit1(ks[mask], ys[mask]); return 100 * (math.exp(b_ + a_ * math.log(ks[-1])) / ys[-1] - 1)
CAPTION = CAPTION_TMPL.replace("EX4", f"{abs(extrap(ns <= 256000)):.0f}\\%").replace("EX8", f"{abs(extrap((ns >= 8000) & (ns <= 256000))):.0f}\\%")
rows.append(("GPT-2", "Top 1\\%", "$k$", ci(a, lo, hi), "--", f"{r2:.2f}", str(len(ns))))
ns40, d40 = zip(*[(n, data[0] if n == 4000 else perq(r, "filter_top40_ekfac_heldout")) for n, r in ADAM.items()]); ns40 = np.array(ns40)
y40 = np.array([d.mean() for d in d40]); a40, _, r40 = fit1(ns40, y40); lo40, hi40 = boot1(ns40, d40)
rows.append(("", "Top 40", "$N$", ci(a40, lo40, hi40), "--", f"{r40:.2f}", str(len(ns40))))
pts = [(n, int(round(0.01 * n)), d) for n, d in zip(ns, data)] + [(n, 40, d) for n, d in zip(ns40, d40) if n != 4000]
yj = np.array([d.mean() for *_, d in pts]); cf, r2f = fitj(pts, yj, True); loj, hij = bootj(pts); cr, r2r = fitj(pts, yj, False)
rows.append(("", "Joint, 1\\% and top 40", "$k, N$", ci(cf[0], loj[0], hij[0]), ci(cf[1], loj[1], hij[1]), f"{r2f:.2f}", str(len(pts))))
rows.append(("", "Joint, $\\beta = 0$", "$k$", f"{cr[0]:.2f}", "0", f"{r2r:.2f}", str(len(pts))))
# ---- Qwen
qpts = []
for lab, sub, k_of in (("Top 1\\%", "filter_proponents_ekfac", lambda n: int(round(0.01 * n))), ("Top 5\\%", "filter_prop5pct_ekfac", lambda n: int(round(0.05 * n))),
                       ("Top 10\\%", "filter_prop10pct_ekfac", lambda n: int(round(0.1 * n))), ("Top 40", "filter_top40_ekfac", lambda n: 40),
                       ("Top 200", "filter_top200_ekfac", lambda n: 200), ("Top 400", "filter_top400_ekfac", lambda n: 400)):
    pts_ = [(n, perq(r, QWEN_ALIAS.get((n, sub), sub))) for n, r in QWEN.items()]; pts_ = [(n, d) for n, d in pts_ if d is not None]
    xs = np.array([n for n, _ in pts_]); ds = [d for _, d in pts_]; yq = np.array([d.mean() for d in ds])
    aq, _, rq = fit1(xs, yq); lq, hq = boot1(xs, ds)
    rows.append(("Qwen 1.5B" if lab == "Top 1\\%" else "", lab, "$N$", ci(aq, lq, hq), "--", f"{rq:.2f}", str(len(xs))))
    for n, d in pts_:
        if not any(nn == n and kk == k_of(n) for nn, kk, _ in qpts):
            qpts.append((n, k_of(n), d))
yq = np.array([d.mean() for *_, d in qpts]); cq, r2q = fitj(qpts, yq, True); loq, hiq = bootj(qpts)
rows.append(("", "Joint, all series", "$k, N$", ci(cq[0], loq[0], hiq[0]), ci(cq[1], loq[1], hiq[1]), f"{r2q:.2f}", str(len(qpts))))

lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + CAPTION + "}", r"\label{" + LABEL + "}", r"\small",
         r"\begin{tabular}{@{}llcllcr@{}}", r"\toprule",
         r"Model & Series & Variable & $\alpha$ [95\% CI] & $\beta$ [95\% CI] & $R^2$ & Points \\", r"\midrule"]
for i, r in enumerate(rows):
    if r[0] and i:
        lines.append(r"\addlinespace")
    lines.append(" & ".join(r) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
OUT.parent.mkdir(exist_ok=True); OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"\nwrote {OUT}")
