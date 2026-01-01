"""Quantify query-to-query variation in QLD (query loss difference).

Three questions, one table each:
  (1) how much larger are in-distribution QLDs than held-out ones, per (N, rule, method);
  (2) how much do QLDs vary between queries drawn from the SAME set, relative to the
      retrain-noise floor given by the random-removal controls (random_sd);
  (3) is that spread a stable property of the query -- Spearman rank agreement of the
      per-query QLD across attribution methods and across training scales.

QLD = filter_change - random_mean, per query, from <run>/<filter>[_heldout]/filter_summary.csv.
Prints tables; --csv writes the per-query long table for downstream use.
"""
import argparse, csv, glob, itertools, os
import numpy as np
from scipy import stats

E = "/data/anon/paper_runs/experiments"
RUNS = {4000: "plan_adam_eps1e17_4k_bs256", 8000: "plan_adam_eps1e17_8k_bs256",
        16000: "sm_adamw_eps1e17_16k_bs256", 32000: "plan_adam_eps1e17_32k_bs256",
        64000: "plan_adam_eps1e17_64k_bs256", 128000: "plan_adam_eps1e17_128k_bs256",
        256000: "plan_adam_eps1e17_256k_bs256", 512000: "plan_adam_eps1e17_512k_bs256"}
RULES = {"filter_proponents": "top1%", "filter_top40": "top40"}
METHODS = ("ekfac", "bm25", "magic", "jina")
NQ = 20
# query_20 chunks re-appear in train from 64k up (README 2026-09-04); below that the
# query set is verified disjoint from train.
OVERLAP_FROM = 64000


def read_summary(path):
    """-> {query_idx: (qld, control_sd, n_controls)}"""
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        try:
            q, a, b = int(r["query"]), float(r["filter_change"]), float(r["random_mean"])
            sd, n = float(r["random_sd"]), int(r["random_n"])
        except (KeyError, ValueError):
            continue
        if a == a and b == b:
            out[q] = (a - b, sd, n)
    return out


def cell(run, rule, method, heldout):
    """Merged dir if complete, else pooled shards. Shard query ids are already global."""
    base = f"{E}/{run}/{rule}_{method}" + ("_heldout" if heldout else "")
    d = read_summary(base + "/filter_summary.csv")
    if len(d) == NQ:
        return d
    for sh in sorted(glob.glob(base + "_q*_*")):
        if os.path.isdir(sh) and not any(t in sh for t in (".nan", ".invalid", ".partial", ".merge_tmp")):
            for q, v in read_summary(sh + "/filter_summary.csv").items():
                d.setdefault(q, v)
    return d


def collect():
    rows = []
    for n, run in RUNS.items():
        for rule in RULES:
            for method in METHODS:
                for heldout in (False, True):
                    d = cell(run, rule, method, heldout)
                    if len(d) < NQ:
                        continue
                    for q, (qld, sd, nc) in sorted(d.items()):
                        rows.append(dict(n=n, rule=RULES[rule], method=method,
                                         qset="held-out" if heldout else "in-dist",
                                         query=q, qld=qld, ctrl_sd=sd, ctrl_n=nc))
    return rows


def by_cell(rows):
    out = {}
    for r in rows:
        out.setdefault((r["n"], r["rule"], r["method"], r["qset"]), []).append(r)
    return {k: sorted(v, key=lambda r: r["query"]) for k, v in out.items()}


def stats_for(rs):
    x = np.array([r["qld"] for r in rs])
    noise = float(np.sqrt(np.mean(np.square([r["ctrl_sd"] for r in rs]))))
    sd = float(x.std(ddof=1))
    p10, p90 = np.percentile(x, [10, 90])
    return dict(n=len(x), mean=float(x.mean()), sd=sd, cv=sd / float(x.mean()),
                lo=float(x.min()), hi=float(x.max()),
                spread=float(p90 / p10) if p10 > 0 else float("nan"),
                noise=noise, signal=float(np.sqrt(max(sd ** 2 - noise ** 2, 0.0))),
                frac=1 - min(noise ** 2 / sd ** 2, 1.0),
                ctrl_n=int(np.median([r["ctrl_n"] for r in rs])),
                ci=1.96 * sd / np.sqrt(len(x)))


def t1(cells):
    print("\n(1) IN-DISTRIBUTION vs HELD-OUT QUERIES  (QLD mean +- 95% CI over 20 queries)")
    print(f"{'N':>7} {'rule':>6} {'meth':>5} {'in-dist':>17} {'held-out':>17} "
          f"{'ratio':>6} {'Welch p':>9}")
    gaps = []
    for (n, rule, meth, qs), rs in sorted(cells.items()):
        if qs != "in-dist" or (n, rule, meth, "held-out") not in cells:
            continue
        ho = cells[(n, rule, meth, "held-out")]
        a, b = stats_for(rs), stats_for(ho)
        p = stats.ttest_ind([r["qld"] for r in rs], [r["qld"] for r in ho], equal_var=False).pvalue
        ratio = a["mean"] / b["mean"]
        gaps.append((n, rule, meth, ratio, a["mean"] - b["mean"], p))
        print(f"{n//1000:>6}k {rule:>6} {meth:>5} {a['mean']:>9.4f}+-{a['ci']:.4f} "
              f"{b['mean']:>9.4f}+-{b['ci']:.4f} {ratio:>6.2f} {p:>9.1e}")
    for lab, sel in (("N < 64k (query set disjoint from train)", lambda g: g[0] < OVERLAP_FROM),
                     ("N >= 64k (query chunks re-appear in train)", lambda g: g[0] >= OVERLAP_FROM)):
        sub = [g for g in gaps if sel(g)]
        if not sub:
            continue
        r = np.array([g[3] for g in sub])
        print(f"  {lab}: {len(sub)} cells, ratio median {np.median(r):.2f} "
              f"[{r.min():.2f}, {r.max():.2f}], geometric mean {np.exp(np.log(r).mean()):.2f}")
    if len(gaps) > 1:
        lo = [g[3] for g in gaps if g[0] < OVERLAP_FROM]
        hi = [g[3] for g in gaps if g[0] >= OVERLAP_FROM]
        if lo and hi:
            p = stats.mannwhitneyu(hi, lo, alternative="greater").pvalue
            print(f"  ratio higher at N>=64k than below: Mann-Whitney p={p:.3f}")
    return gaps


def t2(cells):
    print("\n(2) QUERY-TO-QUERY SPREAD WITHIN ONE QUERY SET")
    print("    noise = RMS of the random-removal control SD (per-retrain noise floor);")
    print("    p90/p10 = spread of the middle 80% of queries; ctrl = control retrains per query")
    print(f"{'N':>7} {'rule':>6} {'meth':>5} {'qset':>9} {'mean':>8} {'sd':>8} {'CV':>6} "
          f"{'min':>8} {'max':>8} {'p90/p10':>8} {'noise':>8} {'sd/noise':>8} {'var_q':>6} {'ctrl':>5}")
    agg = {}
    for k, rs in sorted(cells.items()):
        st_ = stats_for(rs)
        n, rule, meth, qs = k
        agg.setdefault(qs, []).append(st_)
        print(f"{n//1000:>6}k {rule:>6} {meth:>5} {qs:>9} {st_['mean']:>8.4f} {st_['sd']:>8.4f} "
              f"{st_['cv']:>5.0%} {st_['lo']:>8.4f} {st_['hi']:>8.4f} {st_['spread']:>8.1f} "
              f"{st_['noise']:>8.4f} {st_['sd']/st_['noise']:>8.1f} {st_['frac']:>5.0%} {st_['ctrl_n']:>5}")
    for qs, ss in sorted(agg.items()):
        cv = np.array([s["cv"] for s in ss]); sp = np.array([s["spread"] for s in ss])
        print(f"  {qs:>9}: {len(ss)} cells, CV median {np.median(cv):.0%} "
              f"[{cv.min():.0%},{cv.max():.0%}]; p90/p10 median {np.nanmedian(sp):.1f}x")
    print("    noise floor, restricted to cells with 100 control retrains "
          "(a 3-retrain control SD is too noisy to serve as a floor):")
    for qs, ss in sorted(agg.items()):
        ss = [s for s in ss if s["ctrl_n"] >= 100]
        if not ss:
            continue
        rat = np.array([s["sd"] / s["noise"] for s in ss]); fr = np.array([s["frac"] for s in ss])
        print(f"  {qs:>9}: {len(ss)} cells, sd/noise median {np.median(rat):.0f}x "
              f"[{rat.min():.0f},{rat.max():.0f}]; query-attributable share of QLD variance "
              f"median {np.median(fr):.1%} (min {fr.min():.1%})")


def t3(cells):
    print("\n(3) IS THE SPREAD A PROPERTY OF THE QUERY?  Spearman rho of per-query QLD")
    print("  (a) across attribution methods, same run and query set")
    rhos = []
    groups = {}
    for (n, rule, meth, qs), rs in cells.items():
        groups.setdefault((n, rule, qs), {})[meth] = [r["qld"] for r in rs]
    for (n, rule, qs), d in sorted(groups.items()):
        for m1, m2 in itertools.combinations(sorted(d), 2):
            rho, p = stats.spearmanr(d[m1], d[m2])
            rhos.append((rho, n, rule))
            print(f"{n//1000:>6}k {rule:>6} {qs:>9} {m1:>5} vs {m2:>5}  rho={rho:>6.2f}  p={p:.3f}")
    if rhos:
        r = np.array([x[0] for x in rhos])
        print(f"  -> {len(r)} pairs, median rho {np.median(r):.2f}, "
              f"{(r > 0).mean():.0%} positive, mean {r.mean():.2f}")
        for lab, sel in (("N < 64k, query text absent from train", lambda t: t[1] < OVERLAP_FROM),
                         ("N >= 64k, query text present in train", lambda t: t[1] >= OVERLAP_FROM)):
            for rule in ("top40", "top1%"):
                v = np.array([t[0] for t in rhos if sel(t) and t[2] == rule])
                if len(v):
                    print(f"     {rule:>5}, {lab}: {len(v)} pairs, median rho {np.median(v):.2f} "
                          f"[{v.min():.2f}, {v.max():.2f}]")
    print("  (b) across training scales, same rule/method/query set (adjacent rungs)")
    rhos2 = []
    for rule in set(RULES.values()):
        for meth in METHODS:
            for qs in ("in-dist", "held-out"):
                ns = sorted(n for n in RUNS if (n, rule, meth, qs) in cells)
                for a, b in zip(ns, ns[1:]):
                    x = [r["qld"] for r in cells[(a, rule, meth, qs)]]
                    y = [r["qld"] for r in cells[(b, rule, meth, qs)]]
                    rho, p = stats.spearmanr(x, y)
                    rhos2.append(rho)
                    print(f"{a//1000:>4}k->{b//1000:<5}k {rule:>6} {meth:>5} {qs:>9}  "
                          f"rho={rho:>6.2f}  p={p:.3f}")
    if rhos2:
        r = np.array(rhos2)
        print(f"  -> {len(r)} pairs, median rho {np.median(r):.2f}, "
              f"{(r > 0).mean():.0%} positive, mean {r.mean():.2f}")


def boot_cv(x, b=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.choice(x, (b, len(x)), replace=True)
    return np.percentile(d.std(axis=1, ddof=1) / d.mean(axis=1), [2.5, 97.5])


def appendix(cells, fmt):
    """Full per-cell dispersion table -- every cell behind the main-body CV claim."""
    sep, end = (" & ", r" \\") if fmt == "latex" else (" | ", "")
    head = ["N", "rule", "method", "queries", "mean QLD", "SD", "CV", "CV 95% CI",
            "min", "max", "p90/p10", "ctrl SD", "ctrl n"]
    if fmt == "latex":
        print(r"\begin{tabular}{rllrrrrlrrrrr}\hline")
        print(sep.join(head) + end + r"\hline")
    else:
        print(sep.join(head)); print(sep.join("---" for _ in head))
    for (n, rule, meth, qs), rs in sorted(cells.items(), key=lambda kv: (kv[0][3], kv[0][0], kv[0][1], kv[0][2])):
        st_ = stats_for(rs)
        x = np.array([r["qld"] for r in rs])
        lo, hi = boot_cv(x)
        sp = "n/a" if st_["spread"] != st_["spread"] else f"{st_['spread']:.1f}"
        print(sep.join([f"{n // 1000}k", rule, meth, f"{qs} (n={st_['n']})",
                        f"{st_['mean']:.4f}", f"{st_['sd']:.4f}", f"{st_['cv']:.2f}",
                        f"[{lo:.2f}, {hi:.2f}]", f"{st_['lo']:.4f}", f"{st_['hi']:.4f}",
                        sp, f"{st_['noise']:.4f}", str(st_["ctrl_n"])]) + end)
    if fmt == "latex":
        print(r"\hline\end{tabular}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="write the per-query long table here")
    ap.add_argument("--appendix", choices=("markdown", "latex"),
                    help="print the full per-cell dispersion table and exit")
    a = ap.parse_args()
    rows = collect()
    cells = by_cell(rows)
    if a.appendix:
        appendix(cells, a.appendix)
        raise SystemExit
    print(f"{len(cells)} complete 20-query cells, {len(rows)} per-query QLDs")
    t1(cells); t2(cells); t3(cells)
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("\nwrote", a.csv)
