"""Per-query Spearman: predicted attribution mass removed vs measured loss change.

filter_deltas.py reports a MEAN loss change per run -- how much damage removing
the top 1% does on average. It says nothing about whether the scorer knows WHICH
queries it hurt most. This does.

For each query q the proponent filter removes that query's own top
subset_fraction of documents (rank=1, n_removed = 1% of the corpus), retrains,
and measures loss_change(q). The scorer's own prediction of how much that
removal should hurt is the attribution mass it removed:

    score_sum(q) = sum of q's top-k attribution scores

Spearman over queries between score_sum and loss_change asks: does the scorer
rank queries by damage correctly? A scorer can have a large mean delta (it finds
influential data) while ranking queries no better than chance.

This is NOT LDS. LDS fixes a query and varies the subset, then averages the
per-query Spearman. This fixes the selection rule and varies the query. They can
disagree, and where they do is informative rather than contradictory.

CI is a 10k bootstrap resampling QUERIES, seed 0. Writes
data/filter_query_spearman.csv.
"""
import csv
import glob
import json
import os

import numpy as np
import yaml
from scipy.stats import spearmanr

ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments",
         "/mnt/ssd-1/lucia/paper_runs/experiments"]
OUT = "/mnt/ssd-2/lucia/metasmoothness/data/filter_query_spearman.csv"
rng = np.random.default_rng(0)


def load_scores(scores_dir):
    """Return [n_docs, n_queries] score matrix, or None if absent."""
    info_p = os.path.join(scores_dir, "info.json")
    bin_p = os.path.join(scores_dir, "scores.bin")
    if not (os.path.isfile(info_p) and os.path.isfile(bin_p)):
        return None
    info = json.load(open(info_p))
    d = info["dtype"]
    dt = np.dtype({"names": d["names"], "formats": d["formats"],
                   "offsets": d["offsets"], "itemsize": d["itemsize"]})
    a = np.memmap(bin_p, dtype=dt, mode="r")
    nq = int(info["num_scores"])
    return np.stack(
        [np.asarray(a["score_%d" % i], dtype=np.float64) for i in range(nq)], axis=1)


def find(node, key, want_str=True):
    """First value for `key` anywhere in a nested config."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                if want_str and isinstance(v, str):
                    return v
                if not want_str and isinstance(v, (int, float)):
                    return float(v)
            r = find(v, key, want_str)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = find(v, key, want_str)
            if r is not None:
                return r
    return None


def boot_spearman(x, y):
    rho = spearmanr(x, y).statistic
    n = len(x)
    if n < 3 or not np.isfinite(rho):
        return rho, float("nan"), float("nan")
    idx = rng.integers(0, n, size=(10_000, n))
    bs = []
    for row in idx:
        # A resample can be constant in x or y; spearmanr returns nan there.
        # Drop those rather than letting them poison the percentiles.
        r = spearmanr(x[row], y[row]).statistic
        if np.isfinite(r):
            bs.append(r)
    if len(bs) < 100:
        return rho, float("nan"), float("nan")
    return rho, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


rows = []
seen = set()
for root in ROOTS:
    for prop in sorted(glob.glob("%s/*/filter_*_*/filter_proponents.csv" % root)):
        d = os.path.dirname(prop)
        run = os.path.basename(os.path.dirname(d))
        src = os.path.basename(d).split("_")[-1]
        if (run, src) in seen:
            continue
        seen.add((run, src))
        cfg_p = os.path.join(d, "config.yaml")
        if not os.path.isfile(cfg_p):
            continue
        cfg = yaml.safe_load(open(cfg_p))
        sdir = find(cfg, "scores")
        frac = find(cfg, "subset_fraction", want_str=False)
        if not sdir or frac is None:
            rows.append({"run": run, "scorer": src, "status": "no scores/frac in config"})
            continue
        S = load_scores(sdir)
        if S is None:
            rows.append({"run": run, "scorer": src, "status": "scores missing: %s" % sdir})
            continue
        pq = list(csv.DictReader(open(prop)))
        k = int(round(frac * S.shape[0]))
        xs, ys, bad = [], [], 0
        for r in pq:
            q = int(r["query"])
            if q >= S.shape[1]:
                bad += 1
                continue
            # Reproduce the filter's own selection: query q's top-k documents.
            top = np.partition(S[:, q], -k)[-k:]
            xs.append(float(top.sum()))
            ys.append(float(r["loss_change"]))
            if int(r["n_removed"]) != k:
                bad += 1
        if len(xs) < 3:
            rows.append({"run": run, "scorer": src, "status": "too few queries"})
            continue
        rho, lo, hi = boot_spearman(np.array(xs), np.array(ys))
        rows.append({"run": run, "scorer": src, "n_queries": len(xs), "k_removed": k,
                     "rho": "%.4f" % rho, "lo": "%.4f" % lo, "hi": "%.4f" % hi,
                     "status": ("n_removed mismatch on %d queries" % bad) if bad else "ok"})

cols = ["run", "scorer", "n_queries", "k_removed", "rho", "lo", "hi", "status"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})

print("%-40s %-6s %3s %8s %18s  %s" % ("run", "scorer", "n", "rho", "95% CI", "status"))
for r in sorted(rows, key=lambda r: (r["scorer"], r["run"])):
    if "rho" in r:
        print("%-40s %-6s %3d %8s [%7s,%7s]  %s" % (
            r["run"], r["scorer"], r["n_queries"], r["rho"], r["lo"], r["hi"], r["status"]))
    else:
        print("%-40s %-6s %3s %8s %18s  %s" % (
            r["run"], r["scorer"], "", "", "", r["status"]))

fin = [r for r in rows if "rho" in r]
for sc in ("magic", "ekfac"):
    v = [float(r["rho"]) for r in fin if r["scorer"] == sc and np.isfinite(float(r["rho"]))]
    if v:
        print("\n%s: %d runs, median rho %+.3f, mean %+.3f, %d/%d positive"
              % (sc, len(v), float(np.median(v)), float(np.mean(v)),
                 sum(1 for x in v if x > 0), len(v)))
print("\nwrote %s" % OUT)
