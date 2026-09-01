"""Can the filter delta at large scale be predicted from small-scale points?

Question: along the bs256 scaling row (4k..256k docs), fit on the FIRST
n_train scaling points and predict the observed top-1% removal delta at the
remaining, larger scales. Two families of predictor:

    scale-only  -- extrapolate the delta against log2(n_docs); the score sums
                   are never used. This is the "mean the first 3 points and
                   draw a line" baseline.
    score-based -- regress observed delta on the scorer's own prediction of
                   damage, the attribution mass removed (mean over queries of
                   sum of that query's top-1% scores). Predicting a new scale
                   needs only that scale's scores: forward passes, no retrains.

Per-query variants pool the 20 (score_sum, loss_change) pairs per point.

Data: score matrices at <run>/scores (MAGIC) and <run>/ekfac_scores/scores
(EK-FAC); observed per-query loss changes from
filter_proponents_<scorer>/filter_summary.csv. Rows with fewer than 20 queries
of retrain data are reported and EXCLUDED -- a partial row is not a noisy
version of a finished one. Needs no GPU. Writes data/filter_extrapolation.csv.
"""
import csv
import json
import os

import numpy as np

ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments",
         "/mnt/ssd-1/lucia/paper_runs/experiments"]
OUT = "/mnt/ssd-2/lucia/metasmoothness/data/filter_extrapolation.csv"

# bs256 scaling row, in scale order. (n_docs, run_id)
ROWS = {
    "adamw": [(4000, "plan_adam_eps1e17_4k_bs256"),
              (8000, "plan_adam_eps1e17_8k_bs256"),
              (16000, "sm_adamw_eps1e17_16k_bs256"),
              (32000, "plan_adam_eps1e17_32k_bs256"),
              (64000, "plan_adam_eps1e17_64k_bs256"),
              (128000, "plan_adam_eps1e17_128k_bs256"),
              (256000, "plan_adam_eps1e17_256k_bs256")],
    "muon": [(4000, "plan_muon_eps1e17_4k_bs256"),
             (8000, "plan_muon_eps1e17_8k_bs256"),
             (16000, "sm_muon_eps1e17_16k_bs256"),
             (32000, "plan_muon_eps1e17_32k_bs256"),
             (64000, "plan_muon_eps1e17_64k_bs256"),
             (128000, "plan_muon_eps1e17_128k_bs256"),
             (256000, "plan_muon_eps1e17_256k_bs256")],
}
SCORE_SUBDIR = {"magic": "scores", "ekfac": "ekfac_scores/scores"}
N_QUERIES = 20


def load_scores(scores_dir):
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


def find_dir(run, sub):
    for root in ROOTS:
        p = os.path.join(root, run, sub)
        if os.path.exists(p):
            return p
    return None


def gather(run, n_docs, scorer):
    """Per-query (score_sum, loss_change) for one run+scorer, or (None, why)."""
    summ = find_dir(run, "filter_proponents_%s/filter_summary.csv" % scorer)
    if summ is None:
        return None, "no filter_summary"
    obs = {int(r["query"]): float(r["filter_change"])
           for r in csv.DictReader(open(summ))}
    if len(obs) < N_QUERIES:
        return None, "partial: %d/%d queries" % (len(obs), N_QUERIES)
    sdir = find_dir(run, SCORE_SUBDIR[scorer])
    S = load_scores(sdir) if sdir else None
    if S is None:
        return None, "no scores"
    if S.shape[0] != n_docs or S.shape[1] < N_QUERIES:
        return None, "score shape %s != (%d, %d)" % (S.shape, n_docs, N_QUERIES)
    k = int(round(0.01 * n_docs))
    xs = np.array([np.partition(S[:, q], -k)[-k:].sum() for q in range(N_QUERIES)])
    ys = np.array([obs[q] for q in range(N_QUERIES)])
    return (xs, ys), "ok"


def ols(x, y):
    b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return a, b


points = {}  # (opt, scorer) -> list of dicts in scale order
for opt, row in ROWS.items():
    for scorer in ("magic", "ekfac"):
        pts = []
        for n_docs, run in row:
            got, why = gather(run, n_docs, scorer)
            if got is None:
                print("skip %-32s %-5s %s" % (run, scorer, why))
                continue
            xs, ys = got
            pts.append({"n_docs": n_docs, "run": run,
                        "xq": xs, "yq": ys,
                        "x": float(xs.mean()), "y": float(ys.mean())})
        points[(opt, scorer)] = pts

out = []
for (opt, scorer), pts in sorted(points.items()):
    if len(pts) < 4:
        print("\n%s/%s: only %d complete points, nothing to hold out" %
              (opt, scorer, len(pts)))
        continue
    print("\n== %s / %s: %d scaling points" % (opt, scorer, len(pts)))
    print("   %8s %12s %12s" % ("n_docs", "score_sum", "obs delta"))
    for p in pts:
        print("   %8d %12.4f %12.5f" % (p["n_docs"], p["x"], p["y"]))
    for n_train in (3, 4, 5):
        if n_train >= len(pts):
            continue
        tr, te = pts[:n_train], pts[n_train:]
        ln = np.log2([p["n_docs"] for p in tr])
        ytr = [p["y"] for p in tr]
        xtr = [p["x"] for p in tr]
        a_s, b_s = ols(ln, ytr)                      # scale-line
        a_g, b_g = ols(ln, np.log(ytr))              # power law in n_docs
        a_x, b_x = ols(xtr, ytr)                     # score OLS
        b_o = (np.dot(xtr, ytr) / np.dot(xtr, xtr))  # score through origin
        # per-query pooled fits
        xq = np.concatenate([p["xq"] for p in tr])
        yq = np.concatenate([p["yq"] for p in tr])
        a_q, b_q = ols(xq, yq)
        b_qo = float(np.dot(xq, yq) / np.dot(xq, xq))
        preds = {
            "mean3": lambda p: float(np.mean(ytr)),
            "scale-line": lambda p: a_s + b_s * np.log2(p["n_docs"]),
            "scale-power": lambda p: float(np.exp(a_g + b_g * np.log2(p["n_docs"]))),
            "score-ols": lambda p: a_x + b_x * p["x"],
            "score-origin": lambda p: b_o * p["x"],
            "perq-ols": lambda p: float(np.mean(a_q + b_q * p["xq"])),
            "perq-origin": lambda p: float(np.mean(b_qo * p["xq"])),
        }
        print("  train=%d test=%s" % (n_train, [p["n_docs"] for p in te]))
        for name, f in preds.items():
            errs = [f(p) - p["y"] for p in te]
            rmse = float(np.sqrt(np.mean(np.square(errs))))
            detail = "  ".join("%dk:%+.3f(pred %.3f/obs %.3f)"
                               % (p["n_docs"] // 1000, f(p) - p["y"], f(p), p["y"])
                               for p in te)
            print("    %-12s rmse %.4f   %s" % (name, rmse, detail))
            for p in te:
                out.append({"optimizer": opt, "scorer": scorer,
                            "n_train": n_train, "model": name,
                            "n_docs": p["n_docs"], "pred": f(p),
                            "obs": p["y"], "err": f(p) - p["y"],
                            "rmse_all_test": rmse})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["optimizer", "scorer", "n_train", "model",
                                      "n_docs", "pred", "obs", "err",
                                      "rmse_all_test"])
    w.writeheader()
    w.writerows(out)
print("\nwrote %s" % OUT)
