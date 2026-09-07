#!/usr/bin/env python3
"""Runtime calculator for bergson runs, calibrated on data/run_durations.csv (see run_durations.py).

Modeled on the compute-planning sheet's "empirical constant" tab: instead of FLOPs x MFU we fit, per
(step, method, model, gpu_type, gpus), an empirical rate  minutes per 1k document-passes  from finished runs
(median across runs, with spread), then predict new runs from document count, passes and GPU allocation.

    python runtime_calculator.py fit                       # -> data/run_rates.csv (+ prints the table)
    python runtime_calculator.py predict --step validate --method filter-proponents --model Qwen2.5-1.5B \
        --n-docs 512000 --gpus 2 --runs 120 --parallel 50    # -> hours per run, GPU-hours, wall-clock with N parallel jobs
    python runtime_calculator.py grid --model Qwen2.5-1.5B --n-docs 256000 --parallel 50   # a whole row's pipeline

Document-passes: `validate` runs train twice (full corpus, then the (1-fraction) subset), so passes = 2 - fraction;
`train`/`magic` train once; scoring steps (ekfac/score/bif) are charged per corpus pass with their query batching
folded into the empirical rate (rates are only valid for the same query_batch_size / apply_batch_size regime).
Rates are per job, not per GPU: a 2-GPU job's rate already reflects 2 GPUs.
"""
import argparse, csv, math, os, statistics as st, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "scripts" in os.path.abspath(__file__) else "/mnt/ssd-2/lucia/metasmoothness"
DUR = os.path.join(ROOT, "data", "run_durations.csv"); RATES = os.path.join(ROOT, "data", "run_rates.csv")


def stage_of(r):
    """Calibration label: validate/train/magic keep their method; ekfac splits into the Hessian fit (run names
    with `h64`, or any run without a `resume` flag that fits factors) and the scoring stage (resumes a fitted Hessian)."""
    if r["step"] == "ekfac":
        return "hessian" if ("h64" in r["run"] or r["resume"] != "True") else "score"
    return r["method"]


def passes_of(r):
    if r["step"] == "validate":
        f = float(r["subset_fraction"] or 0); ep = float(r["epochs"] or 1)
        return (1 + (1 - f)) * ep
    if r["step"] in ("train", "magic"):
        return float(r["epochs"] or 1)
    if r["step"] == "ekfac" and stage_of(r) == "score":
        qbs = int(float(r["query_batch_size"] or 0)) or 20
        return math.ceil(20 / qbs)  # one corpus pass per query batch
    return 1.0


def key(r):
    return (r["step"], stage_of(r), r["model"], r["gpu_type_inferred"], r["gpus"])


def ols(xs, ys):
    """duration_min = a + b * kdocpasses. The marginal rate b is the median through-origin rate of the largest runs
    (those within a factor 2 of the largest doc-pass count), where fixed overhead is negligible and the fleet's
    per-step speed is what matters; the overhead a is the median excess of the remaining (small) runs over b*x, floored
    at 0. Plain least squares was rejected: heteroscedastic small runs flattened the slope by ~40%."""
    pts = sorted(zip(xs, ys)); xmax = pts[-1][0]
    big = [(x, y) for x, y in pts if x >= 0.5 * xmax] or pts[-1:]
    b = st.median(y / x for x, y in big)
    small = [(x, y) for x, y in pts if x < 0.5 * xmax]
    a = max(0.0, st.median(y - b * x for x, y in small)) if small else 0.0
    return a, b


def fit(write=True):
    rows = [r for r in csv.DictReader(open(DUR)) if r["status"] == "done" and r["duration_min"] and r["n_docs"]]
    groups = defaultdict(list)
    for r in rows:
        if r["resume"] == "True" and r["step"] not in ("validate", "ekfac"):
            continue  # resumed runs of other steps measure only their last segment
        dp = float(r["n_docs"]) * passes_of(r) / 1000.0
        if dp <= 0: continue
        groups[key(r)].append((dp, float(r["duration_min"]), float(r["n_docs"])))
    out = []
    for k, v in sorted(groups.items()):
        xs = [x[0] for x in v]; ys = [x[1] for x in v]
        a, b = ols(xs, ys)
        resid = [y / (a + b * x) for x, y in zip(xs, ys)]
        out.append({"step": k[0], "stage_or_method": k[1], "model": k[2], "gpu_type": k[3], "gpus": k[4], "n_runs": len(v),
                    "overhead_min": round(a, 1), "marginal_min_per_1k_docpasses": round(b, 3),
                    "actual_over_predicted_p10": round(sorted(resid)[max(0, int(0.1 * len(resid)) - 1)], 2),
                    "actual_over_predicted_p90": round(sorted(resid)[min(len(resid) - 1, int(0.9 * len(resid)))], 2),
                    "n_docs_min": int(min(x[2] for x in v)), "n_docs_max": int(max(x[2] for x in v)),
                    "gpu_hours_per_1k_docpasses_marginal": round(b / 60 * int(k[4]), 3)})
    if write:
        with open(RATES, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
        print(f"{len(out)} rate rows -> {RATES}")
    return out


def load_rates():
    return list(csv.DictReader(open(RATES))) if os.path.exists(RATES) else fit(write=False)


def predict(step, method, model, n_docs, gpus, gpu_type, passes, runs=1, parallel=1, rates=None):
    rates = rates or load_rates()
    cands = [r for r in rates if r["step"] == step and r["model"] == model and str(r["gpus"]) == str(gpus) and r["gpu_type"] == gpu_type and (not method or r["stage_or_method"] == method)]
    if not cands:
        raise SystemExit(f"no calibration for step={step} stage/method={method} model={model} gpus={gpus} gpu_type={gpu_type}; calibrated combinations:\n" + "\n".join(f"  {r['step']:9s} {r['stage_or_method']:18s} {r['model']:13s} {r['gpu_type']:15s} gpus={r['gpus']} (n={r['n_runs']})" for r in rates))
    r = cands[0]
    a, b, lo, hi = (float(r[c]) for c in ("overhead_min", "marginal_min_per_1k_docpasses", "actual_over_predicted_p10", "actual_over_predicted_p90"))
    dp = n_docs * passes / 1000.0
    per_run_h = (a + b * dp) / 60
    waves = math.ceil(runs / max(1, parallel))
    return {"per_run_hours": per_run_h, "per_run_hours_p10_p90": (per_run_h * lo, per_run_h * hi), "gpu_hours_total": per_run_h * int(gpus) * runs,
            "wall_clock_hours": per_run_h * waves, "waves": waves, "calibration_runs": r["n_runs"], "calibrated_n_docs": (r["n_docs_min"], r["n_docs_max"])}


EPOCHS = 2  # the Qwen rows train 2 epochs; doc-passes = corpus passes x epochs
GRID = [  # one Qwen 1.5B EK-FAC row: (label, step, stage_or_method, gpus, gpu_type, doc-passes per doc, runs)
    ("base model (1 training pass x 2 epochs)", "validate", "lds", 2, "A40-48GB", 1.0 * EPOCHS, 1),
    ("control banks (18; base + subset pass)", "validate", "lds", 2, "A40-48GB", 1.95 * EPOCHS, 18),
    ("EK-FAC Hessian fit on 64k docs (lotus)", "ekfac", "hessian", 6, "A100-SXM4-80GB", 1.0, 1),
    ("EK-FAC scoring, 3 query passes (lotus)", "ekfac", "score", 6, "A100-SXM4-80GB", 3.0, 1),
    ("filter shards (120; base + filtered pass)", "validate", "filter-proponents", 2, "A40-48GB", 1.99 * EPOCHS, 120),
]


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    p = sub.add_parser("predict"); p.add_argument("--step", required=True); p.add_argument("--method", default="")
    p.add_argument("--model", required=True); p.add_argument("--n-docs", type=int, required=True); p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--gpu-type", default="A40-48GB"); p.add_argument("--passes", type=float, default=None, help="doc-passes per document; default: validate 1.99 x epochs, train/magic = epochs, ekfac score = ceil(20/qbs)")
    p.add_argument("--epochs", type=float, default=2.0); p.add_argument("--query-batch-size", type=int, default=7)
    p.add_argument("--runs", type=int, default=1); p.add_argument("--parallel", type=int, default=1)
    g = sub.add_parser("grid"); g.add_argument("--model", default="Qwen2.5-1.5B"); g.add_argument("--n-docs", type=int, required=True); g.add_argument("--parallel", type=int, default=34)
    a = ap.parse_args()
    if a.cmd == "fit":
        for r in fit():
            print(f"{r['step']:9s} {r['stage_or_method']:18s} {r['model']:13s} {r['gpu_type']:15s} gpus={r['gpus']} n={r['n_runs']:4d}  {r['overhead_min']:6.1f} min + {r['marginal_min_per_1k_docpasses']:7.3f} min/1k doc-passes  actual/pred p10-p90 [{r['actual_over_predicted_p10']}, {r['actual_over_predicted_p90']}]  docs {r['n_docs_min']}-{r['n_docs_max']}")
    elif a.cmd == "predict":
        if a.passes is not None: passes = a.passes
        elif a.step == "validate": passes = 1.99 * a.epochs
        elif a.step in ("train", "magic"): passes = a.epochs
        elif a.step == "ekfac" and a.method == "score": passes = math.ceil(20 / a.query_batch_size)
        else: passes = 1.0
        o = predict(a.step, a.method, a.model, a.n_docs, a.gpus, a.gpu_type, passes, a.runs, a.parallel)
        print(f"per run: {o['per_run_hours']:.2f} h (p10-p90 {o['per_run_hours_p10_p90'][0]:.2f}-{o['per_run_hours_p10_p90'][1]:.2f}); {a.runs} runs = {o['gpu_hours_total']:.0f} GPU-hours; "
              f"wall clock {o['wall_clock_hours']:.1f} h in {o['waves']} waves of {a.parallel}; calibrated on {o['calibration_runs']} runs at {o['calibrated_n_docs'][0]}-{o['calibrated_n_docs'][1]} docs")
    else:
        tot_gpu = 0; tot_wall = 0
        print(f"{a.model} row at {a.n_docs} docs, {a.parallel} A40 pairs available:")
        for label, step, method, gpus, gt, passes, runs in GRID:
            n = 64000 if "64k" in label else a.n_docs
            try:
                o = predict(step, method, a.model, n, gpus, gt, passes, runs, a.parallel if gpus == 2 else 1)
            except SystemExit as e:
                print(f"  {label:42s} no calibration"); continue
            tot_gpu += o["gpu_hours_total"]; tot_wall += o["wall_clock_hours"]
            print(f"  {label:42s} {o['per_run_hours']:7.1f} h/run x{runs:4d} = {o['gpu_hours_total']:8.0f} GPU-h, wall {o['wall_clock_hours']:7.1f} h ({o['waves']} waves)")
        print(f"  {'TOTAL (serial phases)':42s} {tot_gpu:8.0f} GPU-h, wall {tot_wall:.1f} h = {tot_wall/24:.1f} days")


if __name__ == "__main__":
    main()
