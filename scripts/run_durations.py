#!/usr/bin/env python3
"""Inventory of how long every finished bergson run took, per hardware, for a runtime calculator.

    python run_durations.py [--out data/run_durations.csv]

For every run directory (depth <= 3 under both experiment roots) with a config.yaml, read the step config and take
  start = mtime of config.yaml (bergson writes it when the run starts; on `resume: true` relaunches it is rewritten,
          so such rows measure only the last segment and are flagged resume=True)
  end   = newest mtime of the step's terminal artifact (validate -> filter_summary.csv / validation*.csv / summary*.csv;
          ekfac/score/bif -> scores/scores.bin; magic -> scores/scores.bin or per_query/*.pt; train -> model weights)
GPU type is inferred from the fleet placement rules: 6-GPU jobs ran on lotus-0 (A100-SXM4-80GB); every 2/4/8-GPU
job ran on A40-48GB pods. Logs carry no config path or timestamps, so per-phase rates are derived, not parsed.
Training-step estimates: passes * ceil(n_docs_used / batch_size) * epochs, where a `validate` run makes two passes
(full-corpus base, then the subset/filtered retrain on (1-fraction) of the corpus).
"""
import argparse, csv, glob, math, os, re, sys, time
import yaml

ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments", "/mnt/ssd-1/lucia/paper_runs/experiments"]
MODEL_PARAMS = {"gpt2": 124e6, "gpt2-medium": 355e6, "Qwen2.5-1.5B": 1.54e9, "Qwen2.5-3B": 3.09e9}


def n_from_name(name):
    m = re.search(r"(\d+)(k|m)?", name or "")
    if not m: return None
    n = int(m.group(1)); u = m.group(2)
    return n * (1000 if u == "k" else 1_000_000 if u == "m" else 1)


def model_name(path, row=""):
    """Model family from the config's model path, falling back to the row naming convention when the path is a
    row-local fine-tuned model (e.g. <row>/base/model)."""
    for k in MODEL_PARAMS:
        if k.lower() in (path or "").lower(): return k
    r = row.lower()
    if r.startswith("qwen15b"): return "Qwen2.5-1.5B"
    if r.startswith("qwen3b"): return "Qwen2.5-3B"
    if "medium" in r: return "gpt2-medium"
    if r.startswith(("plan_", "sm_", "heldout", "lds_", "bank", "muon", "adam")): return "gpt2"
    p = (path or "").rstrip("/")
    if p.endswith("/model"): p = p[:-6]
    return os.path.basename(p)


def newest(patterns, d):
    t = 0.0; which = ""
    for pat in patterns:
        for f in glob.glob(os.path.join(d, pat)):
            try:
                s = os.stat(f)
            except OSError:
                continue
            if s.st_size == 0: continue  # banks create an empty validation csv at start; only a written marker counts
            if s.st_mtime > t: t, which = s.st_mtime, os.path.relpath(f, d)
    return t, which


END = {  # terminal artifacts written at the END of a step (not files bergson creates at start)
    "validate": ["filter_summary.csv", "summary*.csv"],
    "ekfac": ["scores/scores.bin"], "score": ["scores/scores.bin"], "bif": ["scores/scores.bin"],
    "magic": ["scores/scores.bin", "per_query/*.pt", "validation*.csv"],
    "train": ["model/model.safetensors", "retrained/base/model.safetensors", "checkpoints/step_*.ckpt/.metadata", "model/*.safetensors"],
}

rows = []
for root in ROOTS:
    if not os.path.isdir(root): continue
    for cfgp in glob.glob(os.path.join(root, "*", "config.yaml")) + glob.glob(os.path.join(root, "*", "*", "config.yaml")) + glob.glob(os.path.join(root, "*", "*", "*", "config.yaml")):
        d = os.path.dirname(cfgp)
        try:
            cfg = yaml.safe_load(open(cfgp))
        except Exception as e:
            continue
        if not isinstance(cfg, dict) or "steps" not in cfg or not cfg["steps"]: continue
        step = cfg["steps"][0]
        if not isinstance(step, dict) or not step: continue
        stype, sc = next(iter(step.items()))
        if stype == "metasmoothness" or not isinstance(sc, dict): continue
        ic = sc.get("index_cfg", sc)  # ekfac nests most fields under index_cfg
        data = ic.get("data", {}) or {}
        query = (sc.get("hessian_pipeline_cfg", {}) or {}).get("query") or sc.get("query") or {}
        dataset = os.path.basename((data.get("dataset") or "").rstrip("/"))
        nproc = ((ic.get("distributed") or {}).get("nproc_per_node")) or ((sc.get("distributed") or {}).get("nproc_per_node")) or 1
        row_name0 = os.path.relpath(d, root).split("/")[0]
        model = model_name(ic.get("model") or sc.get("model") or "", row_name0)
        method = sc.get("method") or ((sc.get("hessian_cfg") or {}).get("method") if stype == "ekfac" else "") or ""
        bs = sc.get("batch_size") or ic.get("token_batch_size") or ""
        epochs = sc.get("num_epochs") or ""
        frac = sc.get("subset_fraction") or ""
        nsub = sc.get("num_subsets")
        resume = bool(sc.get("resume") or ic.get("resume") or (sc.get("hessian_pipeline_cfg") or {}).get("resume"))
        start = os.stat(cfgp).st_mtime
        end, marker = newest(END.get(stype, ["scores/scores.bin"]), d)
        row_name = os.path.relpath(d, root).split("/")[0]
        n_docs = n_from_name(dataset) or n_from_name(row_name)
        rel = os.path.relpath(d, root)
        # training-pass estimate
        passes = steps_est = ""
        if stype in ("validate", "train", "magic") and bs and n_docs:
            ep = float(epochs or 1)
            if stype == "validate":
                f = float(frac or 0)
                sub_n = n_docs * (1 - f) if method == "filter-proponents" or nsub == 0 else n_docs * (1 - f)
                steps_est = int(math.ceil(n_docs / bs) * ep + math.ceil(sub_n / bs) * ep); passes = 2
            else:
                steps_est = int(math.ceil(n_docs / bs) * ep); passes = 1
        gpu_type = "A100-SXM4-80GB" if int(nproc) in (5, 6) else "A40-48GB"
        done = end > start
        dur = (end - start) / 60 if done else ""
        rows.append({
            "volume": root.split("/")[2], "row": row_name, "run": rel, "step": stype, "method": method,
            "model": model, "params": int(MODEL_PARAMS.get(model, 0)) or "", "dataset": dataset, "n_docs": n_docs or "",
            "batch_size": bs, "epochs": epochs, "subset_fraction": frac, "num_subsets": nsub if nsub is not None else "",
            "query_dataset": os.path.basename((query.get("dataset") or "").rstrip("/")) if isinstance(query, dict) else "",
            "query_batch_size": (sc.get("score_cfg") or {}).get("query_batch_size", "") if stype == "ekfac" else "",
            "apply_batch_size": (sc.get("hessian_pipeline_cfg") or {}).get("apply_batch_size", "") if stype == "ekfac" else "",
            "gpus": nproc, "gpu_type_inferred": gpu_type, "precision": ic.get("precision") or sc.get("precision") or "",
            "resume": resume, "passes": passes, "train_steps_est": steps_est,
            "start_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime(start)),
            "end_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime(end)) if done else "",
            "end_marker": marker if done else "", "status": "done" if done else "incomplete_or_running",
            "duration_min": f"{dur:.1f}" if dur != "" else "",
            "sec_per_train_step": f"{dur*60/steps_est:.2f}" if (dur != "" and steps_est) else "",
            "gpu_hours": f"{dur/60*int(nproc):.2f}" if dur != "" else "",
        })

rows.sort(key=lambda r: (r["volume"], r["row"], r["run"]))
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/mnt/ssd-2/lucia/metasmoothness/data/run_durations.csv")
out = ap.parse_args().out
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
done = [r for r in rows if r["status"] == "done"]
print(f"{len(rows)} runs ({len(done)} finished) -> {out}")
from collections import Counter
print(Counter((r["step"], r["model"], r["gpu_type_inferred"], r["gpus"]) for r in done).most_common(20))
