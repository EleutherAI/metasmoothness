"""Gate-time audit for one-query filter shards. For every shard config:
  (1) the query slice's row is byte-identical to row a of the full 20-query set (row a <-> score column a);
  (2) the score slice equals column a of the full score matrix and its config says higher_is_better: true;
  (3) the removed set bergson will pick (top-k of the nonzero-score pool, k = round(frac * pool)) equals the
      top-k of the full matrix for that query;
  (4) that selection overlaps >= 60% with the reference-N selection for the same query on the aligned prefix.
Exit 1 on any failure. Nothing here spends a retrain.
Env overrides (defaults = qwen 64k plan A): EXP, SCORES (full scores dir), REF_SCORES, REF_N, N,
CFGS ("prefix:frac,prefix:frac"), QSET (full query set)."""
import json, os, sys, yaml, numpy as np
from pathlib import Path
from datasets import load_from_disk

E = Path("/mnt/ssd-2/lucia/paper_runs/experiments"); D = Path("/mnt/ssd-2/lucia/datasets_local")
exp = Path(os.environ.get("EXP", E / "qwen15b_64k_bs256"))
scores_dir = Path(os.environ.get("SCORES", exp / "ekfac_scores_64k/scores"))
ref_dir = Path(os.environ.get("REF_SCORES", E / "qwen15b_32k_bs256/ekfac_scores/scores"))
REF_N = int(os.environ.get("REF_N", 32000)); N_EXPECT = int(os.environ.get("N", 64000))
CFGS = [(p, float(f)) for p, f in (x.split(":") for x in os.environ.get("CFGS", "filter_proponents_ekfac:0.01,filter_top40_ekfac:0.000625").split(","))]
full_q = load_from_disk(os.environ.get("QSET", str(D / "query_20_qwen.hf"))); assert len(full_q) == 20


def load(d):
    info = json.load(open(d / "info.json")); dd = info["dtype"]
    dt = np.dtype({"names": dd["names"], "formats": [np.dtype(f) for f in dd["formats"]], "offsets": dd["offsets"], "itemsize": dd["itemsize"]})
    mm = np.memmap(d / "scores.bin", dtype=dt, mode="r", shape=(info["num_rows"],))
    S = np.stack([np.asarray(mm[f"score_{j}"], dtype=np.float64) for j in range(info["num_scores"])])
    W = np.stack([np.asarray(mm[f"written_{j}"]) for j in range(info["num_scores"])]).astype(bool)
    return S, W


def bergson_selection(col, frac):
    """Mirror validate.py: pool = docs with nonzero score, k = max(1, round(frac * pool)), top-k by score (higher is better)."""
    valid = np.nonzero(col != 0)[0]
    k = max(1, round(frac * len(valid)))
    order = valid[np.argsort(-col[valid], kind="stable")]
    return set(order[:k].tolist()), k


F, W = load(scores_dir); assert W.all(), f"scores not fully written: {W.mean():.4f}"
SR, WR = load(ref_dir); assert WR.all()
N = F.shape[1]; assert F.shape == (20, N_EXPECT), F.shape
fails = 0
for prefix, frac in CFGS:
    for a in range(20):
        p = exp / f"{prefix}_q{a}_{a+1}.yaml"
        if not p.exists():
            continue
        cfg = yaml.safe_load(open(p))["steps"][0]["validate"]; problems = []
        if cfg["subset_fraction"] != frac or cfg["num_subsets"] != 0:
            problems.append("config fields")
        qs = load_from_disk(cfg["query"]["dataset"])
        if len(qs) != 1 or qs[0]["input_ids"] != full_q[a]["input_ids"]:
            problems.append("query slice is not row a of the full set")
        sl, wl = load(Path(cfg["scores"]))
        if sl.shape != (1, N) or not wl.all() or not np.array_equal(sl[0], F[a]):
            problems.append("score slice != column a")
        try:
            hib = yaml.safe_load(open(Path(cfg["scores"]) / "config.yaml"))["steps"][0]["score"]["score_cfg"]["higher_is_better"]
        except Exception:
            hib = None
        if hib is not True:
            problems.append(f"higher_is_better={hib!r}")
        sel, k = bergson_selection(sl[0], frac); full_sel, _ = bergson_selection(F[a], frac)
        if sel != full_sel:
            problems.append("selection from slice != from full matrix")
        kr = max(1, round(frac * REF_N))
        sref = set(np.argsort(-SR[a])[:kr].tolist()); sal = set(np.argsort(-F[a][:REF_N])[:kr].tolist()); ov = len(sref & sal) / kr
        if ov < 0.6:
            problems.append(f"overlap with ref selection only {ov:.2f}")
        fails += bool(problems)
        print(f"{prefix[-14:]} q{a:2d} k={k:4d} overlap_ref={ov:.2f} {'OK' if not problems else 'FAIL: ' + '; '.join(problems)}")
print(f"SELECTION AUDIT: {'PASS' if fails == 0 else f'FAIL ({fails} shards)'}")
sys.exit(1 if fails else 0)
