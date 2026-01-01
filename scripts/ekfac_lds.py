"""EK-FAC LDS: correlate bank ground truth with EK-FAC scores.

Usage:
    python ekfac_lds.py --scores DIR --bank DIR [--n-boot 10000]

--scores: an ekfac_scores/scores dir (scores.bin + info.json, the bergson
structured memmap: float32 score_i + bool written_i per query).
--bank: a dir containing subsets.json and eval_q20/validation.csv
(columns subset,query,diff,...). The estimator is the established pipeline:
per-query Spearman between each subset's summed scores and its measured
query-loss diff, averaged over queries; CI from a 10k bootstrap resampling
subsets per query.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def load_scores(scores_dir: Path) -> np.ndarray:
    info = json.loads((scores_dir / "info.json").read_text())
    dt = np.dtype({k: info["dtype"][k] for k in ("names", "formats", "offsets", "itemsize")})
    raw = np.memmap(scores_dir / "scores.bin", dtype=dt, mode="r")
    n_q = info["num_scores"]
    for q in range(n_q):
        assert raw[f"written_{q}"].all(), f"query {q} has unwritten scores"
    return np.stack([raw[f"score_{q}"] for q in range(n_q)], axis=1)  # (docs, queries)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scores = load_scores(args.scores)
    subsets = json.loads(next(args.bank.rglob("subsets.json")).read_text())
    # A sharded bank keeps only the pre-shard prefix in validation.csv; the
    # merged file is the complete ground truth. Prefer it whenever it exists.
    merged = sorted(args.bank.rglob("validation_merged.csv"))
    val_path = merged[0] if merged else next(args.bank.rglob("validation.csv"))
    val = pd.read_csv(val_path)
    print(f"bank ground truth: {val_path.name} "
          f"({val['subset'].nunique()} subsets, {val['query'].nunique()} queries)")
    n_sub = val.subset.nunique()
    n_q = val["query"].nunique()
    assert scores.shape[1] == n_q, (scores.shape, n_q)

    # (subsets, queries) summed scores over each subset's removed docs.
    # Negated: raw bergson scores are gradient-signed; the recorded ekfac_lds
    # convention is loss-signed (validated against the per-epoch grid values,
    # which this reproduces exactly after negation).
    sums = -np.stack([scores[np.asarray(subsets[s])].sum(0) for s in range(n_sub)])
    diffs = val.pivot(index="subset", columns="query", values="diff").to_numpy()
    assert diffs.shape == sums.shape

    per_q = np.array([spearmanr(sums[:, q], diffs[:, q]).statistic for q in range(n_q)])
    lds = per_q.mean()

    rng = np.random.default_rng(args.seed)
    r_sums = np.stack([rankdata(sums[:, q]) for q in range(n_q)], axis=1)
    r_diffs = np.stack([rankdata(diffs[:, q]) for q in range(n_q)], axis=1)
    boots = np.empty(args.n_boot)
    for b in range(args.n_boot):
        idx = rng.integers(0, n_sub, n_sub)
        a, d = r_sums[idx], r_diffs[idx]
        a = a - a.mean(0)
        d = d - d.mean(0)
        num = (a * d).sum(0)
        den = np.sqrt((a**2).sum(0) * (d**2).sum(0))
        boots[b] = (num / den).mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])

    print(f"ekfac_lds {lds:.4f} [{lo:.4f}, {hi:.4f}]  n_subsets={n_sub} n_queries={n_q}")
    print("per-query:", np.round(per_q, 4).tolist())


if __name__ == "__main__":
    main()
