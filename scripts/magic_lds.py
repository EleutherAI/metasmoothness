"""MAGIC LDS from a bank's validation.csv — the grid's one implementation.

Usage:
    python magic_lds.py <run_dir-or-validation.csv> [--n-boot 10000] [--seed 0]

Definitions (CONTROLS "Attribution / estimator"):
- ``magic_lds`` is the MEAN over queries of the per-query Spearman correlation
  between each subset's summed MAGIC scores (``score_sum``) and its measured
  query-loss change (``diff``) — not the Spearman of pooled pairs.
- The 95% CI bootstrap resamples SUBSETS with replacement (10k resamples,
  seed 0), recomputing the per-query Spearmans and their mean per resample.
  Queries are the paired unit for optimizer contrasts and are not resampled
  here.
- Input is ``validation.csv`` alone (columns: subset, query, diff, score_sum);
  per-query score artifacts are not needed.

This implementation produced the recorded numbers for the first clean banks
(e.g. plan_adam_eps1e17_4k_bs256: 0.9295 [0.9195, 0.9381]); using anything else
for ``magic_lds``/``magic_ci_*`` cells makes CIs incomparable across rows.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import sys
# `python -P` keeps a bergson checkout's cwd off sys.path, but it also drops
# this script's own directory -- put just that one entry back for the import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_config  # noqa: E402


def magic_lds(validation_csv: Path, n_boot: int = 10000, seed: int = 0):
    v = pd.read_csv(validation_csv)
    diffs = v.pivot(index="subset", columns="query", values="diff").to_numpy()
    sums = v.pivot(index="subset", columns="query", values="score_sum").to_numpy()
    n_sub, n_q = diffs.shape

    per_q = np.array(
        [spearmanr(sums[:, q], diffs[:, q]).statistic for q in range(n_q)]
    )
    point = per_q.mean()

    rd = np.stack([rankdata(diffs[:, q]) for q in range(n_q)], axis=1)
    rs = np.stack([rankdata(sums[:, q]) for q in range(n_q)], axis=1)

    def mean_spearman(idx: np.ndarray) -> float:
        a = rs[idx] - rs[idx].mean(0)
        d = rd[idx] - rd[idx].mean(0)
        return ((a * d).sum(0) / np.sqrt((a**2).sum(0) * (d**2).sum(0))).mean()

    rng = np.random.default_rng(seed)
    boots = np.array(
        [mean_spearman(rng.integers(0, n_sub, n_sub)) for _ in range(n_boot)]
    )
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, lo, hi, per_q, n_sub


def merge_slices(run_dir: Path) -> Path:
    """Concatenate validation.csv and any validation_<a>_<b>.csv slices.

    Sharded banks (NODES.md "Sharding a bank's retrains") write one CSV per
    process; subset indices are global, so the union is the full bank. Writes
    ``validation_merged.csv`` and fails if any subset is missing or duplicated.
    """
    parts = sorted(run_dir.glob("validation_*_*.csv"))
    if (run_dir / "validation.csv").exists():
        parts.insert(0, run_dir / "validation.csv")
    if len(parts) == 1:
        return parts[0]
    v = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates()
    seen = v.groupby("subset")["query"].count()
    n_q = v["query"].nunique()
    bad = seen[seen != n_q]
    assert bad.empty, f"subsets with duplicate/partial rows: {bad.index.tolist()}"
    cfg = run_config.resolve(run_dir)
    expected = int(re.search(r"num_subsets: (\d+)", cfg.read_text()).group(1))
    missing = sorted(set(range(expected)) - set(seen.index))
    assert not missing, f"missing subsets: {missing}"
    out = run_dir / "validation_merged.csv"
    v.sort_values(["subset", "query"]).to_csv(out, index=False)
    print(f"merged {len(parts)} slice CSVs -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="run dir or validation.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    csv = args.path if args.path.suffix == ".csv" else merge_slices(args.path)
    point, lo, hi, per_q, n_sub = magic_lds(csv, args.n_boot, args.seed)
    print(f"magic_lds {point:.4f} [{lo:.4f}, {hi:.4f}]  n_subsets={n_sub} n_queries={len(per_q)}")
    print("per-query:", np.round(per_q, 4).tolist())


if __name__ == "__main__":
    main()
