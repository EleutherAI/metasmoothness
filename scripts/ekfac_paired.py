"""Paired EK-FAC optimizer contrast -- the estimator paired_diff.py uses for MAGIC.

Overlapping confidence intervals are the weak test. CONTROLS pairs over queries
because the subset draws are seeded identically across optimizers, so each query
is a matched unit; the bootstrap resamples QUERIES, not subsets.

Score loading and the loss-sign convention are imported from ekfac_lds.py rather
than reimplemented -- the raw bergson scores are gradient-signed and must be
negated, which is exactly the kind of detail a second copy gets wrong.

    python ekfac_paired.py <run_a> <run_b> <name_a> <name_b>
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_spec = importlib.util.spec_from_file_location(
    "ekfac_lds", "/mnt/ssd-2/lucia/metasmoothness/scripts/ekfac_lds.py")
_ekfac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ekfac)

EXP = ["/mnt/ssd-2/lucia/paper_runs/experiments",
       "/mnt/ssd-1/lucia/paper_runs/experiments"]


def resolve(run_id: str) -> Path:
    for base in EXP:
        p = Path(base) / run_id
        if p.is_dir():
            return p
    sys.exit(f"run dir not found: {run_id}")


def per_query_rho(run: Path) -> np.ndarray:
    scores = _ekfac.load_scores(run / "ekfac_scores" / "scores")
    subsets = json.loads(next(run.rglob("subsets.json")).read_text())
    merged = sorted(run.rglob("validation_merged.csv"))
    val_path = merged[0] if merged else next(run.rglob("validation.csv"))
    val = pd.read_csv(val_path)
    n_sub, n_q = val.subset.nunique(), val["query"].nunique()
    print(f"  {run.name}: {val_path.name} ({n_sub} subsets, {n_q} queries)")

    # Negated to the loss-signed convention, matching ekfac_lds.py.
    sums = -np.stack([scores[np.asarray(subsets[s])].sum(0) for s in range(n_sub)])
    diffs = val.pivot(index="subset", columns="query", values="diff").to_numpy()
    assert diffs.shape == sums.shape, (diffs.shape, sums.shape)
    return np.array([spearmanr(sums[:, q], diffs[:, q]).statistic for q in range(n_q)])


a, b = resolve(sys.argv[1]), resolve(sys.argv[2])
a_name, b_name = sys.argv[3], sys.argv[4]
ra, rb = per_query_rho(a), per_query_rho(b)
assert len(ra) == len(rb), f"query counts differ: {len(ra)} vs {len(rb)}"

d = ra - rb
rng = np.random.default_rng(0)
idx = rng.integers(0, len(d), size=(10_000, len(d)))
boot = d[idx].mean(axis=1)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"{a_name} EK-FAC mean per-query rho = {ra.mean():.4f}")
print(f"{b_name} EK-FAC mean per-query rho = {rb.mean():.4f}")
print(f"paired {a_name} - {b_name} = {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
      f"half-width {(hi - lo) / 2:.4f}")
print(f"query wins for {a_name}: {(d > 0).sum()}/{len(d)}")
