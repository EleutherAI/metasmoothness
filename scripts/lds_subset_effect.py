"""Is the LDS jump explained by the three added subsets, or by changed rows?

The muon 4000-step row read 0.3273 [0.254, 0.394] on a 97-subset merge and
0.4541 [0.407, 0.498] on the finished 100-subset merge. Those intervals are
disjoint, so this is not sampling noise -- one of the two is wrong.

Two candidates:
  (a) subsets 68, 98, 99 are genuinely that influential;
  (b) the 97-subset merge contained stale rows for subsets that later-finishing
      shards rewrote, so the two merges disagree about SHARED subsets too.

Recomputing on the current merged file with those three dropped separates them.
If it lands near 0.3273 the answer is (a); near 0.4541, the answer is (b) and the
earlier value was built on rows that no longer exist.
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

B = "/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_64k_bs32/bank_from_filter"
rows = list(csv.DictReader(open(B + "/validation_merged.csv")))

def lds(drop=()):
    per_q = defaultdict(list)
    for r in rows:
        s = int(r["subset"])
        if s in drop:
            continue
        per_q[int(r["query"])].append((float(r["score_sum"]), float(r["diff"])))
    rhos = []
    for q, pairs in per_q.items():
        x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
        rhos.append(spearmanr(x, y).statistic)
    n = len(next(iter(per_q.values())))
    return float(np.mean(rhos)), n

full, n_full = lds()
less, n_less = lds(drop={68, 98, 99})
print("  all 100 subsets              lds %.4f  (n_subsets=%d)" % (full, n_full))
print("  dropping 68, 98, 99          lds %.4f  (n_subsets=%d)" % (less, n_less))
print()
print("  the 97-subset merge reported 0.3273")
if abs(less - 0.3273) < 0.02:
    print("  -> matches: those three subsets really do move it, (a)")
else:
    print("  -> does NOT match: the earlier merge disagreed about shared subsets too, (b)")
    print("     the 0.3273 reading was built on rows that have since been rewritten")
