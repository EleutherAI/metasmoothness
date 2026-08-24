"""Is tuned learning rate related to attributability across the grid?"""
import csv

import numpy as np
from scipy.stats import spearmanr

rows = [r for r in csv.DictReader(open("/mnt/ssd-2/lucia/metasmoothness/experiments.csv"))
        if r["magic_lds"].strip() and r["lr"].strip()]
lr = np.array([float(r["lr"]) for r in rows])
magic = np.array([float(r["magic_lds"]) for r in rows])
ms = np.array([float(r["metasmoothness"]) if r["metasmoothness"].strip() else np.nan
               for r in rows])

print(f"{'run':32s} {'lr':>8s} {'MAGIC':>8s} {'ms':>8s}")
for r, a, b, c in sorted(zip(rows, lr, magic, ms), key=lambda t: t[1]):
    msf = f"{c:8.4f}" if not np.isnan(c) else f"{'-':>8s}"
    print(f"{r['run_id'][:32]:32s} {a:8.1e} {b:8.4f} {msf}")

print(f"\nn = {len(rows)}")
print(f"spearman(lr, MAGIC)                 = {spearmanr(lr, magic).statistic:+.3f}  "
      f"p={spearmanr(lr, magic).pvalue:.3f}")
# Distance from the anchor lr, in octaves -- tests "deviating from 2e-4 is what hurts"
dev = np.abs(np.log2(lr / 2e-4))
print(f"spearman(|log2(lr/2e-4)|, MAGIC)    = {spearmanr(dev, magic).statistic:+.3f}  "
      f"p={spearmanr(dev, magic).pvalue:.3f}")

print("\nthe three collapsed rows and their lr:")
for r, a, b in sorted(zip(rows, lr, magic), key=lambda t: t[2])[:3]:
    print(f"  {r['run_id'][:34]:34s} lr={a:.1e}  MAGIC={b:.4f}")
print("\nrows sharing lr 5e-5 (bs16's value):")
for r, a, b in zip(rows, lr, magic):
    if abs(a - 5e-5) < 1e-9:
        print(f"  {r['run_id'][:34]:34s} MAGIC={b:.4f}")
print("\nrows sharing lr 1e-4:")
for r, a, b in zip(rows, lr, magic):
    if abs(a - 1e-4) < 1e-9:
        print(f"  {r['run_id'][:34]:34s} MAGIC={b:.4f}")
