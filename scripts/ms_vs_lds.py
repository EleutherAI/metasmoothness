"""Relationship between the metasmoothness probe and measured attributability."""
import csv

import numpy as np
from scipy.stats import spearmanr

rows = [r for r in csv.DictReader(open("/mnt/ssd-2/lucia/metasmoothness/experiments.csv"))
        if r["metasmoothness"].strip() and r["magic_lds"].strip()]

ms = np.array([float(r["metasmoothness"]) for r in rows])
magic = np.array([float(r["magic_lds"]) for r in rows])
ek = np.array([float(r["ekfac_lds"]) if r["ekfac_lds"].strip() else np.nan for r in rows])

print(f"{'run':32s} {'ms':>9s} {'MAGIC':>8s} {'EK-FAC':>8s}")
for r, a, b, c in sorted(zip(rows, ms, magic, ek), key=lambda t: -t[1]):
    print(f"{r['run_id'][:32]:32s} {a:9.4f} {b:8.4f} "
          f"{c:8.4f}" if not np.isnan(c) else
          f"{r['run_id'][:32]:32s} {a:9.4f} {b:8.4f} {'-':>8s}")

print(f"\nn = {len(rows)}")
print(f"spearman(ms, MAGIC)  = {spearmanr(ms, magic).statistic:+.3f}  "
      f"p={spearmanr(ms, magic).pvalue:.3f}")
m = ~np.isnan(ek)
print(f"spearman(ms, EK-FAC) = {spearmanr(ms[m], ek[m]).statistic:+.3f}  "
      f"p={spearmanr(ms[m], ek[m]).pvalue:.3f}")

# Within-pair direction: the optimizer contrast, where both arms share everything else.
by = {r["run_id"]: (float(r["metasmoothness"]), float(r["magic_lds"])) for r in rows}
PAIRS = [("plan_adam_eps1e17_4k_bs256", "plan_muon_eps1e17_4k_bs256", "4k"),
         ("plan_adam_eps1e17_8k_bs256", "plan_muon_eps1e17_8k_bs256", "8k"),
         ("plan_adam_eps1e17_16k_bs128", "plan_muon_eps1e17_16k_bs128", "bs128"),
         ("sm_adamw_eps1e17_16k_bs256", "sm_muon_eps1e17_16k_bs256", "anchor")]
print(f"\n{'pair':8s} {'ms adamw-muon':>14s} {'MAGIC adamw-muon':>18s}   agree?")
for a, b, label in PAIRS:
    if a in by and b in by:
        dms = by[a][0] - by[b][0]
        dmg = by[a][1] - by[b][1]
        print(f"{label:8s} {dms:+14.4f} {dmg:+18.4f}   "
              f"{'yes' if (dms > 0) == (dmg > 0) else 'NO'}")
