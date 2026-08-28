"""Record EK-FAC LDS values that were computable from banks already on disk.

Four rows had a finished bank and finished EK-FAC scores but no ekfac_lds in
experiments.csv, so they were invisible to every correlation and to
reclaim_banks' "statistics extracted" test. Nothing needed recomputing -- the
ground truth was sitting in each row's validation.csv.

n_subsets is recorded per row and is NOT always 100: three of these banks stopped
early. An LDS on 57 subsets is a valid estimate with a wider interval, not a
broken one, and the column is what tells a later reader which is which.
"""
import csv
import shutil
from pathlib import Path

EXP = Path("/mnt/ssd-2/lucia/metasmoothness/experiments.csv")

VALUES = {
    "london16k_bs256_muon":        (0.3156, 0.2736, 0.3545, 100),
    "london16k_bs256_adamw":       (0.3165, 0.2669, 0.3651, 74),
    "gpt2medium_16k_bs32":         (0.3419, 0.2768, 0.4024, 65),
    "plan_muon_eps1e17_64k_bs256": (0.4591, 0.3861, 0.5210, 57),
}

rows = list(csv.DictReader(open(EXP)))
cols = list(rows[0].keys())
touched = []
for r in rows:
    v = VALUES.get(r["run_id"])
    if not v:
        continue
    if (r.get("ekfac_lds") or "").strip():
        print(f"  {r['run_id']}: already has ekfac_lds {r['ekfac_lds']}, leaving alone")
        continue
    r["ekfac_lds"], r["ekfac_ci_lo"], r["ekfac_ci_hi"] = f"{v[0]:.4f}", f"{v[1]:.4f}", f"{v[2]:.4f}"
    r["ekfac_n_subsets"] = str(v[3])
    touched.append((r["run_id"], v, (r.get("filter_ekfac_delta") or "").strip()))

if touched:
    shutil.copy(EXP, EXP.with_suffix(".csv.bak"))
    with open(EXP, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
for run, v, delta in touched:
    print(f"  {run:30s} lds={v[0]:.4f} n_subsets={v[3]:3d}  "
          f"{'delta ' + delta[:7] + ' -> joins the correlation now' if delta else 'no delta yet -- needs a filter run'}")
print(f"  wrote {len(touched)} row(s)")
