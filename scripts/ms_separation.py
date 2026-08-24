"""Does ms separate the collapsed rows from the healthy ones?

A rank correlation over all rows is the wrong statistic here: ms is confined to a
0.007-wide band for every gently-tuned config, so 11 of 13 ranks are noise and
Spearman reports ~0 regardless of what happens at the extremes. What the paper
actually needs to know is whether ms moves when attributability breaks.
"""
import csv

rows = [r for r in csv.DictReader(open("/mnt/ssd-2/lucia/metasmoothness/experiments.csv"))
        if r["metasmoothness"].strip() and r["magic_lds"].strip()]
rows.sort(key=lambda r: float(r["metasmoothness"]))

print(f"{'run':34s} {'ms':>8s} {'MAGIC':>8s}")
for r in rows:
    print(f"{r['run_id'][:34]:34s} {float(r['metasmoothness']):8.4f} "
          f"{float(r['magic_lds']):8.4f}")

low = [r for r in rows if float(r["metasmoothness"]) < 0.95]
high = [r for r in rows if float(r["metasmoothness"]) >= 0.95]
print()
print(f"ms < 0.95  (n={len(low)}):  MAGIC "
      f"{min(float(r['magic_lds']) for r in low):.4f} - "
      f"{max(float(r['magic_lds']) for r in low):.4f}")
print(f"ms >= 0.95 (n={len(high)}): MAGIC "
      f"{min(float(r['magic_lds']) for r in high):.4f} - "
      f"{max(float(r['magic_lds']) for r in high):.4f}")
worst_high = min(float(r["magic_lds"]) for r in high)
best_low = max(float(r["magic_lds"]) for r in low)
print(f"\nseparation: {'CLEAN' if best_low < worst_high else 'OVERLAPPING'} "
      f"(best low-ms MAGIC {best_low:.4f} vs worst high-ms MAGIC {worst_high:.4f})")
