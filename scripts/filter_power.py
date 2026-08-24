"""Recipe-independent summaries of tail-filter results.

Raw `filter_change` is a query-loss delta in nats, and those shrink as the
training set grows -- so the raw number is NOT comparable across dataset sizes.
The fix is the one ROAR uses: express the filter's damage against random
removals of the SAME size on the SAME model. bergson gives us a matched control
(the row's own leave-k-out bank), so two unitless statistics fall out per query:

  z          = (filter_change - random_mean) / random_sd
  percentile = 1 - rank/(random_n + 1)   [1.0 = more damaging than every random]

`percentile` is the one to lean on across recipes: it is nonparametric, so it
cannot be distorted by the loss-delta scale changing with dataset size.

Writes data/filter_power.csv (per query, raw + normalized) and prints a summary.
"""
import csv
import glob
import os

import numpy as np

OUT = "/mnt/ssd-2/lucia/metasmoothness/data/filter_power.csv"
ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments",
         "/mnt/ssd-1/lucia/paper_runs/experiments"]

rows = []
for root in ROOTS:
    for summ in sorted(glob.glob(f"{root}/*/filter_*_*/filter_summary.csv")):
        run_dir = os.path.dirname(summ)
        run = os.path.basename(os.path.dirname(run_dir))
        tag = os.path.basename(run_dir)          # filter_proponents_magic
        parts = tag.split("_")
        method, source = parts[1], parts[2]
        with open(summ) as f:
            for r in csv.DictReader(f):
                fc = float(r["filter_change"])
                rm = float(r["random_mean"])
                sd = float(r["random_sd"])
                n = int(r["random_n"])
                rank = int(r["rank"])
                rows.append(dict(
                    run=run, method=method, source=source,
                    query=int(r["query"]), n_removed=int(r["n_removed"]),
                    filter_change=fc, random_mean=rm, random_sd=sd,
                    random_n=n, rank=rank,
                    z=(fc - rm) / sd if sd else float("nan"),
                    percentile=1.0 - rank / (n + 1),
                ))

if not rows:
    raise SystemExit("no filter_summary.csv yet")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {OUT} ({len(rows)} query rows)\n")

print(f"{'run':30s} {'src':6s} {'n':>3s} {'raw mean':>10s} {'z mean':>8s} "
      f"{'pctile':>8s} {'beat-all':>9s}")
key = lambda r: (r["run"], r["source"])
for k in sorted({key(r) for r in rows}):
    sub = [r for r in rows if key(r) == k]
    z = np.array([r["z"] for r in sub])
    p = np.array([r["percentile"] for r in sub])
    raw = np.array([r["filter_change"] for r in sub])
    beat = float((p >= 1.0).mean())
    print(f"{k[0][:30]:30s} {k[1]:6s} {len(sub):3d} {raw.mean():10.5f} "
          f"{z.mean():8.3f} {p.mean():8.3f} {beat:9.2f}")
