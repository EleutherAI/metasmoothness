"""Find rows whose LDS is computable from a bank already on disk.

The first pass required a 100-subset bank and found four rows nobody had
registered. A smaller bank still gives a real LDS -- just a wider interval, and
n_subsets records which -- so this lowers the bar to 20 and reports what is left.

Reports, per row: subsets retrained, whether the ground-truth validation CSV
covers them, whether EK-FAC scores exist, and what is already in experiments.csv.
Only rows missing ekfac_lds but having everything needed to compute it are new.
"""
import csv
import glob
import os

E = ["/mnt/ssd-2/lucia/paper_runs/experiments", "/mnt/ssd-1/lucia/paper_runs/experiments"]
rows = {r["run_id"]: r for r in csv.DictReader(open("/mnt/ssd-2/lucia/metasmoothness/experiments.csv"))}

seen = set()
found = []
for root in E:
    for d in sorted(glob.glob(root + "/*/")):
        run = os.path.basename(d.rstrip("/"))
        if run in seen:
            continue
        best = None
        for sub in ("", "bank_from_filter"):
            b = os.path.join(d, sub) if sub else d
            n = len(glob.glob(os.path.join(b, "retrained", "subset_*")))
            if n >= 20 and (best is None or n > best[1]):
                best = (b, n)
        if not best:
            continue
        seen.add(run)
        bank, n = best
        val = next((v for v in ("validation_merged.csv", "validation.csv")
                    if os.path.isfile(os.path.join(bank, v))), None)
        scores = os.path.isfile(os.path.join(d, "ekfac_scores", "scores", "info.json"))
        r = rows.get(run, {})
        has_lds = bool((r.get("ekfac_lds") or "").strip())
        has_delta = bool((r.get("filter_ekfac_delta") or "").strip())
        found.append((run, n, val, scores, has_lds, has_delta, run in rows))

print("  %-32s %5s %-22s %-7s %-8s %-7s" % ("run", "subs", "ground truth", "scores", "has lds", "delta"))
new = 0
for run, n, val, scores, has_lds, has_delta, registered in sorted(found, key=lambda t: -t[1]):
    tag = ""
    if not has_lds and val and scores:
        tag = "  <== LDS computable now" if registered else "  <== computable, row not registered"
        new += 1
    print("  %-32s %5d %-22s %-7s %-8s %-7s%s"
          % (run, n, val or "NONE", "yes" if scores else "no",
             "yes" if has_lds else "no", "yes" if has_delta else "no", tag))
print(f"  {new} row(s) with an uncomputed LDS")
