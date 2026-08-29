"""Every piece of MAGIC data on disk, and where it is."""
import csv
import glob
import os

E = "/mnt/ssd-2/lucia/paper_runs/experiments"
ROOT = "/mnt/ssd-2/lucia/metasmoothness"
rows = list(csv.DictReader(open(ROOT + "/experiments.csv")))

print("  RECORDED IN experiments.csv")
n = 0
for r in sorted(rows, key=lambda r: int((r.get("steps") or "0") or 0)):
    lds, d = (r.get("magic_lds") or "").strip(), (r.get("filter_magic_delta") or "").strip()
    if not (lds or d):
        continue
    n += 1
    print("    %-34s steps=%-5s magic_lds=%-9s delta=%-9s"
          % (r["run_id"], r.get("steps", ""), lds[:8] or "-", d[:8] or "-"))
print(f"    {n} row(s) with MAGIC results recorded")

print("  FINISHED MAGIC SCORES ON DISK")
n = 0
for d in sorted(glob.glob(E + "/*/")):
    run = os.path.basename(d.rstrip("/"))
    for sub in ("", "magic_scores", "magic_scores_ssd2", "magic_scores_only", "magic_scores_timing"):
        p = os.path.join(d, sub, "scores", "info.json")
        if os.path.isfile(p):
            n += 1
            print("    %-34s %s" % (run, sub or "(row root)"))
print(f"    {n} finished score set(s)")

print("  PARTIAL PER-QUERY WORK PRESERVED (resumes, never recomputed)")
tot = 0
for d in sorted(glob.glob(E + "/*/")):
    run = os.path.basename(d.rstrip("/"))
    best = {}
    for pq in glob.glob(os.path.join(d, "**", "per_query"), recursive=True):
        k = len(glob.glob(pq + "/q*.pt"))
        if k:
            best[os.path.relpath(pq, d)] = k
    if best:
        tot += sum(best.values())
        main = sorted(best.items(), key=lambda kv: -kv[1])[0]
        print("    %-34s %d query file(s) across %d dir(s), largest %s=%d"
              % (run, sum(best.values()), len(best), main[0], main[1]))
print(f"    {tot} per-query file(s) total")
