#!/usr/bin/env python3
"""Freshness of each in-progress filter run, keyed on the per-query CSV.

The earlier check used the checkpoint directory's mtime, which was a fair proxy
while runs wrote a full sqrt trajectory per query. Since save_mode moved to
interval that directory barely changes, so it now reports healthy runs as
stalled. filter_proponents.csv gains a row per finished query, which is the
thing that actually tracks progress.
"""
import glob
import os
import time

now = time.time()
rows = []
for d in glob.glob("/mnt/ssd-*/lucia/paper_runs/experiments/*/filter_proponents_*/"):
    d = d.rstrip("/")
    if os.path.exists(os.path.join(d, "filter_summary.csv")):
        continue
    f = os.path.join(d, "filter_proponents.csv")
    if not os.path.exists(f):
        continue
    n = sum(1 for _ in open(f)) - 1
    age = (now - os.path.getmtime(f)) / 60
    run = os.path.basename(os.path.dirname(d))
    src = os.path.basename(d).replace("filter_proponents_", "")
    rows.append((age, n, run, src))

rows.sort()
print(f"{'csv age':>8}{'queries':>9}  state    run / source")
for a, n, r, s in rows:
    state = "live" if a < 25 else "STALLED"
    print(f"{a:7.0f}m{n:>9}  {state:<8} {r} / {s}")
