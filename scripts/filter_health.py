#!/usr/bin/env python3
"""Filter-run health: freshness AND a live process, checked together.

Neither signal alone is sufficient, and both have now produced a wrong call:

  * checkpoint mtime -- barely moves since save_mode became interval, so healthy
    runs looked stalled (seventeen at once)
  * CSV mtime -- keeps a recent timestamp for a while after the process dies, so
    a killed cut row looked live

A run is only healthy if its per-query CSV is fresh AND something on this node
still holds its config. Anything else is reported for what it is.
"""
import glob
import os
import subprocess
import time

FRESH_MIN = 25
now = time.time()
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout

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
    held = f"{d}.yaml" in ps
    run = os.path.basename(os.path.dirname(d))
    src = os.path.basename(d).replace("filter_proponents_", "")
    if held and age < FRESH_MIN:
        state = "running"
    elif held:
        state = "HUNG"          # process alive, no progress
    else:
        state = "dead"          # no process; whatever it wrote is all there is
    rows.append((state, age, n, run, src))

order = {"running": 0, "HUNG": 1, "dead": 2}
rows.sort(key=lambda r: (order[r[0]], r[1]))
print(f"{'state':<9}{'csv age':>8}{'queries':>9}  run / source")
for state, age, n, run, src in rows:
    print(f"{state:<9}{age:7.0f}m{n:>9}  {run} / {src}")
