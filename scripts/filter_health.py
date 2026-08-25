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

# `ps` only sees THIS node. Filter runs are spread across ten, so a node-local
# check reports every job running elsewhere as dead -- it called 7 of 8 dead while
# 10 filter processes were alive across the fleet. Acting on that would clear a
# live run's output.
#
# The claim directory is the fleet-wide signal: run_filter_slot.sh mkdir-s
# _claims/<run>__<source> before starting and removes it on exit, so its presence
# means some node owns the job. Local ps still counts, for the node you happen to
# be on.
CLAIMS = "/mnt/ssd-2/lucia/paper_runs/_claims"

# The claim name must be exactly <run_id>__<source>, source being magic or ekfac.
# A hand-launched run that invents its own suffix (e.g. __ekfacfilter) is
# invisible both here and to run_filter_slot's stale-output guard, which is how
# two live 15/20-query jobs came to read as dead with no owner.


def claim_host(run, source):
    """Hostname holding this job's claim, or None."""
    try:
        with open(os.path.join(CLAIMS, f"{run}__{source}", "host")) as fh:
            return fh.read().strip()
    except OSError:
        return None

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
    owner = claim_host(run, src)
    held = owner is not None or f"{d}.yaml" in ps
    if held and age < FRESH_MIN:
        state = "running"
    elif held:
        state = "HUNG"          # process alive, no progress
    else:
        state = "dead"          # no process; whatever it wrote is all there is
    rows.append((state, age, n, run, src, owner or "-"))

order = {"running": 0, "HUNG": 1, "dead": 2}
rows.sort(key=lambda r: (order[r[0]], r[1]))
print(f"{'state':<9}{'csv age':>8}{'queries':>9}  {'owner':<14}run / source")
for state, age, n, run, src, owner in rows:
    print(f"{state:<9}{age:7.0f}m{n:>9}  {owner:<14}{run} / {src}")
