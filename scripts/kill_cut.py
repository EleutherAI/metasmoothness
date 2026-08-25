#!/usr/bin/env python3
"""Stop every process working on a cut row, queue scripts included.

Killing only the bergson worker is not enough: the slot script simply advances
to the next job in its list, which for a cut row is usually the other scorer's
arm of the same cut row. The queue has to go too.

Deletes nothing.
"""
import os
import signal
import subprocess
import sys

sys.path.insert(0, "/mnt/ssd-2/lucia/metasmoothness/scripts")
from axes import is_cut  # noqa: E402

ps = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
killed = []
for line in ps.splitlines():
    parts = line.split(None, 1)
    if len(parts) != 2 or not parts[0].isdigit():
        continue
    pid, args = int(parts[0]), parts[1]
    if "kill_cut" in args:
        continue
    if "-m bergson" not in args and "run_filter_slot.sh" not in args:
        continue
    # which run_id does this process concern?
    hit = None
    for token in args.replace("/", " ").split():
        if token.startswith(("plan_", "sm_", "tune_")) and is_cut(token):
            hit = token
            break
    if hit:
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append((pid, hit))
        except OSError:
            pass

for pid, hit in killed:
    print(f"  killed {pid}  (cut: {hit})")
print(f"stopped {len(killed)} process(es) on cut rows")
