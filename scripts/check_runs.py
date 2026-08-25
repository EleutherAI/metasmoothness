"""Health for NON-filter runs: scoring, ms, bank builds, experiments.

filter_health.py covers filter_proponents_* only. Everything else -- EK-FAC and
MAGIC scoring, ms, bank builds -- had no check at all, which is how two 64k EK-FAC
scoring jobs sat dead for five hours after an NCCL watchdog abort while the fleet
kept reporting as saturated.

A run is only interesting if it is INCOMPLETE and not progressing. Completion is
per kind -- an old log on a finished run is not a problem, and treating it as one
buries the real failures (the first version of this flagged 67 of 78 runs).

Liveness is log freshness plus the claim directory, the fleet-wide signal, since
a node-local `ps` cannot see a job on another node.

    python scripts/check_runs.py [--stale-min 45] [--all]
"""
import argparse
import glob
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--stale-min", type=float, default=45)
ap.add_argument("--all", action="store_true", help="include completed runs")
a = ap.parse_args()

CLAIMS = "/mnt/ssd-2/lucia/paper_runs/_claims"
now = time.time()

# log stem -> (label, path relative to the run dir that exists once it is done)
KINDS = {
    "ekfac":        ("ekfac score",    "ekfac_scores/scores"),
    "magic_scores": ("magic score",    "base_traj/scores"),
    "ms":           ("metasmoothness", "ms/metasmoothness.json"),
    # A bank build writes validation.csv at the START and appends to it, so its
    # existence means nothing. Count retrained subsets instead -- that was
    # hiding two live bank builds from this report entirely.
    "bank_build":   ("bank build",     "bank_from_filter/retrained/subset_99"),
    "experiment":   ("experiment",     "retrained/subset_99"),
    "base_traj":    ("trajectory",     "base_traj/model"),
}

claims = {os.path.basename(c) for c in glob.glob(os.path.join(CLAIMS, "*"))}

rows = []
for log in sorted(glob.glob("/mnt/ssd-*/lucia/paper_runs/experiments/*/*.log")):
    stem = os.path.basename(log)[:-4]
    if stem not in KINDS:
        continue
    label, done_rel = KINDS[stem]
    run_dir = os.path.dirname(log)
    run = os.path.basename(run_dir)

    done = os.path.exists(os.path.join(run_dir, done_rel))
    if done and not a.all:
        continue

    age = (now - os.path.getmtime(log)) / 60
    owner = next((c for c in claims if c.startswith(run + "__")), None)
    if owner:
        try:
            with open(os.path.join(CLAIMS, owner, "host")) as fh:
                owner = fh.read().strip()
        except OSError:
            owner = "?"

    if done:
        state = "done"
    elif age < a.stale_min:
        state = "running"
    elif owner:
        state = "STALE"      # claimed but not writing -- may be hung
    else:
        state = "DEAD"       # nobody owns it and it stopped short
    rows.append((state, age, label, run, owner or "-"))

order = {"running": 0, "done": 1, "STALE": 2, "DEAD": 3}
rows.sort(key=lambda r: (order[r[0]], -r[1]))
print(f"{'state':<9}{'log age':>9}  {'kind':<16}{'owner':<14}run")
for state, age, label, run, owner in rows:
    print(f"{state:<9}{age:8.0f}m  {label:<16}{owner:<14}{run}")

bad = [r for r in rows if r[0] in ("STALE", "DEAD")]
print(f"\n{len(rows)} incomplete runs, {len(bad)} not progressing")
