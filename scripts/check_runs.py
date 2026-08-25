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

# A run whose artifact was consumed and then deleted is finished, not dead. The
# derived result lives in experiments.csv, so consult it: plan_adam_eps1e17_16k_bs64
# reported DEAD every sweep for 60 hours because its EK-FAC scores had been used
# (ekfac_lds 0.4239) and cleaned up. That noise is what hides a real failure.
DERIVED = {"ekfac score": "ekfac_lds", "magic score": "magic_lds"}
_recorded = {}
try:
    import csv as _csv
    for _r in _csv.DictReader(open("/mnt/ssd-2/lucia/metasmoothness/experiments.csv")):
        _recorded[_r["run_id"]] = _r
except OSError:
    pass


def result_recorded(run, label):
    col = DERIVED.get(label)
    return bool(col and (_recorded.get(run) or {}).get(col))


rows = []
for log in sorted(glob.glob("/mnt/ssd-*/lucia/paper_runs/experiments/*/*.log")):
    stem = os.path.basename(log)[:-4]
    if stem not in KINDS:
        continue
    label, done_rel = KINDS[stem]
    run_dir = os.path.dirname(log)
    run = os.path.basename(run_dir)

    done = os.path.exists(os.path.join(run_dir, done_rel)) or result_recorded(run, label)
    if done and not a.all:
        continue

    age = (now - os.path.getmtime(log)) / 60
    # A run has several claims at once -- magicscore, ekfacscore, experiment,
    # bankbuild -- so matching on the run id alone lets a LIVE claim of one kind
    # mask a DEAD run of another. That is what hid a dead 64k EK-FAC scoring
    # behind that row's healthy MAGIC scoring: it reported STALE (claimed but
    # quiet) instead of DEAD (nobody owns it), which reads as "wait" rather than
    # "relaunch".
    CLAIM_SUFFIX = {
        "ekfac score": "ekfacscore",
        "magic score": "magicscore",
        "metasmoothness": "ms",
        "bank build": "bankbuild",
        "experiment": "experiment",
        "trajectory": "traj",
    }
    _suffix = CLAIM_SUFFIX.get(label)
    if _suffix:
        owner = f"{run}__{_suffix}" if f"{run}__{_suffix}" in claims else None
    else:
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

# The same run can exist on ssd-1 and ssd-2, and the glob finds both. Report once.
_seen = set()
_rows = []
for _row in rows:
    _key = (_row[2], _row[3])       # (kind, run)
    if _key in _seen:
        continue
    _seen.add(_key)
    _rows.append(_row)
rows = _rows

order = {"running": 0, "done": 1, "STALE": 2, "DEAD": 3}
rows.sort(key=lambda r: (order[r[0]], -r[1]))
print(f"{'state':<9}{'log age':>9}  {'kind':<16}{'owner':<14}run")
for state, age, label, run, owner in rows:
    print(f"{state:<9}{age:8.0f}m  {label:<16}{owner:<14}{run}")

bad = [r for r in rows if r[0] in ("STALE", "DEAD")]
print(f"\n{len(rows)} incomplete runs, {len(bad)} not progressing")
