"""Mirror every run config that produced data into the repo.

gen_experiment_run.py and gen_tuning_run.py already commit their configs
(configs/experiments, configs/tuning). The ms probes, EK-FAC scoring, tail-filter
validation, bank slices and diagnostic probes do not, so the exact config behind
several recorded numbers lives only on scratch disks.
"""
import shutil
from pathlib import Path

REPO = Path("/mnt/ssd-2/lucia/metasmoothness")
ROOTS = [Path("/mnt/ssd-2/lucia/paper_runs/experiments"),
         Path("/mnt/ssd-1/lucia/paper_runs/experiments")]
DIAG = Path("/mnt/ssd-2/lucia/paper_runs/diagnostics")

# (glob, destination subdir) -- destination name is <run>__<file>
JOBS = [
    ("*/ms.yaml", "ms"),
    ("*/ekfac.yaml", "ekfac"),
    ("*/filter_proponents_*.yaml", "filter"),
    ("*/slice_*.yaml", "slices"),
]

copied, seen = 0, set()
for glob, sub in JOBS:
    dest = REPO / "configs" / sub
    dest.mkdir(parents=True, exist_ok=True)
    for root in ROOTS:
        for src in sorted(root.glob(glob)):
            run = src.parent.name
            key = (sub, run, src.name)
            if key in seen:          # migrated runs appear under both roots
                continue
            seen.add(key)
            out = dest / f"{run}__{src.name}"
            if out.exists() and out.read_bytes() == src.read_bytes():
                continue
            shutil.copy2(src, out)
            copied += 1

# Diagnostic probes: their own directory, name is already descriptive.
dest = REPO / "configs" / "diagnostics"
dest.mkdir(parents=True, exist_ok=True)
for src in sorted(DIAG.glob("*/ms.yaml")):
    out = dest / f"{src.parent.name}__ms.yaml"
    if not (out.exists() and out.read_bytes() == src.read_bytes()):
        shutil.copy2(src, out)
        copied += 1

print(f"mirrored {copied} configs")
for sub in ("ms", "ekfac", "filter", "slices", "diagnostics"):
    d = REPO / "configs" / sub
    print(f"  configs/{sub:12s} {len(list(d.glob('*.yaml')))} files")
