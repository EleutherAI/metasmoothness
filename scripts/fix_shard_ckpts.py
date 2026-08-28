"""Give each MAGIC query shard its own checkpoints directory of symlinks.

First attempt symlinked the shard's whole `checkpoints` directory at the main
run's. That is wrong: bergson writes a temporary per-query checkpoint into that
directory and deletes it afterwards, so N shards sharing one directory delete each
other's temp files and die with

    FileNotFoundError: .../magic_scores_ssd2_q6_8/checkpoints/step_1...

Instead make `checkpoints` a REAL directory containing one symlink per trajectory
entry. The trajectory is shared read-only and costs nothing (the alternative is
copying ~100 GB per shard onto a volume with 472 GB free), while every file the
shard creates lands in its own directory. Deleting a symlink removes the link, not
the target, so a shard cleaning up cannot damage the trajectory.
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path("/mnt/ssd-2/lucia/paper_runs/experiments") / sys.argv[1]
main_ckpts = ROOT / sys.argv[2] / "checkpoints"
if not main_ckpts.is_dir():
    sys.exit(f"no trajectory at {main_ckpts}")
entries = sorted(p for p in main_ckpts.iterdir() if p.name != "per_query")

# Only the shards named on the command line, and only directories -- the glob also
# matches each shard's .yaml. Never convert a RUNNING shard: replacing the
# directory it is reading from would kill it.
wanted = set(sys.argv[3:])
if not wanted:
    sys.exit("name the shard suffixes to convert, e.g. q6_8 q8_10")

fixed = 0
for shard in sorted(ROOT.glob(f"{sys.argv[2]}_q*_*")):
    if not shard.is_dir() or shard.name.split("_")[-2] + "_" + shard.name.split("_")[-1] not in wanted:
        continue
    ck = shard / "checkpoints"
    if ck.is_symlink():
        ck.unlink()
    elif ck.is_dir() and any(p.is_symlink() for p in ck.iterdir()):
        continue                      # already converted
    elif ck.is_dir():
        shutil.rmtree(ck)
    ck.mkdir(parents=True, exist_ok=True)
    for e in entries:
        link = ck / e.name
        if not link.exists():
            link.symlink_to(e)
    fixed += 1
    print(f"  {shard.name}: {len(entries)} trajectory entries linked individually")
print(f"  converted {fixed} shard(s); each now owns its checkpoints directory")
