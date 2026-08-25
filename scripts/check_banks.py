"""Check every retrain bank for truncated files before trusting or publishing it.

Three zero-byte config.json files under retrained/ silently killed four filter
runs: the proponent phase never reads the bank, so a job always got to exactly
20/20 queries and then died at the random control. The weights were fine; only
the architecture file was lost, most likely to a truncated write while the volume
was full.

Run this before a publish, and after any period where ssd-1 or ssd-2 hit 100%.

    python scripts/check_banks.py [--fix]

--fix repairs a truncated config.json from a sibling subset in the same row, and
only when every good config in that row is byte-identical -- which it must be,
since all subsets of a run share an architecture.
"""
import argparse
import glob
import hashlib
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--fix", action="store_true")
a = ap.parse_args()

MIN_CONFIG = 10        # a real config.json is ~1 KB
MIN_WEIGHTS = 1 << 20  # a real model.safetensors is hundreds of MB

bad_cfg, bad_w, fixed, refused = [], [], 0, 0

for retrained in sorted(glob.glob("/mnt/ssd-*/lucia/paper_runs/experiments/*/retrained")):
    row = os.path.basename(os.path.dirname(retrained))

    good, digests = None, set()
    for c in glob.glob(os.path.join(retrained, "subset_*", "config.json")):
        if os.path.getsize(c) >= MIN_CONFIG:
            digests.add(hashlib.md5(open(c, "rb").read()).hexdigest())
            good = good or c

    for c in sorted(glob.glob(os.path.join(retrained, "subset_*", "config.json"))):
        if os.path.getsize(c) >= MIN_CONFIG:
            continue
        bad_cfg.append(c)
        if not a.fix:
            continue
        if len(digests) == 1 and good:
            with open(good, "rb") as fh:
                data = fh.read()
            with open(c, "wb") as fh:
                fh.write(data)
            os.chmod(c, 0o644)
            fixed += 1
        else:
            print(f"  REFUSED {c}: {len(digests)} distinct sibling configs", file=sys.stderr)
            refused += 1

    for w in sorted(glob.glob(os.path.join(retrained, "subset_*", "model.safetensors"))):
        if os.path.getsize(w) < MIN_WEIGHTS:
            bad_w.append(w)

for c in bad_cfg:
    print(f"  truncated config : {c}")
for w in bad_w:
    print(f"  truncated weights: {w}   NOT REPAIRABLE -- the subset must be retrained")

print(f"{len(bad_cfg)} truncated config.json, {len(bad_w)} truncated model.safetensors")
if a.fix:
    print(f"repaired {fixed}, refused {refused}")
sys.exit(1 if (bad_w or (bad_cfg and not a.fix)) else 0)
