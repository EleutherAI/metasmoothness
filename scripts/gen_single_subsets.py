"""Emit a one-subset bank shard config for each subset a bank still owes.

A 5-wide shard walks its range sequentially at ~54 min a subset, so the tail of a
nearly-finished bank trickles in over hours while whole nodes sit idle. Every
shard reads the same subsets.json, so which pair retrains a given subset does not
change what is removed -- only how long the bank takes to close.

Usage: gen_single_subsets.py <run_id> <subset> [<subset> ...]
"""
import sys
from pathlib import Path

import yaml

ROOT = Path("/mnt/ssd-2/lucia/paper_runs/experiments") / sys.argv[1]
BANK = ROOT / "bank_from_filter"
template = next(ROOT.glob("bank_shard_*_*.yaml"), None)
if template is None:
    sys.exit(f"no bank_shard template in {ROOT}")

for s in (int(a) for a in sys.argv[2:]):
    if (BANK / "retrained" / f"subset_{s}").is_dir():
        print(f"  subset_{s} already retrained, skipping")
        continue
    doc = yaml.safe_load(template.read_text())
    v = doc["steps"][0]["validate"]
    subsets = Path(v["subsets"])
    if not subsets.is_file():
        sys.exit(f"refusing: {subsets} missing -- shards must share one subsets.json")
    v["subset_start"], v["subset_stop"] = s, s + 1
    out = ROOT / f"bank_shard_{s}_{s+1}.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"  wrote {out.name}")
