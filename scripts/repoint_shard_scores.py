"""Repoint each filter shard config at its OWN sliced scores dir.

shard_filter.py slices only the query dataset. Its docstring justifies that with
"scores ... only read by validate_scores (the bank path), not by
tail_filter_retrain" -- but a filter whose random control comes from a bank DOES
go through validate_scores, which asserts one score column per query document:

    ValueError: scores has 20 query columns but the query dataset has 2 documents

That fires after the run has already trained, so every shard burns a retrain
before dying. Slicing scores.bin is what scripts/shard_scores.py is for; this
just points each shard at the slice that already exists.

Usage: repoint_scores.py <run_dir> <source>
"""
import sys
from pathlib import Path

import yaml

run = Path(sys.argv[1])
source = sys.argv[2]

bounds = [(i, i + 2) for i in range(0, 20, 2)]
fixed = skipped = 0
for a, b in bounds:
    cfg = run / f"filter_proponents_{source}_q{a}_{b}.yaml"
    if not cfg.exists():
        print(f"  MISSING config {cfg.name}")
        continue
    slice_dir = run / f"scores_q{a}_{b}"
    if not (slice_dir / "info.json").exists():
        sys.exit(f"missing score slice {slice_dir} -- run shard_scores.py first")

    doc = yaml.safe_load(cfg.read_text())
    v = doc["steps"][0]["validate"]
    before = v.get("scores")
    if before == str(slice_dir):
        print(f"  already correct: {cfg.name}")
        skipped += 1
        continue
    # sanity: the query slice really does hold b-a documents
    qds = v["query"]["dataset"]
    assert qds.endswith(f"query_20_q{a}_{b}.hf"), (cfg.name, qds)
    v["scores"] = str(slice_dir)
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"  {cfg.name}: scores -> {slice_dir.name}")
    fixed += 1

print(f"  fixed={fixed} already_ok={skipped}")
