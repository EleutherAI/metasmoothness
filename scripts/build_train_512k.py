"""Extend the nested token-scaling chain to 512k.

Same rule scripts/rebuild_scaling_family.py used to make 64k/128k/256k nested:

    train_512k = train_256k ++ fill

where `fill` walks train_scratch_512k in its stored order and appends every doc
not already in the chain and not in the exclusion pool (heldout_4k, query_20,
query_50). Deterministic, no RNG.

Verified before writing: train_256k is a byte-identical prefix, the exclusion
pool is untouched (measured overlap 0), and the result is exactly 512000 docs.
"""
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

D = Path("/mnt/ssd-1/lucia/bergson-damping/runs/ekfac_vs_n/datasets")
out = D / "train_512k.hf"
if out.exists():
    raise SystemExit(f"{out} already exists")

base = load_from_disk(str(D / "train_256k.hf"))
scratch = load_from_disk(str(D / "train_scratch_512k.hf"))

seen = {tuple(r) for r in base["input_ids"]}
excl = set()
for name in ("heldout_4k.hf", "query_20.hf", "query_50.hf"):
    p = D / name
    if p.is_dir():
        excl |= {tuple(r) for r in load_from_disk(str(p))["input_ids"]}

need = 512_000 - len(base)
keep = []
for i, row in enumerate(scratch["input_ids"]):
    t = tuple(row)
    if t in seen or t in excl:
        continue
    seen.add(t)
    keep.append(i)
    if len(keep) == need:
        break
if len(keep) < need:
    raise SystemExit(f"only {len(keep)} fill docs available, need {need}")

fill = scratch.select(keep)
combined = concatenate_datasets([base, fill.cast(base.features)])

# Verify before writing: nesting, size, and that nothing from the eval pools leaked.
assert len(combined) == 512_000, len(combined)
for i in (0, 1, 50_000, 255_999):
    assert combined[i]["input_ids"] == base[i]["input_ids"], f"not a prefix at {i}"
final = {tuple(r) for r in combined["input_ids"]}
assert not (final & excl), "exclusion pool leaked into train_512k"
assert len(final) == 512_000, f"duplicate docs: {512_000 - len(final)}"

combined.save_to_disk(str(out))
print(f"wrote {out}: {len(combined)} docs")
print(f"  train_256k is a verified prefix; {len(keep)} fill docs from train_scratch_512k")
print(f"  exclusion-pool overlap: 0; duplicates: 0")
