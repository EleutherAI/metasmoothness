"""Build train_1M.hf as train_512k ++ fill, preserving the nested chain.

Why not just use build_pool.py's output directly: it excludes ONLY query_50,
whereas the existing chain was built excluding heldout_4k, query_20 and query_50.
So a fresh pool keeps chunks the chain skipped and every index after the first
skip shifts. Measured: train_1000k.hf matches train_4k at indices 0/1/1333 but
diverges by 3998, and diverges from train_512k by 170666. It is a valid chunk
pool, just not a prefix-superset of the chain.

So use it as the SCRATCH POOL and apply build_train_512k.py's rule instead:

    train_1M = train_512k ++ (chunks of the pool not already in the chain
                              and not in heldout_4k / query_20 / query_50)

That is the same deterministic, RNG-free rule that produced the 512k rung, so the
result is nested by construction. Everything is verified before writing.
"""
import sys
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

DS = Path("/mnt/ssd-2/lucia/datasets_local")
SSD1 = Path("/mnt/ssd-1/lucia/bergson-damping/runs/ekfac_vs_n/datasets")
OUT = DS / "train_1M.hf"
TARGET = 1_000_000

if OUT.exists():
    sys.exit(f"{OUT} already exists")

base = load_from_disk(str(DS / "train_512k.hf"))
pool = load_from_disk(str(DS / "train_1000k.hf"))
print(f"  base train_512k: {len(base):,}   pool train_1000k: {len(pool):,}")
assert len(base) == 512_000, len(base)

# Exclusion pool. heldout_4k lives on ssd-1; reading ssd-1 is fine, D23 only
# forbids writing there. Refuse to proceed if it cannot be read -- silently
# skipping it is how eval data leaks into a training set.
excl = set()
for name, root in (("heldout_4k", SSD1), ("query_20", DS), ("query_50", DS)):
    p = root / f"{name}.hf"
    if not p.is_dir():
        p = (DS if root is SSD1 else SSD1) / f"{name}.hf"
    if not p.is_dir():
        sys.exit(f"cannot find exclusion set {name}.hf -- refusing to build")
    d = load_from_disk(str(p))
    excl |= {tuple(r) for r in d["input_ids"]}
    print(f"  exclusion {name}: {len(d)} chunks (from {p.parent})")

seen = {tuple(r) for r in base["input_ids"]}
print(f"  distinct chunks in base: {len(seen):,}")

need = TARGET - len(base)
keep = []
for i, row in enumerate(pool["input_ids"]):
    t = tuple(row)
    if t in seen or t in excl:
        continue
    seen.add(t)
    keep.append(i)
    if len(keep) == need:
        break
print(f"  fill selected: {len(keep):,} of {need:,} needed")
if len(keep) < need:
    sys.exit(f"pool exhausted: only {len(keep)} usable fill chunks, need {need}")

fill = pool.select(keep)
combined = concatenate_datasets([base, fill.cast(base.features)])

# Verify before writing.
assert len(combined) == TARGET, len(combined)
for i in (0, 1, 100_000, 255_999, 511_998, 511_999):
    assert combined[i]["input_ids"] == base[i]["input_ids"], f"not a prefix at {i}"
final = {tuple(r) for r in combined["input_ids"]}
assert len(final) == TARGET, f"duplicates: {TARGET - len(final)}"
assert not (final & excl), "exclusion pool leaked into train_1M"

combined.save_to_disk(str(OUT))
print(f"  wrote {OUT}: {len(combined):,} chunks")
print(f"  train_512k verified as a byte-identical prefix; {len(keep):,} fill chunks appended")
print("  exclusion-pool overlap 0; duplicates 0")
