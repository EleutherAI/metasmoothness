"""Rebuild the token-scaling dataset family as one nested chain, and fix heldout.

Problem (measured 2026-08-20): train_4k .. train_32k are nested prefixes of one
shuffled order, but train_64k shares only ~4k docs with train_32k — the 64k/128k/256k
family is a different draw. And heldout_4k, built disjoint from train_4k..128k, is
fully contained in train_256k.

Rebuild rule (deterministic, no RNG):
  train_64k'  = train_32k  ++ fill
  train_128k' = train_64k'  ++ fill
  train_256k' = train_128k' ++ fill
where `fill` walks the OLD 64k/128k/256k datasets in their stored order and appends
every doc not already in the chain and not in the exclusion pool
(heldout_4k ∪ query_50 ∪ query_20). train_4k..32k, heldout_4k and the query sets are
byte-identical to before, so every existing bank, score and heldout number stays valid.

Writes train_{64,128,256}k.hf in place (old versions moved to *.old_disjoint/) and
verifies: chain nesting, exclusion-pool disjointness, doc identity/lengths, sizes.
"""

import argparse
import shutil
from pathlib import Path
from typing import Sequence

from datasets import Dataset, concatenate_datasets, load_from_disk

_ap = argparse.ArgumentParser()
_ap.add_argument("--datasets_dir",
                 default="/mnt/ssd-1/lucia/bergson-damping/runs/ekfac_vs_n/datasets")
BASE = Path(_ap.parse_args().datasets_dir)


def load(path: Path) -> Dataset:
    ds = load_from_disk(str(path))
    assert isinstance(ds, Dataset), f"{path} is a DatasetDict, expected a flat Dataset"
    return ds


def key(ids):
    # Full-sequence hash: first-k-token keys could alias distinct docs that share a prefix.
    return hash(tuple(ids))


def keys(ds):
    return [key(x) for x in ds["input_ids"]]


def main():
    chain = load(BASE / "train_32k.hf")
    chain_keys = set(keys(chain))
    assert len(chain_keys) == 32000, "train_32k has duplicate docs?"

    exclude = set()
    for name in ["heldout_4k", "query_50", "query_20"]:
        exclude |= set(keys(load(BASE / f"{name}.hf")))
    print(f"exclusion pool: {len(exclude)} docs (heldout + queries)")

    # Deterministic fill pool: old 64k, then old 128k, then old 256k, stored order.
    fills = []
    for name in ["train_64k", "train_128k", "train_256k"]:
        old = BASE / f"{name}.hf.old_disjoint"
        fills.append(load(old if old.exists() else BASE / f"{name}.hf"))

    fill_rows, seen = [], set(chain_keys)
    for ds in fills:
        ks = keys(ds)
        for i, k in enumerate(ks):
            if k in seen or k in exclude:
                continue
            seen.add(k)
            fill_rows.append((ds, i))
    print(f"fill pool: {len(fill_rows)} unique admissible docs")
    need = 256_000 - 32_000
    assert len(fill_rows) >= need, f"not enough fill docs: {len(fill_rows)} < {need}"

    # Materialise the fill in pool order, then cut the chain at each size.
    by_ds = {}
    for ds, i in fill_rows[:need]:
        by_ds.setdefault(id(ds), (ds, []))[1].append(i)
    parts: Sequence[Dataset] = [chain] + [ds.select(idxs) for ds, idxs in by_ds.values()]
    full = concatenate_datasets(list(parts))
    assert len(full) == 256_000

    # Stage first: the fill pool is mmapped from the OLD files, so writing over them
    # in place is both self-overwrite (datasets refuses) and would corrupt the pool.
    for n in [64_000, 128_000, 256_000]:
        name = f"train_{n // 1000}k"
        full.select(range(n)).save_to_disk(str(BASE / f"{name}.hf.new"))
        print(f"staged {name}.hf.new: {n} docs")
    # Swap only after every stage landed.
    for n in [64_000, 128_000, 256_000]:
        name = f"train_{n // 1000}k"
        out, old = BASE / f"{name}.hf", BASE / f"{name}.hf.old_disjoint"
        if not old.exists():
            shutil.move(str(out), str(old))
        shutil.move(str(BASE / f"{name}.hf.new"), str(out))
        print(f"swapped in {name}.hf (old kept at {old.name})")

    # ---- verification ----
    k32 = set(keys(load(BASE / "train_32k.hf")))
    prev = k32
    for n in [64, 128, 256]:
        ds = load(BASE / f"train_{n}k.hf")
        ks = keys(ds)
        ks_set = set(ks)
        assert len(ks_set) == len(ks) == n * 1000, f"{n}k: dup or size"
        assert prev <= ks_set, f"{n}k: nesting broken"
        assert not (ks_set & exclude), f"{n}k: exclusion pool leaked in"
        L = set(ds["length"])
        assert L == {512}, f"{n}k: lengths {L}"
        prev = ks_set
        print(f"verified train_{n}k: nested, disjoint from heldout+queries, 512-tok")
    print("chain: 4k c 8k c 16k c 32k c 64k c 128k c 256k — rebuilt and verified")


if __name__ == "__main__":
    main()
