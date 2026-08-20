"""Push the paper's dataset family to the Hub as one repo with named splits.

Splits: train_{4,8,16,32,64,128,256}k (one nested chain), heldout_4k (model selection),
query_50 / query_20 (LDS queries). Re-runs of this script overwrite the repo's splits,
so only run it after `rebuild_scaling_family.py` has verified the chain.

Usage:
    python push_scaling_datasets.py [--repo EleutherAI/bergson-smollm2-scaling] [--private]
"""

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

SPLITS = ["train_4k", "train_8k", "train_16k", "train_32k", "train_64k",
          "train_128k", "train_256k", "heldout_4k", "query_50", "query_20"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="EleutherAI/bergson-smollm2-scaling")
    ap.add_argument("--datasets_dir",
                    default="/mnt/ssd-1/lucia/bergson-damping/runs/ekfac_vs_n/datasets")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    base = Path(args.datasets_dir)
    dd = {}
    for name in SPLITS:
        ds = load_from_disk(str(base / f"{name}.hf"))
        assert isinstance(ds, Dataset)
        assert set(ds["length"]) == {512}, f"{name}: non-512 docs"
        dd[name] = ds
        print(f"  {name}: {len(ds)} docs")

    # Independent recheck of the two invariants the paper relies on.
    def keyset(ds):
        return set(hash(tuple(x)) for x in ds["input_ids"])

    prev = None
    train_keys = {}
    for name in SPLITS[:7]:
        ks = keyset(dd[name])
        assert len(ks) == len(dd[name]), f"{name}: duplicate docs"
        if prev is not None:
            assert prev <= ks, f"{name}: not a superset of the previous rung"
        prev = ks
        train_keys[name] = ks
    for name in ["heldout_4k", "query_50", "query_20"]:
        assert not (keyset(dd[name]) & train_keys["train_256k"]), \
            f"{name} overlaps train_256k"
    print("verified: nested chain + heldout/query disjointness")

    DatasetDict(dd).push_to_hub(args.repo, private=args.private)
    print(f"pushed {len(SPLITS)} splits to {args.repo}")


if __name__ == "__main__":
    main()
