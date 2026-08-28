# Dataset provenance for the token-scaling chain

Written 2026-08-28 after the recipe could not be found from anything in this repo,
which blocked the 1M rung for hours. The scripts that build these datasets lived
only in `/mnt/ssd-1/lucia/bergson-damping/scripts/ekfac_vs_n/`, the Hub copy has a
frontmatter-only README, and nothing anywhere named the corpus. Copies now live in
`scripts/provenance/` so the chain is reproducible from this repo alone.

## The recipe

    corpus     EleutherAI/SmolLM2-135M-10B   (10,058,156 raw docs)
    split      train[:N]                     (a deterministic PREFIX, no shuffle)
    tokenizer  gpt2
    chunking   bergson.data.tokenize_and_chunk, chunk_size 512
    yield      ~1.95 chunks per raw doc

One row = one 512-token chunk = one document to the attribution pipeline. Rows carry
`input_ids` (`List(int32)`, length 512) and `length`. No `doc_ids`.

Import `tokenize_and_chunk` from the **bergson-damping** checkout when rebuilding.
`build_pool.py` warns that chunking picks `num_proc` from dataset size and carries
remainder tokens per shard, so re-chunking a different raw slice shifts every chunk
boundary. A different bergson can silently break nesting.

## The nesting rule, and the trap in it

The token axis is only a clean comparison because every smaller N is a
**byte-identical prefix** of every larger one. Two different scripts build to that
property and they do NOT agree on the exclusion set:

* `build_pool.py --target N` excludes **query_50 only**.
* `build_datasets.py`, `build_32k.py` and `build_train_512k.py` exclude
  **heldout_4k, query_20 and query_50**.

So a pool built by `build_pool.py` is a valid chunk pool but **not** a
prefix-superset of the existing chain: it keeps chunks the chain skipped, and every
index after the first skip shifts. Measured on `train_1000k.hf` built this way:

    train_4k    matches at 0, 1, 1333 ... diverges by index 3998
    train_512k  diverges by index 170666
    27 duplicate chunks in the pool

That is why `train_1M.hf` was NOT built with `build_pool.py` directly. It uses the
same rule that produced the 512k rung:

    train_1M = train_512k ++ (pool chunks not already in the chain
                              and not in heldout_4k / query_20 / query_50)

Deterministic, no RNG. `scripts/build_train_1m.py` implements it and verifies before
writing. Verified after writing, against every rung:

    train_4k / 16k / 64k / 128k / 256k / 512k : prefix OK
    distinct 1,000,000, duplicates 0
    query_20, query_50, heldout_4k overlap: 0

**Verify, do not trust.** The chain was already rebuilt once
(`rebuild_scaling_family.py`) because 64k-256k turned out to be a different draw
from 4k-32k. A build that satisfies the rule by construction can still be wrong if
the inputs are not what you think.

## Where things live

    heldout_4k.hf   only on /mnt/ssd-1/.../ekfac_vs_n/datasets  (read-only under D23)
    query_20/50.hf  /mnt/ssd-2/lucia/datasets_local
    train_*.hf      /mnt/ssd-2/lucia/datasets_local
    Hub             EleutherAI/bergson-smollm2-scaling  (4k..256k only; no 512k/1M)

The Hub copy stops at 256k and has no prose README. If these are ever republished,
push this file with them.
