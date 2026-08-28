"""Build a train pool of arbitrary size, nested with the existing 4k/8k/16k/32k splits.

Generalizes build_32k.py: same corpus, chunk size, ordering and query-exclusion, so
train_{N}.hf is a prefix-superset of every smaller split. Raw docs are scaled to the
target with headroom, since chunking yields fewer chunks than docs.

Run: python scripts/ekfac_vs_n/build_pool.py --target 64000
"""

from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from simple_parsing import parse
from transformers import AutoTokenizer

from bergson.data import tokenize_and_chunk

CORPUS = "EleutherAI/SmolLM2-135M-10B"
CHUNK_SIZE = 512
# Measured: 100000 raw docs -> 195141 chunks (~1.95 chunks/doc). Use a conservative
# 1 raw doc per chunk so short-document shards can't under-fill the target.
RAW_PER_CHUNK = 1.0


@dataclass
class BuildPoolConfig:
    target: int = 64000
    """Largest split to emit, in 512-token chunks."""

    also_emit: str = ""
    """Comma-separated smaller sizes to slice from the same pool, e.g.
    "64000,128000". Slicing one pool is what makes the splits nested:
    tokenize_and_chunk picks num_proc from the dataset size and carries
    remainder tokens per shard, so re-chunking a different raw slice
    shifts every chunk boundary."""

    out_dir: str = "runs/ekfac_vs_n/datasets"
    """Directory holding query_50.hf and the train_*.hf splits."""

    model: str = "gpt2"
    """Tokenizer to chunk with."""

    raw_docs: int = 0
    """Raw docs to pull. 0 = scale from target with headroom."""


def main(build_cfg: BuildPoolConfig):
    out_dir = Path(build_cfg.out_dir)
    n_raw = build_cfg.raw_docs or int(build_cfg.target * RAW_PER_CHUNK * 1.25)

    tokenizer = AutoTokenizer.from_pretrained(build_cfg.model)
    raw = load_dataset(CORPUS, split=f"train[:{n_raw}]").select_columns(["text"])
    chunks = tokenize_and_chunk(raw, tokenizer, chunk_size=CHUNK_SIZE)
    print(f"Pool: {len(chunks)} chunks from {n_raw} raw docs")

    query = load_from_disk(str(out_dir / "query_50.hf"))
    q_set = {tuple(x) for x in query["input_ids"]}

    ids, lens = [], []
    for row in chunks:
        if tuple(row["input_ids"]) in q_set:
            continue
        ids.append(row["input_ids"])
        lens.append(row["length"])
        if len(ids) >= build_cfg.target:
            break
    if len(ids) < build_cfg.target:
        raise RuntimeError(
            f"only {len(ids)} non-query chunks from {n_raw} raw docs, "
            f"need {build_cfg.target}; rerun with a larger --raw_docs"
        )

    sizes = sorted(
        {build_cfg.target}
        | {int(s) for s in build_cfg.also_emit.split(",") if s.strip()}
    )
    for n in sizes:
        ds = Dataset.from_dict({"input_ids": ids[:n], "length": lens[:n]})
        path = out_dir / f"train_{n // 1000}k.hf"
        ds.save_to_disk(str(path))
        q_overlap = sum(tuple(x) in q_set for x in ds["input_ids"])
        print(f"Saved {len(ds)} chunks -> {path} (query overlap {q_overlap}, want 0)")


if __name__ == "__main__":
    main(parse(BuildPoolConfig))
