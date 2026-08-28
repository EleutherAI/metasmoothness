"""Build GPT-2-tokenized 512-token-chunk datasets for the EK-FAC-vs-N study.

Streams raw text from EleutherAI/SmolLM2-135M-10B, tokenizes + packs into
fixed 512-token chunks with the GPT-2 tokenizer (reusing bergson's own
``tokenize_and_chunk`` so packing is identical to the training pipeline), and
saves nested training pools plus a disjoint 50-chunk query set to disk.

Layout of the produced ``chunks`` pool (all disjoint):
    chunks[0:16000]  -> training pool (train_16k = pool, train_8k = pool[:8000],
                        train_4k = pool[:4000], all nested subsets)
    chunks[16000:16050] -> 50 held-out query chunks

Each saved split keeps only ``input_ids`` + ``length`` (no ``doc_ids``), so the
attribution pipeline treats one row = one document (one 512-token chunk), which
is exactly what the leave-k-out bank and per-query EK-FAC scores index into.

Run:
    python scripts/ekfac_vs_n/build_datasets.py \
        --out_dir runs/ekfac_vs_n/datasets
"""

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from bergson.data import tokenize_and_chunk

CORPUS = "EleutherAI/SmolLM2-135M-10B"
CHUNK_SIZE = 512
# Training pools (nested) and the query set carved from a single chunk stream.
TRAIN_SIZES = {"4k": 4000, "8k": 8000, "16k": 16000}
MAX_TRAIN = max(TRAIN_SIZES.values())
N_QUERY = 50
# Raw docs to pull; dclm_edu docs average ~1k tokens so this yields well over
# (MAX_TRAIN + N_QUERY) 512-token chunks.
N_RAW_DOCS = 40000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="runs/ekfac_vs_n/datasets")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading {N_RAW_DOCS} raw docs from {CORPUS} ...")
    raw = load_dataset(CORPUS, split=f"train[:{N_RAW_DOCS}]")
    raw = raw.select_columns(["text"])

    print(f"Tokenizing + chunking to {CHUNK_SIZE}-token chunks (gpt2) ...")
    chunks = tokenize_and_chunk(raw, tokenizer, chunk_size=CHUNK_SIZE)
    print(f"Produced {len(chunks)} chunks total.")

    needed = MAX_TRAIN + N_QUERY
    if len(chunks) < needed:
        raise RuntimeError(
            f"Only {len(chunks)} chunks produced, need >= {needed}. "
            f"Increase N_RAW_DOCS."
        )

    # Keep one row = one chunk: input_ids + length only (drop doc_ids so the
    # pipeline never re-maps chunks back to their original corpus documents).
    def keep(split: Dataset) -> Dataset:
        return Dataset.from_dict(
            {"input_ids": split["input_ids"], "length": split["length"]}
        )

    train_pool = chunks.select(range(MAX_TRAIN))
    query = chunks.select(range(MAX_TRAIN, MAX_TRAIN + N_QUERY))

    query_path = out_dir / "query_50.hf"
    keep(query).save_to_disk(str(query_path))
    print(f"Saved {N_QUERY} query chunks -> {query_path}")

    for name, n in TRAIN_SIZES.items():
        split = train_pool.select(range(n))
        path = out_dir / f"train_{name}.hf"
        keep(split).save_to_disk(str(path))
        print(f"Saved {n} train chunks -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
