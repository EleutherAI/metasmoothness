"""Build the 32k training pool, keeping the existing 50-chunk query set disjoint.

The original query set is chunks[16000:16050], which sits *inside* a naive
chunks[0:32000] range -- training on that would leak the queries. So train_32k =
chunks[0:16000] + chunks[16050:32050] (32000 chunks), and as a belt-and-braces
guard we also drop any chunk whose input_ids exactly match a query chunk.

Reuses bergson's tokenize_and_chunk so packing matches the other splits.

Run: python scripts/ekfac_vs_n/build_32k.py --out_dir runs/ekfac_vs_n/datasets
"""

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer

from bergson.data import tokenize_and_chunk

CORPUS = "EleutherAI/SmolLM2-135M-10B"
CHUNK_SIZE = 512
N_RAW_DOCS = 40000  # same slice used by build_datasets.py -> identical pool
TARGET = 32000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="runs/ekfac_vs_n/datasets")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    raw = load_dataset(CORPUS, split=f"train[:{N_RAW_DOCS}]").select_columns(["text"])
    chunks = tokenize_and_chunk(raw, tokenizer, chunk_size=CHUNK_SIZE)
    print(f"Pool: {len(chunks)} chunks")

    # Query chunks to exclude (identity match on input_ids).
    query = load_from_disk(str(out_dir / "query_50.hf"))
    q_set = {tuple(x) for x in query["input_ids"]}

    ids, lens = [], []
    for row in chunks:
        t = tuple(row["input_ids"])
        if t in q_set:
            continue
        ids.append(row["input_ids"])
        lens.append(row["length"])
        if len(ids) >= TARGET:
            break
    if len(ids) < TARGET:
        raise RuntimeError(f"only {len(ids)} non-query chunks, need {TARGET}")

    ds = Dataset.from_dict({"input_ids": ids, "length": lens})
    path = out_dir / "train_32k.hf"
    ds.save_to_disk(str(path))
    print(f"Saved {len(ds)} train chunks -> {path}")

    # Sanity: overlap with existing train_16k (should be nested) and query (0).
    q_overlap = sum(tuple(x) in q_set for x in ds["input_ids"])
    print(f"query overlap in train_32k: {q_overlap} (want 0)")


if __name__ == "__main__":
    main()
