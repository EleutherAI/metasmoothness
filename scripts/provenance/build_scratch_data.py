"""Build a large GPT-2 chunk pool for the from-scratch loss probe / re-init runs.

Same 512-token gpt2 packing as the other splits, disjoint from the 50 query
chunks (identity filter on input_ids). Nested: the first 4k/8k/16k/32k of this
pool can serve the re-init LDS points, and the whole thing feeds the loss-vs-N
probe.
"""

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer

from bergson.data import tokenize_and_chunk

CORPUS = "EleutherAI/SmolLM2-135M-10B"
CHUNK = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="runs/ekfac_vs_n/datasets")
    ap.add_argument("--n_docs", type=int, default=380000)
    ap.add_argument("--target_chunks", type=int, default=512000)
    ap.add_argument("--name", default="train_scratch_512k.hf")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    tok = AutoTokenizer.from_pretrained("gpt2")
    raw = load_dataset(CORPUS, split=f"train[:{args.n_docs}]").select_columns(["text"])
    chunks = tokenize_and_chunk(raw, tok, chunk_size=CHUNK)
    print(f"Pool: {len(chunks)} chunks from {args.n_docs} docs")

    q = load_from_disk(str(out_dir / "query_50.hf"))
    q_set = {tuple(x) for x in q["input_ids"]}

    ids, lens = [], []
    for row in chunks:
        if tuple(row["input_ids"]) in q_set:
            continue
        ids.append(row["input_ids"])
        lens.append(row["length"])
        if len(ids) >= args.target_chunks:
            break
    if len(ids) < args.target_chunks:
        print(f"WARNING: only {len(ids)} chunks (< {args.target_chunks}); "
              f"increase --n_docs")
    ds = Dataset.from_dict({"input_ids": ids, "length": lens})
    path = out_dir / args.name
    ds.save_to_disk(str(path))
    print(f"Saved {len(ds)} chunks -> {path}")


if __name__ == "__main__":
    main()
