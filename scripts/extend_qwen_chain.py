"""Extend the Qwen chunk pool by CONCATENATION, not by re-chunking.

Re-chunking a larger raw prefix shifts every chunk boundary, so the existing rungs
would change underneath and stop nesting. Chunking only the NEW raw documents and
appending leaves every existing chunk byte-identical, which is the method
build_train_1m.py already uses for the gpt2 chain.

The seam costs at most one partial chunk of tokens where the two passes meet --
the same thing that already happens at every internal shard boundary, so it is not
a new class of defect.

heldout MOVES: the old heldout sat at [128020,132020), which the 256k rung would
otherwise swallow as training data. It is rebuilt after the largest rung.
"""
import sys

sys.path.insert(0, "/mnt/ssd-1/lucia/bergson-damping")

from datasets import concatenate_datasets, load_dataset, load_from_disk  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from bergson.data import tokenize_and_chunk  # noqa: E402

MODEL = "/mnt/ssd-2/lucia/models/Qwen2.5-1.5B"   # same tokenizer as the 7B, still on disk
OUT = "/mnt/ssd-2/lucia/datasets_local"
OLD_RAW_END = 88_269          # the raw prefix the existing pool was chunked from
NEW_RAW_END = 165_000
N_QUERY, N_HELDOUT = 20, 4000
TARGET = 256_000

tok = AutoTokenizer.from_pretrained(MODEL)
old = concatenate_datasets([
    load_from_disk(f"{OUT}/query_20_qwen.hf"),
    load_from_disk(f"{OUT}/train_128k_qwen.hf"),
])
print(f"  existing chunks reused: {len(old):,} (query + train_128k, unchanged)")

raw = load_dataset("EleutherAI/SmolLM2-135M-10B",
                   split=f"train[{OLD_RAW_END}:{NEW_RAW_END}]")
new = tokenize_and_chunk(raw, tok, 512)
print(f"  new chunks from raw[{OLD_RAW_END}:{NEW_RAW_END}]: {len(new):,}")

pool = concatenate_datasets([old, new])
need = N_QUERY + TARGET + N_HELDOUT
print(f"  pool {len(pool):,}  need {need:,}")
if len(pool) < need:
    sys.exit("pool too small -- raise NEW_RAW_END")

pool.select(range(N_QUERY, N_QUERY + TARGET)).save_to_disk(f"{OUT}/train_256k_qwen.hf")
print(f"  wrote train_256k_qwen  [{N_QUERY},{N_QUERY+TARGET})")
lo = N_QUERY + TARGET
pool.select(range(lo, lo + N_HELDOUT)).save_to_disk(f"{OUT}/heldout_4k_qwen_v2.hf")
print(f"  wrote heldout_4k_qwen_v2  [{lo},{lo+N_HELDOUT}) -- clear of the 256k rung")
