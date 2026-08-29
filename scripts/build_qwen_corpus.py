"""Build the Qwen-tokenized corpus for the 7B proponent-filter row.

Same recipe as notes/dataset_provenance.md -- corpus EleutherAI/SmolLM2-135M-10B,
deterministic prefix of split train, bergson.data.tokenize_and_chunk at chunk_size
512 -- with the tokenizer swapped for Qwen's.

The chunks are NOT comparable to the gpt2 chain and cannot be: a different
tokenizer moves every chunk boundary, so "the same 4000 documents" does not exist
across tokenizers. What carries over is the RULE, and what matters for this row is
internal: the query set is disjoint from the training set, and both are a
deterministic prefix with no shuffle.

Query first, train after it, so the two can never overlap by construction.
"""
import sys

sys.path.insert(0, "/mnt/ssd-1/lucia/bergson-damping")

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from bergson.data import tokenize_and_chunk  # noqa: E402

MODEL = "/mnt/ssd-2/lucia/models/Qwen2.5-7B"
OUT = "/mnt/ssd-2/lucia/datasets_local"
N_QUERY, N_TRAIN = 20, 4000
# ~1.95 chunks per raw doc for gpt2; Qwen's vocab is larger so it yields FEWER
# chunks per doc. Take a generous prefix and slice exactly afterwards.
RAW_DOCS = 8000

tok = AutoTokenizer.from_pretrained(MODEL)
print(f"  tokenizer {MODEL}  vocab={tok.vocab_size}")

raw = load_dataset("EleutherAI/SmolLM2-135M-10B", split=f"train[:{RAW_DOCS}]")
print(f"  raw docs {len(raw):,}")

# chunk_size is positional-ish in this bergson; num_proc is keyword-only.
chunks = tokenize_and_chunk(raw, tok, 512)
print(f"  chunks {len(chunks):,}  ({len(chunks)/len(raw):.2f} per raw doc)")
need = N_QUERY + N_TRAIN
if len(chunks) < need:
    sys.exit(f"only {len(chunks)} chunks, need {need} -- raise RAW_DOCS")

q = chunks.select(range(N_QUERY))
t = chunks.select(range(N_QUERY, N_QUERY + N_TRAIN))
q.save_to_disk(f"{OUT}/query_20_qwen.hf")
t.save_to_disk(f"{OUT}/train_4k_qwen.hf")
print(f"  wrote query_20_qwen.hf ({len(q)}) and train_4k_qwen.hf ({len(t)})")
print(f"  columns {t.column_names}  first len {len(t[0]['input_ids'])}")
