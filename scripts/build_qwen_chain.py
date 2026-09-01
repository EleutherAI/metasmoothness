"""Nested Qwen-tokenized train sets for the 7B 1% scaling curve.

Same property the gpt2 chain relies on: every smaller N is a byte-identical
PREFIX of every larger one, so the token axis compares like with like. Built by
chunking one deterministic raw prefix ONCE and slicing it, which makes nesting
true by construction rather than by a rule that has to be re-applied per rung.

Layout, all disjoint:
    [0, 20)            query_20_qwen
    [20, 20+N)         train_{N}_qwen   for each N
    [20+Nmax, +4000)   heldout_4k_qwen

Rebuilding heldout here too, so it sits after the LARGEST train set rather than
after the 4k one -- otherwise the bigger rungs would train on their own eval data.
That is the failure this ordering exists to prevent.
"""
import sys

sys.path.insert(0, "/mnt/ssd-1/lucia/bergson-damping")

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from bergson.data import tokenize_and_chunk  # noqa: E402

MODEL = "/mnt/ssd-2/lucia/models/Qwen2.5-7B"
OUT = "/mnt/ssd-2/lucia/datasets_local"
NS = [4000, 8000, 16000, 32000, 64000, 128000, 256000]
N_QUERY, N_HELDOUT = 20, 4000
# 1.85 chunks/doc measured on this corpus with this tokenizer; +15% headroom.
# Chunk boundaries depend on the RAW PREFIX SIZE, so every rung must come from
# ONE chunking pass -- adding a rung later re-chunks and breaks nesting with
# everything already built. Build the largest rung you will ever want, now.
RAW_DOCS = int((N_QUERY + max(NS) + N_HELDOUT) / 1.85 * 1.15)

tok = AutoTokenizer.from_pretrained(MODEL)
print(f"  raw docs requested {RAW_DOCS:,}")
raw = load_dataset("EleutherAI/SmolLM2-135M-10B", split=f"train[:{RAW_DOCS}]")
chunks = tokenize_and_chunk(raw, tok, 512)
print(f"  chunks {len(chunks):,} ({len(chunks)/len(raw):.2f} per doc)")

need = N_QUERY + max(NS) + N_HELDOUT
if len(chunks) < need:
    sys.exit(f"only {len(chunks):,} chunks, need {need:,} -- raise RAW_DOCS")

chunks.select(range(N_QUERY)).save_to_disk(f"{OUT}/query_20_qwen.hf")
print(f"  query_20_qwen  [0,{N_QUERY})")
for n in NS:
    chunks.select(range(N_QUERY, N_QUERY + n)).save_to_disk(f"{OUT}/train_{n//1000}k_qwen.hf")
    print(f"  train_{n//1000}k_qwen  [{N_QUERY},{N_QUERY+n})")
lo = N_QUERY + max(NS)
chunks.select(range(lo, lo + N_HELDOUT)).save_to_disk(f"{OUT}/heldout_4k_qwen.hf")
print(f"  heldout_4k_qwen  [{lo},{lo+N_HELDOUT})  -- after the largest train set")
