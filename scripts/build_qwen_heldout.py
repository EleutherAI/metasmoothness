"""Held-out set in Qwen tokenization, disjoint from train_4k_qwen and query_20_qwen.

lr must be selected on held-out loss, never train loss (README: on repeated small
corpora the train-loss optimum memorises and can generalise worse than the
untrained model). The existing heldout_4k is gpt2-tokenized and unusable here.

Takes chunks after the train slice from the same deterministic pool, so all three
sets are disjoint by construction: [0,20) query, [20,4020) train, [4020,8020) heldout.
"""
import sys

sys.path.insert(0, "/mnt/ssd-1/lucia/bergson-damping")

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from bergson.data import tokenize_and_chunk  # noqa: E402

MODEL = "/mnt/ssd-2/lucia/models/Qwen2.5-7B"
OUT = "/mnt/ssd-2/lucia/datasets_local"
tok = AutoTokenizer.from_pretrained(MODEL)
raw = load_dataset("EleutherAI/SmolLM2-135M-10B", split="train[:8000]")
chunks = tokenize_and_chunk(raw, tok, 512)
h = chunks.select(range(4020, 8020))
h.save_to_disk(f"{OUT}/heldout_4k_qwen.hf")
print(f"  wrote heldout_4k_qwen.hf ({len(h)} chunks, indices 4020..8019)")
print("  disjoint: query [0,20)  train [20,4020)  heldout [4020,8020)")
