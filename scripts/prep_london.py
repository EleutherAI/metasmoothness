"""Build london-llm-1800 into the same shape as our train_*.hf datasets.

The published corpus is ALREADY TOKENIZED, but not with GPT-2's tokenizer: rows
are 2048-token `input_ids` over a ~32k vocab, and decoding them as GPT-2 gives
gibberish. The repo ships the tokenizer that produced them under `tokenizer/`,
so the route is decode with theirs, re-encode with GPT-2, then pack to 512-token
chunks exactly like the smollm2 corpora.

Ours are {input_ids: 512 tokens, length}. Matching that means a london run
differs from a smollm2 run only in the text.
"""
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "0"

from datasets import Dataset, load_dataset, load_from_disk  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402

REPO = "postgrammar/london-llm-1800"
DST = "/mnt/ssd-2/lucia/datasets_local"
CHUNK = 512
SIZES = [16000, 32000, 64000, 128000]

ref = load_from_disk(f"{DST}/train_16k.hf")
assert ref.column_names == ["input_ids", "length"], ref.column_names
assert len(ref[0]["input_ids"]) == CHUNK
print(f"reference format confirmed: {ref.column_names}, chunk {CHUNK}", flush=True)

# It is a DATASET repo, so AutoTokenizer cannot resolve it by name -- pull the
# tokenizer file directly and load it as a fast tokenizer.
_tok_file = hf_hub_download(REPO, "tokenizer/tokenizer.json", repo_type="dataset")
src_tok = PreTrainedTokenizerFast(tokenizer_file=_tok_file)
gpt2 = AutoTokenizer.from_pretrained("gpt2")
print(f"source tokenizer vocab {len(src_tok)}, gpt2 vocab {gpt2.vocab_size}", flush=True)

# Stream: 11.65M rows of 2048 tokens is far more than needed. The largest target
# is 128k chunks x 512 = 65.5M gpt2 tokens; source rows carry ~2048 tokens each,
# so ~40k rows suffices even after re-tokenisation shrinks or grows the count.
stream = load_dataset(REPO, split="train", streaming=True)

eos = gpt2.eos_token_id
buf, rows, target = [], [], max(SIZES)
for i, rec in enumerate(stream):
    text = src_tok.decode(rec["input_ids"], skip_special_tokens=True)
    buf.extend(gpt2(text, add_special_tokens=False)["input_ids"])
    buf.append(eos)
    while len(buf) >= CHUNK:
        rows.append(buf[:CHUNK])
        buf = buf[CHUNK:]
    if len(rows) >= target:
        break
    if i % 500 == 0:
        print(f"  src row {i}: {len(rows)}/{target} chunks", flush=True)

print(f"packed {len(rows)} chunks from {i + 1} source rows", flush=True)
if len(rows) < min(SIZES):
    sys.exit(f"only {len(rows)} chunks, below smallest target {min(SIZES)}")

# Nested by construction, same rule the smollm2 chain uses: every smaller N is a
# prefix of every larger one, so the token axis stays a clean comparison.
for n in SIZES:
    if len(rows) < n:
        print(f"skipping {n}: only {len(rows)} chunks", flush=True)
        continue
    out = Dataset.from_dict({"input_ids": rows[:n], "length": [CHUNK] * n})
    path = f"{DST}/london_{n // 1000}k.hf"
    out.save_to_disk(path)
    os.system(f"chmod -R a+rX {path}")
    print(f"wrote {path}: {len(out)} rows", flush=True)

print("DONE", flush=True)
