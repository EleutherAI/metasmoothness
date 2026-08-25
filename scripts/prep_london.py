"""Build london-llm-1800 into the same shape as our train_*.hf datasets.

Ours are {input_ids: 512 tokens, length}, gpt2 tokenizer, one row per chunk. This
packs the historical corpus the same way so a london run differs from a smollm2
run ONLY in the text, not in tokenizer, chunk length or row semantics.
"""
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "0"

from datasets import load_dataset, load_from_disk  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

DST = "/mnt/ssd-2/lucia/datasets_local"
CHUNK = 512
SIZES = [16000, 32000, 64000]

ref = load_from_disk(f"{DST}/train_16k.hf")
assert ref.column_names == ["input_ids", "length"], ref.column_names
assert len(ref[0]["input_ids"]) == CHUNK, len(ref[0]["input_ids"])
print(f"reference format confirmed: {ref.column_names}, chunk {CHUNK}", flush=True)

ds = load_dataset("postgrammar/london-llm-1800")
split = "train" if "train" in ds else list(ds)[0]
d = ds[split]
print(f"downloaded: split={split} rows={len(d)} cols={d.column_names}", flush=True)

text_col = next((c for c in ("text", "content", "raw", "body")
                 if c in d.column_names), None)
if text_col is None:
    text_col = next(c for c in d.column_names
                    if isinstance(d[0][c], str) and len(d[0][c]) > 50)
print(f"text column: {text_col}", flush=True)

tok = AutoTokenizer.from_pretrained("gpt2")

# Pack into fixed 512-token chunks, exactly like the reference corpora: tokenize
# each document, concatenate with EOS between, then slice. Packing rather than
# truncating keeps every chunk full, so `length` is constant and the row count is
# what controls dataset size -- the same assumption every N in the grid relies on.
buf, rows, target = [], [], max(SIZES)
eos = tok.eos_token_id
for i, rec in enumerate(d):
    buf.extend(tok(rec[text_col], add_special_tokens=False)["input_ids"])
    buf.append(eos)
    while len(buf) >= CHUNK:
        rows.append(buf[:CHUNK])
        buf = buf[CHUNK:]
        if len(rows) >= target:
            break
    if len(rows) >= target:
        break
    if i % 2000 == 0:
        print(f"  doc {i}: {len(rows)}/{target} chunks", flush=True)

print(f"packed {len(rows)} chunks from {i + 1} documents", flush=True)
if len(rows) < min(SIZES):
    sys.exit(f"corpus yields only {len(rows)} chunks, below the smallest size {min(SIZES)}")

from datasets import Dataset  # noqa: E402

for n in SIZES:
    if len(rows) < n:
        print(f"skipping {n}: only {len(rows)} chunks available", flush=True)
        continue
    out = Dataset.from_dict({"input_ids": rows[:n], "length": [CHUNK] * n})
    path = f"{DST}/london_{n // 1000}k.hf"
    out.save_to_disk(path)
    os.system(f"chmod -R a+rX {path}")
    print(f"wrote {path}: {len(out)} rows", flush=True)

print("DONE", flush=True)
