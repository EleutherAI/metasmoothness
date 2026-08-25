"""Build a london held-out set, disjoint from the london train chunks.

prep_london.py consumed source rows 0..31723 to pack 128,003 training chunks. This
starts well past that (row 40,000) so no source document contributes to both, then
packs 4,000 chunks the same way -- gpt2 tokenizer, 512 tokens, {input_ids, length}.

Selecting an lr for a london run on the smollm2 heldout_4k would pick whatever fits
the wrong distribution, which is the opposite of what this corpus is for.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "0"

from datasets import Dataset, load_dataset, load_from_disk  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402

REPO = "postgrammar/london-llm-1800"
DST = "/mnt/ssd-2/lucia/datasets_local"
CHUNK = 512
N_HELDOUT = 4000
SKIP = 40000  # prep_london used rows 0..31723; leave a wide margin

_tok_file = hf_hub_download(REPO, "tokenizer/tokenizer.json", repo_type="dataset")
src_tok = PreTrainedTokenizerFast(tokenizer_file=_tok_file)
gpt2 = AutoTokenizer.from_pretrained("gpt2")
eos = gpt2.eos_token_id

stream = load_dataset(REPO, split="train", streaming=True)
buf, rows = [], []
for i, rec in enumerate(stream):
    if i < SKIP:
        continue
    text = src_tok.decode(rec["input_ids"], skip_special_tokens=True)
    buf.extend(gpt2(text, add_special_tokens=False)["input_ids"])
    buf.append(eos)
    while len(buf) >= CHUNK:
        rows.append(buf[:CHUNK])
        buf = buf[CHUNK:]
    if len(rows) >= N_HELDOUT:
        break
    if i % 500 == 0:
        print(f"  src row {i}: {len(rows)}/{N_HELDOUT}", flush=True)

rows = rows[:N_HELDOUT]
print(f"packed {len(rows)} held-out chunks, source rows {SKIP}..{i}", flush=True)

# Prove disjointness against the largest train set rather than trusting the offset.
train = load_from_disk(f"{DST}/london_128k.hf")
seen = {tuple(r) for r in train["input_ids"]}
overlap = sum(1 for r in rows if tuple(r) in seen)
print(f"overlap with london_128k: {overlap}", flush=True)
assert overlap == 0, f"{overlap} held-out chunks appear in the training set"

out = Dataset.from_dict({"input_ids": rows, "length": [CHUNK] * len(rows)})
path = f"{DST}/london_heldout_4k.hf"
out.save_to_disk(path)
os.system(f"chmod -R a+rwX {path}")
print(f"wrote {path}: {len(out)} rows", flush=True)
print("DONE", flush=True)
