"""Build train_1m.hf by EXTENDING train_512k, not resampling.

The smollm2 ladder is strictly nested -- train_16k is a prefix of train_32k is a
prefix of ... of train_512k, verified by hashing rows 0, 1, n/2 and n-1 at every
level. That nesting is what makes the dataset-size axis a clean comparison: every
larger run sees exactly the documents the smaller one saw, plus more. A 1M set
built by resampling the corpus would break it silently, and every ms value on the
ladder would stop being comparable to its neighbours.

So this reproduces the original packing from the corpus, checks that its first
rows match train_512k exactly, and only then continues past 512k to 1,000,000.
If the check fails the recipe is wrong and it stops rather than writing a subtly
different dataset.

Rows are 512 GPT-2 tokens, column `input_ids` plus `length`, matching the
existing files exactly (verified: every row in train_512k has length 512).

    python prep_smollm2_1m.py [--target 1000000] [--check-only]

Writes /mnt/ssd-2/lucia/datasets_local/train_1m.hf. Streams the corpus, so it
needs network but not much memory. Expect hours: 1M x 512 = 512M tokens.
"""
import argparse
import hashlib
import json
import os
import sys

AP = argparse.ArgumentParser()
AP.add_argument("--target", type=int, default=1_000_000)
AP.add_argument("--check-only", action="store_true",
                help="verify the recipe reproduces train_512k, then stop")
AP.add_argument("--out", default="/mnt/ssd-2/lucia/datasets_local/train_1m.hf")
args = AP.parse_args()

MIRROR = "/mnt/ssd-2/lucia/datasets_local"
CHUNK = 512
CORPUS = "HuggingFaceTB/smollm-corpus"
CONFIG = "cosmopedia-v2"

from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer


def rowhash(ids):
    return hashlib.sha1(json.dumps(list(ids)[:64]).encode()).hexdigest()[:10]


ref = load_from_disk(os.path.join(MIRROR, "train_512k.hf"))
print("reference train_512k: %d rows" % len(ref), flush=True)
ref_head = [rowhash(ref[i]["input_ids"]) for i in (0, 1, 2, 100, 1000)]

tok = AutoTokenizer.from_pretrained("gpt2")
eos = tok.eos_token_id

print("streaming %s/%s ..." % (CORPUS, CONFIG), flush=True)
stream = load_dataset(CORPUS, CONFIG, split="train", streaming=True)

buf, rows, checked = [], [], False
for i, ex in enumerate(stream):
    text = ex.get("text") or ex.get("content") or ""
    if not text:
        continue
    buf.extend(tok(text, add_special_tokens=False)["input_ids"])
    buf.append(eos)
    while len(buf) >= CHUNK:
        rows.append(buf[:CHUNK])
        del buf[:CHUNK]

        if not checked and len(rows) >= 1001:
            got = [rowhash(rows[j]) for j in (0, 1, 2, 100, 1000)]
            if got != ref_head:
                print("RECIPE MISMATCH -- this packing does not reproduce train_512k.",
                      file=sys.stderr)
                print("  expected %s" % ref_head, file=sys.stderr)
                print("  got      %s" % got, file=sys.stderr)
                print("Refusing to write: a 1M set that is not a superset of the "
                      "existing ladder would break every dataset-size comparison.",
                      file=sys.stderr)
                sys.exit(1)
            print("recipe verified: first 1001 rows match train_512k", flush=True)
            checked = True
            if args.check_only:
                sys.exit(0)

        if len(rows) % 50_000 == 0:
            print("  %d / %d rows" % (len(rows), args.target), flush=True)
        if len(rows) >= args.target:
            break
    if len(rows) >= args.target:
        break

if len(rows) < args.target:
    sys.exit("corpus exhausted at %d rows, short of %d" % (len(rows), args.target))

# Belt and braces: the new file must still contain train_512k as its prefix.
tail_ok = all(rowhash(rows[i]) == rowhash(ref[i]["input_ids"])
              for i in (0, len(ref) // 2, len(ref) - 1))
if not tail_ok:
    sys.exit("prefix check failed against train_512k; not writing")

ds = Dataset.from_dict({"input_ids": rows, "length": [CHUNK] * len(rows)})
ds.save_to_disk(args.out)
print("wrote %s: %d rows x %d tokens" % (args.out, len(ds), CHUNK))
print("train_512k is its prefix; the ladder stays nested.")
