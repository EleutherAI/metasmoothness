"""Rebuild the OLMo magic_full_L32b1024 chunk order, verify it against the
saved doc_ids, and pull per-token scores for the chunks holding two paper
snippets. Writes a token_heatmap_paper.py-compatible JSON."""
import json, re, sys
import numpy as np
from transformers import AutoTokenizer
from bergson.data import load_data_string, tokenize_and_chunk

RUN = "runs/olmo32b/magic_full_L32b1024"
NEEDLES = {
    "pias1": "the major caspases responsible for PIAS1 cleavage are caspase-3",
    "orangutan": "or could spread amongst orangutans in captivity or in the wild",
}
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3.1-32B-Think")
ds = load_data_string("runs/deep_ignorance_ms/mix_85_15", "train", None, "")
norm = lambda s: re.sub(r"\s+", " ", s)
texts = ds["text"]
hits = {}
for k, n in NEEDLES.items():
    hits[k] = [i for i, t in enumerate(texts) if isinstance(t, str) and n in norm(t)]
    print(k, "docs:", hits[k], flush=True)

ch = tokenize_and_chunk(ds, tok, chunk_size=512)
ch = ch.shuffle(seed=42)
saved = np.load(f"{RUN}/scores/doc_ids.npy", mmap_mode="r")
mine = np.asarray(ch["doc_ids"], dtype=np.int64)
print("rows", len(mine), len(saved), "doc_ids identical:", mine.shape == saved.shape and bool((mine == saved).all()), flush=True)

info = json.load(open(f"{RUN}/scores/info.json"))
dt = np.dtype({k: info["dtype"][k] for k in ("names", "formats", "offsets", "itemsize")})
scores = np.memmap(f"{RUN}/scores/scores.bin", dtype=dt, mode="r")
offsets = np.load(f"{RUN}/scores/offsets.npy")

out = {"negative": [], "positive": []}
for k, docs in hits.items():
    for d in docs:
        rows = np.where((saved == d).any(axis=1))[0]
        for r in rows:
            ids = ch[int(r)]["input_ids"]
            text = tok.decode(ids)
            if NEEDLES[k][:40] not in norm(text):
                continue
            s = scores[offsets[r]:offsets[r + 1]]
            assert s["written_0"].all()
            sc = [None] + [float(x) for x in s["score_0"]]
            toks = [{"t": tok.decode([i]), "s": sc[j], "id": int(i), "doc": int(saved[r, j])} for j, i in enumerate(ids)]
            # token index where the snippet starts
            acc = ""; start = 0
            for j, t in enumerate(toks):
                if NEEDLES[k][:20] in norm(acc + t["t"]):
                    start = max(0, j - 8); break
                acc += t["t"]
            print(k, "row", int(r), "doc", d, "snippet near token", start, flush=True)
            out["negative"].append({"kind": k, "row": int(r), "start": start, "tokens": toks})
json.dump(out, open("runs/olmo32b/paper_tokens/olmo_paper_tokens.json", "w"))
print("wrote", len(out["negative"]), "chunks")
