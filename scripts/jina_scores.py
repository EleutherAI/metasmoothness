#!/usr/bin/env python3
"""Build jina-embeddings-v3 semantic score dirs for paper filter rows.

Sibling of bm25_scores.py: same output layout (structured scores.bin +
info.json + config.yaml, higher_is_better), scores are cosine similarity of
asymmetric retrieval embeddings (train doc = retrieval.passage, query =
retrieval.query). Ported from bergson examples/bank_baselines/semantic_baseline.py.

    python jina_scores.py <run_id> [--out-name jina_scores] [--batch-size 16]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, "/mnt/ssd-2/lucia/metasmoothness/scripts")
import run_config  # noqa: E402

sys.path.insert(0, "/mnt/ssd-2/lucia/bergson-filter")
from bergson.data import load_data_string  # noqa: E402

MODEL = "jinaai/jina-embeddings-v5-text-small"
EXP = [Path("/mnt/ssd-2/lucia/paper_runs/experiments"),
       Path("/mnt/ssd-1/lucia/paper_runs/experiments")]
MIRROR = Path("/mnt/ssd-2/lucia/datasets_local")


def mirrored(path: str) -> str:
    local = MIRROR / Path(path.rstrip("/")).name
    return str(local) if local.is_dir() else path


def load_model(device: str):
    from transformers.modeling_utils import PreTrainedModel
    # jina-v3's custom code predates transformers 5.x (see bergson
    # examples/bank_baselines/semantic_baseline.py for the full story).
    if not isinstance(getattr(PreTrainedModel, "all_tied_weights_keys", None), property):
        PreTrainedModel.all_tied_weights_keys = {}
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32)
    for name, buf in model.named_buffers():
        if "lora_dropout_mask" in name:
            buf.data = torch.ones_like(buf)
    return model.to(device).eval()


@torch.no_grad()
def encode(model, texts, prompt_name, batch_size):
    out = []
    for start in range(0, len(texts), batch_size):
        emb = model.encode(texts=texts[start:start + batch_size],
                           task="retrieval", prompt_name=prompt_name)
        if torch.is_tensor(emb):
            emb = emb.float().cpu().numpy()
        emb = np.asarray(emb, dtype=np.float32)
        emb /= np.clip(np.linalg.norm(emb, axis=-1, keepdims=True), 1e-12, None)
        out.append(emb)
        if start % (batch_size * 100) == 0:
            print(f"  {start}/{len(texts)}", flush=True)
    return np.concatenate(out, axis=0)


def write_scores(scores, out, train_ds, query_ds):
    out.mkdir(parents=True, exist_ok=True)
    names, formats = [], []
    for i in range(scores.shape[1]):
        names += [f"score_{i}", f"written_{i}"]
        formats += ["float32", "bool"]
    dt = np.dtype({"names": names, "formats": formats})
    mm = np.memmap(out / "scores.bin", dtype=dt, mode="w+", shape=(scores.shape[0],))
    for i in range(scores.shape[1]):
        mm[f"score_{i}"] = scores[:, i]
        mm[f"written_{i}"] = True
    mm.flush()
    info = {
        "attribute_tokens": False,
        "num_items": int(scores.shape[0]),
        "num_rows": int(scores.shape[0]),
        "num_scores": int(scores.shape[1]),
        "dtype": {"names": names, "formats": formats,
                  "offsets": [dt.fields[n][1] for n in names],
                  "itemsize": dt.itemsize},
    }
    json.dump(info, open(out / "info.json", "w"), indent=2)
    yaml.safe_dump(
        {"steps": [{"score": {"score_cfg": {"score": "individual",
                                            "higher_is_better": True}}}],
         "jina": {"model": MODEL, "train_dataset": train_ds,
                  "query_dataset": query_ds,
                  "tasks": ["retrieval/document", "retrieval/query"]}},
        open(out / "config.yaml", "w"), sort_keys=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--out-name", default="jina_scores")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = next((b / args.run_id for b in EXP if (b / args.run_id).is_dir()), None)
    assert root, f"run dir not found: {args.run_id}"
    out = root / args.out_name
    if (out / "info.json").is_file() and not args.force:
        sys.exit(f"{out} already exists; --force to rebuild")

    exp = run_config.load(root)
    magic = next(s["magic"] for s in exp["steps"] if "magic" in s)
    train_ds = mirrored(magic["data"]["dataset"])
    query_ds = mirrored(magic["query"]["dataset"])
    # datasets here are pre-tokenized gpt2 input_ids; decode back to text
    from transformers import AutoTokenizer
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")

    def texts_of(path):
        ds = load_data_string(path, "train")
        if hasattr(ds, "column_names") and "input_ids" in ds.column_names:
            rows = ds["input_ids"]
            flat = [([t for ch in r for t in ch] if r and isinstance(r[0], list) else r)
                    for r in rows]
            return gpt2_tok.batch_decode(flat, skip_special_tokens=True)
        return [x if isinstance(x, str) else x["text"] for x in ds]

    train_texts = texts_of(train_ds)
    query_texts = texts_of(query_ds)
    print(f"{len(train_texts)} train docs, {len(query_texts)} queries", flush=True)

    model = load_model(args.device)
    q = encode(model, query_texts, "query", args.batch_size)
    t = encode(model, train_texts, "document", args.batch_size)
    scores = t @ q.T  # [n_train, n_query] cosine (embeddings are L2-normalized)
    write_scores(scores.astype(np.float32), out, train_ds, query_ds)
    print(f"wrote {out} shape={scores.shape}")


if __name__ == "__main__":
    main()
