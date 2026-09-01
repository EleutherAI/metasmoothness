#!/usr/bin/env python3
"""Build BM25 lexical score dirs for paper filter rows."""
import argparse
import math
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_config  # noqa: E402

BERGSON = "/mnt/ssd-2/lucia/bergson-filter"
sys.path.insert(0, BERGSON)
from bergson.data import load_data_string  # noqa: E402

EXP = [
    Path("/mnt/ssd-2/lucia/paper_runs/experiments"),
    Path("/mnt/ssd-1/lucia/paper_runs/experiments"),
]
MIRROR = Path("/mnt/ssd-2/lucia/datasets_local")


def mirrored(path: str) -> str:
    local = MIRROR / Path(path.rstrip("/")).name
    return str(local) if local.is_dir() else path


def root_for(run_id: str) -> Path:
    for base in EXP:
        p = base / run_id
        if p.is_dir():
            return p
    raise SystemExit(f"run dir not found: {run_id}")


def docs(ds, prompt_column: str, model: str):
    if "input_ids" in ds.column_names:
        return [list(map(int, ids)) for ids in ds["input_ids"]], "token_ids"
    if prompt_column in ds.column_names:
        return list(ds[prompt_column]), "text"
    raise SystemExit(
        f"dataset has neither {prompt_column!r} nor input_ids: {ds.column_names}"
    )


TOKEN_RE = re.compile(r"(?u)\\b\\w\\w+\\b")


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def text_terms(text: str, ngram_max: int) -> list[str]:
    toks = TOKEN_RE.findall(strip_accents(text).lower())
    out = list(toks)
    if ngram_max >= 2:
        out.extend(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    return out


def token_terms(ids: list[int], ngram_max: int) -> list[tuple[int, ...]]:
    toks = [int(x) for x in ids if int(x) >= 0]
    out = [(x,) for x in toks]
    if ngram_max >= 2:
        out.extend((a, b) for a, b in zip(toks, toks[1:]))
    return out


def make_terms(item, ngram_max: int, mode: str):
    return token_terms(item, ngram_max) if mode == "token_ids" else text_terms(item, ngram_max)


def bm25(train_docs, query_docs, mode, ngram_max=2, min_df=1, k1=1.5, b=0.75):
    n_docs = len(train_docs)
    doc_len = np.zeros(n_docs, dtype=np.float32)
    postings = defaultdict(list)
    dfs = Counter()

    for doc_id, item in enumerate(train_docs):
        counts = Counter(make_terms(item, ngram_max, mode))
        doc_len[doc_id] = sum(counts.values())
        for term, tf in counts.items():
            dfs[term] += 1
            postings[term].append((doc_id, tf))

    if min_df > 1:
        postings = {term: vals for term, vals in postings.items() if dfs[term] >= min_df}

    avgdl = float(doc_len.mean()) if n_docs else 1.0
    if avgdl == 0.0:
        raise SystemExit("all BM25 documents tokenized to length zero")
    denom_const = k1 * (1 - b + b * doc_len / avgdl)
    scores = np.zeros((n_docs, len(query_docs)), dtype=np.float32)

    for qid, item in enumerate(query_docs):
        for term in set(make_terms(item, ngram_max, mode)):
            vals = postings.get(term)
            if not vals:
                continue
            df = dfs[term]
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf in vals:
                scores[doc_id, qid] += idf * tf * (k1 + 1.0) / (tf + denom_const[doc_id])
    return scores


def write_scores(
    scores: np.ndarray,
    out: Path,
    train_ds: str,
    query_ds: str,
    ngram_max: int,
    min_df: int,
):
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
        "dtype": {
            "names": names,
            "formats": formats,
            "offsets": [dt.fields[n][1] for n in names],
            "itemsize": dt.itemsize,
        },
    }
    json.dump(info, open(out / "info.json", "w"), indent=2)
    yaml.safe_dump(
        {
            "steps": [
                {"score": {"score_cfg": {"score": "individual", "higher_is_better": True}}}
            ],
            "bm25": {
                "train_dataset": train_ds,
                "query_dataset": query_ds,
                "ngram_range": [1, ngram_max],
                "min_df": min_df,
            },
        },
        open(out / "config.yaml", "w"),
        sort_keys=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--out-name", default="bm25_scores")
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--min-df", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = root_for(args.run_id)
    out = root / args.out_name
    if (out / "info.json").is_file() and not args.force:
        print(f"READY {out}")
        return

    exp = run_config.load(root)
    magic = next(s["magic"] for s in exp["steps"] if "magic" in s)
    train_ds_s = mirrored(magic["data"]["dataset"])
    query_ds_s = mirrored(magic["query"]["dataset"])
    train = load_data_string(train_ds_s, magic["data"].get("split", "train"))
    query = load_data_string(query_ds_s, magic["query"].get("split", "train"))
    train_docs, train_mode = docs(train, magic["data"].get("prompt_column", "text"), magic["model"])
    query_docs, query_mode = docs(query, magic["query"].get("prompt_column", "text"), magic["model"])
    if train_mode != query_mode:
        raise SystemExit(f"train/query representation mismatch: {train_mode} vs {query_mode}")
    print(
        f"BM25 {args.run_id}: {len(train_docs)} docs x {len(query_docs)} queries, "
        f"mode={train_mode} ngram=(1,{args.ngram_max}) min_df={args.min_df}"
    )
    scores = bm25(train_docs, query_docs, train_mode, args.ngram_max, args.min_df)
    write_scores(scores, out, train_ds_s, query_ds_s, args.ngram_max, args.min_df)
    print(f"wrote {out} shape={scores.shape}")


if __name__ == "__main__":
    main()
