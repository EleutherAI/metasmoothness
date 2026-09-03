#!/usr/bin/env python3
"""Build a compact WikiText-103 query set with no WikiText-2 articles."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def top_level_title(text: str) -> str | None:
    parts = text.strip().split()
    if len(parts) < 3:
        return None
    leading = 0
    for part in parts:
        if part != "=":
            break
        leading += 1
    trailing = 0
    for part in reversed(parts):
        if part != "=":
            break
        trailing += 1
    if leading != 1 or trailing != 1:
        return None
    return " ".join(parts[1:-1])


def articles(rows):
    title = None
    body = []
    for row in rows:
        text = row["text"]
        next_title = top_level_title(text)
        if next_title is not None:
            if title is not None:
                yield title, "".join(body)
            title = next_title
            body = [text]
        elif title is not None:
            body.append(text)
    if title is not None:
        yield title, "".join(body)


def normalized(title: str) -> str:
    return " ".join(title.casefold().split())


def stable_offset(title: str, available: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{title}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % available


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reservoir-size", type=int, default=500)
    args = parser.parse_args()

    wt2 = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    wt2_titles = {
        normalized(title)
        for split in wt2.values()
        for row in split
        if (title := top_level_title(row["text"])) is not None
    }

    wt103 = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    rng = random.Random(args.seed)
    reservoir = []
    eligible = 0
    for title, text in articles(wt103):
        if normalized(title) in wt2_titles or len(text) < args.sequence_length * 4:
            continue
        eligible += 1
        item = (title, text)
        if len(reservoir) < args.reservoir_size:
            reservoir.append(item)
        else:
            index = rng.randrange(eligible)
            if index < len(reservoir):
                reservoir[index] = item

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    rng.shuffle(reservoir)
    chunks = []
    title_hashes = []
    for title, text in reservoir:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) < args.sequence_length:
            continue
        start = stable_offset(title, len(token_ids) - args.sequence_length + 1, args.seed)
        chunks.append(token_ids[start : start + args.sequence_length])
        title_hashes.append(hashlib.sha256(normalized(title).encode()).hexdigest()[:16])
        if len(chunks) == args.examples:
            break
    if len(chunks) != args.examples:
        raise RuntimeError(f"only built {len(chunks)} of {args.examples} requested chunks")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({"input_ids": chunks}).save_to_disk(str(args.out))
    metadata = {
        "source": "Salesforce/wikitext:wikitext-103-raw-v1:train",
        "selection": "one stable-offset chunk per article; reservoir sample over non-WT2 articles",
        "wikitext2_title_count": len(wt2_titles),
        "eligible_wikitext103_articles": eligible,
        "examples": len(chunks),
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "title_hashes": title_hashes,
    }
    metadata_path = args.out.parent / f"{args.out.name}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({**metadata, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
