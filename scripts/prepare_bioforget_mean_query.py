#!/usr/bin/env python3
"""Materialize a fixed BioForget mean-query set from the prior scaling sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=50)
    args = parser.parse_args()

    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    ids = payload["input_ids"][: args.examples].to(torch.int32)
    if len(ids) != args.examples:
        raise ValueError(f"requested {args.examples} examples, found {len(ids)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({"input_ids": ids.tolist()}).save_to_disk(str(args.output))
    metadata = {
        "source": str(args.source),
        "selection": "first rows of prior seeded BioForget scaling sample",
        "examples": len(ids),
        "sequence_length": ids.shape[1],
        "source_metadata": payload.get("metadata", {}),
    }
    Path(str(args.output) + ".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
