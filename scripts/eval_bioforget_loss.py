#!/usr/bin/env python3
"""Measure GPT-2 loss across the SmolLM scaling checkpoints on BioForget."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_payload(path: Path, input_ids: list[list[int]], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.tensor(input_ids, dtype=torch.int32)
    torch.save({"input_ids": tensor, "metadata": metadata}, path)
    print(f"wrote {path}: {tensor.shape[0]:,} x {tensor.shape[1]:,}", flush=True)


def stable_offset(key: str, available: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % available


def prepare(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    bio = load_dataset(args.bio_source, split=args.bio_split)

    order = list(range(len(bio)))
    random.Random(args.seed).shuffle(order)
    bio_chunks = []
    row_hashes = []
    for index in order:
        row = bio[index]
        text = row.get(args.text_column) or ""
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < args.sequence_length:
            continue
        key = str(row.get("doi") or row.get("title") or index)
        start = stable_offset(key, len(ids) - args.sequence_length + 1, args.seed)
        bio_chunks.append(ids[start : start + args.sequence_length])
        row_hashes.append(hashlib.sha256(key.encode()).hexdigest()[:16])
        if len(bio_chunks) == args.examples:
            break
    if len(bio_chunks) < args.examples:
        raise RuntimeError(
            f"only {len(bio_chunks):,} BioForget rows contain "
            f"{args.sequence_length} tokens"
        )
    save_payload(
        args.bio_out,
        bio_chunks,
        {
            "source": args.bio_source,
            "split": args.bio_split,
            "text_column": args.text_column,
            "selection": "seeded document shuffle; one stable-hash window per document",
            "seed": args.seed,
            "row_hashes": row_hashes,
        },
    )

    control = load_from_disk(str(args.control_source))
    if hasattr(control, "keys") and args.control_split in control:
        control = control[args.control_split]
    order = list(range(len(control)))
    random.Random(args.seed).shuffle(order)
    control_chunks = []
    for index in order:
        ids = control[index]["input_ids"]
        if len(ids) >= args.sequence_length:
            control_chunks.append(ids[: args.sequence_length])
        if len(control_chunks) == args.examples:
            break
    if len(control_chunks) < args.examples:
        raise RuntimeError(f"only {len(control_chunks):,} control chunks available")
    save_payload(
        args.control_out,
        control_chunks,
        {
            "source": str(args.control_source),
            "split": args.control_split,
            "selection": "seeded row shuffle",
            "seed": args.seed,
        },
    )


@torch.inference_mode()
def losses_for_dataset(
    model: torch.nn.Module,
    payload: dict,
    device: torch.device,
    batch_size: int,
) -> dict:
    input_ids = payload["input_ids"]
    sums = []
    counts = []
    for start in range(0, len(input_ids), batch_size):
        batch = input_ids[start : start + batch_size].to(device=device, dtype=torch.long)
        logits = model(input_ids=batch, use_cache=False).logits[:, :-1].float()
        labels = batch[:, 1:]
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        sums.extend(token_losses.sum(dim=1).cpu().tolist())
        counts.extend([labels.shape[1]] * labels.shape[0])
        done = min(start + batch_size, len(input_ids))
        print(f"{done:,}/{len(input_ids):,}", flush=True)
    total_loss = sum(sums)
    total_tokens = sum(counts)
    return {
        "mean_nll": total_loss / total_tokens,
        "per_example_nll": [loss / count for loss, count in zip(sums, counts)],
        "token_count": total_tokens,
        "example_count": len(sums),
        "source_metadata": payload["metadata"],
    }


def evaluate(args: argparse.Namespace) -> None:
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.out}")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval().to(device)
    result = {
        "label": args.label,
        "tokens_seen": args.tokens_seen,
        "model": args.model,
        "precision": "fp32",
        "datasets": {},
    }
    for name, path in (("bioforget", args.bio_data), ("smollm_heldout", args.control_data)):
        print(f"evaluating {args.label} on {name}", flush=True)
        result["datasets"][name] = losses_for_dataset(
            model, load_payload(path), device, args.batch_size
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)


def bootstrap_mean_ci(values: list[float], seed: int, draws: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values_array = np.asarray(values)
    means = []
    for start in range(0, draws, 256):
        batch = min(256, draws - start)
        indices = rng.integers(0, len(values_array), size=(batch, len(values_array)))
        means.append(values_array[indices].mean(axis=1))
    low, high = np.quantile(np.concatenate(means), [0.025, 0.975])
    return float(low), float(high)


def bootstrap_residual_ci(
    bio_delta: list[float], control_delta: list[float], seed: int, draws: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    bio_array = np.asarray(bio_delta)
    control_array = np.asarray(control_delta)
    residuals = []
    for start in range(0, draws, 256):
        batch = min(256, draws - start)
        bio_indices = rng.integers(0, len(bio_array), size=(batch, len(bio_array)))
        control_indices = rng.integers(
            0, len(control_array), size=(batch, len(control_array))
        )
        residuals.append(
            bio_array[bio_indices].mean(axis=1)
            - control_array[control_indices].mean(axis=1)
        )
    low, high = np.quantile(np.concatenate(residuals), [0.025, 0.975])
    return float(low), float(high)


def summarize(args: argparse.Namespace) -> None:
    results = [json.loads(path.read_text()) for path in sorted(args.results.glob("*.json"))]
    if not results:
        raise RuntimeError(f"no JSON results in {args.results}")
    results.sort(key=lambda item: item["tokens_seen"])
    reference = next((item for item in results if item["tokens_seen"] > 0), results[0])
    ref_bio = reference["datasets"]["bioforget"]["per_example_nll"]
    ref_control = reference["datasets"]["smollm_heldout"]["per_example_nll"]
    rows = []
    for item in results:
        bio = item["datasets"]["bioforget"]["per_example_nll"]
        control = item["datasets"]["smollm_heldout"]["per_example_nll"]
        if len(bio) != len(ref_bio) or len(control) != len(ref_control):
            raise ValueError("all evaluations must use identical prepared datasets")
        bio_delta = [x - y for x, y in zip(bio, ref_bio)]
        control_delta = [x - y for x, y in zip(control, ref_control)]
        residual = (sum(bio_delta) / len(bio_delta)) - (
            sum(control_delta) / len(control_delta)
        )
        bio_lo, bio_hi = bootstrap_mean_ci(bio_delta, args.seed, args.bootstrap)
        residual_lo, residual_hi = bootstrap_residual_ci(
            bio_delta, control_delta, args.seed, args.bootstrap
        )
        rows.append(
            {
                "label": item["label"],
                "tokens_seen": item["tokens_seen"],
                "bioforget_nll": item["datasets"]["bioforget"]["mean_nll"],
                "smollm_heldout_nll": item["datasets"]["smollm_heldout"]["mean_nll"],
                "bio_delta_vs_4m": sum(bio_delta) / len(bio_delta),
                "bio_delta_lo": bio_lo,
                "bio_delta_hi": bio_hi,
                "bio_specific_residual_vs_4m": residual,
                "residual_lo": residual_lo,
                "residual_hi": residual_hi,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--bio-source", default="cais/wmdp-bio-forget-corpus")
    prep.add_argument("--bio-split", default="train")
    prep.add_argument("--text-column", default="text")
    prep.add_argument("--control-source", type=Path, required=True)
    prep.add_argument("--control-split", default="train")
    prep.add_argument("--tokenizer", default="gpt2")
    prep.add_argument("--examples", type=int, default=4000)
    prep.add_argument("--sequence-length", type=int, default=512)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--bio-out", type=Path, required=True)
    prep.add_argument("--control-out", type=Path, required=True)
    prep.set_defaults(func=prepare)

    run = sub.add_parser("evaluate")
    run.add_argument("--model", required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--tokens-seen", type=int, required=True)
    run.add_argument("--bio-data", type=Path, required=True)
    run.add_argument("--control-data", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(func=evaluate)

    summary = sub.add_parser("summarize")
    summary.add_argument("--results", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    summary.add_argument("--bootstrap", type=int, default=10000)
    summary.add_argument("--seed", type=int, default=42)
    summary.set_defaults(func=summarize)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
