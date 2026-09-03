#!/usr/bin/env python3
"""Evaluate scaling checkpoints on raw LAMBADA passages and final words."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from eval_bioforget_loss import bootstrap_mean_ci
from transformers import AutoModelForCausalLM, AutoTokenizer


def prepare(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    input_ids = []
    target_starts = []
    for row in dataset:
        text = row["text"].rstrip()
        context, target = text.rsplit(" ", 1)
        context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(" " + target, add_special_tokens=False)["input_ids"]
        if not context_ids or not target_ids:
            raise ValueError("LAMBADA row has an empty context or target")
        input_ids.append(context_ids + target_ids)
        target_starts.append(len(context_ids))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": input_ids,
            "target_starts": target_starts,
            "metadata": {
                "dataset": args.dataset,
                "split": args.split,
                "tokenizer": args.tokenizer,
                "examples": len(input_ids),
                "tokens": sum(map(len, input_ids)),
            },
        },
        args.out,
    )
    print(json.dumps({**torch.load(args.out, weights_only=False)["metadata"], "out": str(args.out)}, indent=2))


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.out}")
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    sequences = payload["input_ids"]
    target_starts = payload["target_starts"]
    model = AutoModelForCausalLM.from_pretrained(args.model).eval().to(args.device)
    pad_id = model.config.eos_token_id
    if pad_id is None:
        raise ValueError("model config has no eos_token_id for padding")

    full_sums = []
    full_counts = []
    target_word_sums = []
    target_token_counts = []
    for start in range(0, len(sequences), args.batch_size):
        batch_sequences = sequences[start : start + args.batch_size]
        batch_targets = target_starts[start : start + args.batch_size]
        lengths = [len(sequence) for sequence in batch_sequences]
        width = max(lengths)
        input_ids = torch.full(
            (len(batch_sequences), width), pad_id, dtype=torch.long, device=args.device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(batch_sequences):
            input_ids[row, : len(sequence)] = torch.tensor(sequence, device=args.device)
            attention_mask[row, : len(sequence)] = 1
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits[:, :-1].float()
        labels = input_ids[:, 1:]
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        for row, (length, target_start) in enumerate(zip(lengths, batch_targets)):
            full = losses[row, : length - 1]
            target = losses[row, target_start - 1 : length - 1]
            full_sums.append(float(full.sum().cpu()))
            full_counts.append(length - 1)
            target_word_sums.append(float(target.sum().cpu()))
            target_token_counts.append(len(target))
        done = min(start + args.batch_size, len(sequences))
        print(f"{done:,}/{len(sequences):,}", flush=True)

    result = {
        "label": args.label,
        "tokens_seen": args.tokens_seen,
        "model": args.model,
        "precision": "fp32",
        "dataset_metadata": payload["metadata"],
        "full_passage": {
            "token_weighted_nll": sum(full_sums) / sum(full_counts),
            "per_example_nll": [
                loss_sum / count for loss_sum, count in zip(full_sums, full_counts)
            ],
            "token_count": sum(full_counts),
        },
        "final_word": {
            "mean_word_nll": sum(target_word_sums) / len(target_word_sums),
            "per_example_word_nll": target_word_sums,
            "target_token_count": sum(target_token_counts),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}")


def summarize(args: argparse.Namespace) -> None:
    results = [json.loads(path.read_text()) for path in args.results.glob("*.json")]
    results.sort(key=lambda item: item["tokens_seen"])
    if not results:
        raise RuntimeError(f"no result JSON files in {args.results}")
    reference = results[0]
    ref_full = reference["full_passage"]["per_example_nll"]
    ref_target = reference["final_word"]["per_example_word_nll"]
    rows = []
    for result in results:
        full = result["full_passage"]["per_example_nll"]
        target = result["final_word"]["per_example_word_nll"]
        if len(full) != len(ref_full) or len(target) != len(ref_target):
            raise ValueError("evaluations do not use identical LAMBADA examples")
        full_delta = [value - base for value, base in zip(full, ref_full)]
        target_delta = [value - base for value, base in zip(target, ref_target)]
        full_lo, full_hi = bootstrap_mean_ci(full_delta, args.seed, args.bootstrap)
        target_lo, target_hi = bootstrap_mean_ci(target_delta, args.seed, args.bootstrap)
        rows.append(
            {
                "label": result["label"],
                "tokens_seen": result["tokens_seen"],
                "full_passage_nll": result["full_passage"]["token_weighted_nll"],
                "full_delta_vs_base": sum(full_delta) / len(full_delta),
                "full_delta_lo": full_lo,
                "full_delta_hi": full_hi,
                "final_word_nll": result["final_word"]["mean_word_nll"],
                "final_word_delta_vs_base": sum(target_delta) / len(target_delta),
                "final_word_delta_lo": target_lo,
                "final_word_delta_hi": target_hi,
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
    prep.add_argument("--dataset", default="EleutherAI/lambada_openai")
    prep.add_argument("--split", default="test")
    prep.add_argument("--tokenizer", default="gpt2")
    prep.add_argument("--out", type=Path, required=True)
    prep.set_defaults(func=prepare)

    run = sub.add_parser("evaluate")
    run.add_argument("--model", required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--tokens-seen", type=int, required=True)
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=int, default=32)
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
