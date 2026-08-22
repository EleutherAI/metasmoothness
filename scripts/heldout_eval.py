"""Held-out cross-entropy for one or more models — the paper's model-selection metric.

Never select lr (or report generalisation) on train loss: on repeated small corpora the
train-loss optimum memorises and can generalise worse than untrained GPT-2. This evaluates
mean per-token CE on a fixed held-out set that is verified disjoint from every train_N.

Usage:
    python heldout_eval.py MODEL_DIR [MODEL_DIR ...]        # HF dirs or hub names
    python heldout_eval.py --heldout PATH gpt2 runs/*/model
"""

import argparse

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()


@torch.no_grad()
def heldout_loss(model, ids: torch.Tensor, device: str, batch_size: int = 32,
                 logit_scale: float = 1.0) -> float:
    """logit_scale must match the run config's value for models trained with the
    bergson logit-scale hook - the scale is run-config state, never persisted in
    the checkpoint, so raw logits of a scaled model are miscalibrated by design
    (evaluating a scale-0.5 model unscaled reads ~4.6 heldout vs its true ~3.3)."""
    model.eval().to(device)
    tot, ntok = 0.0, 0
    for i in range(0, len(ids), batch_size):
        x = ids[i:i + batch_size].to(device)
        logits = model(x).logits * logit_scale
        tot += torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            x[:, 1:].reshape(-1), reduction="sum").item()
        ntok += x[:, 1:].numel()
    model.to("cpu")
    return tot / ntok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="model dirs or hub names")
    ap.add_argument("--heldout",
                    default="/mnt/ssd-1/lucia/bergson-damping/runs/ekfac_vs_n/datasets/heldout_4k.hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--logit-scale", type=float, default=1.0)
    args = ap.parse_args()

    ds = load_from_disk(args.heldout)
    assert isinstance(ds, Dataset)
    ids = torch.tensor(ds["input_ids"])
    print(f"{'model':40} heldout CE  ({len(ds)} docs x {ids.shape[1]} tok)")
    for m in args.models:
        model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32)
        print(f"{m:40} {heldout_loss(model, ids, args.device, args.batch_size, args.logit_scale):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
