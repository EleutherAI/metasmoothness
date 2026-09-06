#!/usr/bin/env python3
"""Per-query loss of each *untrained base* model, the denominator of the
normalized-QLD figures (scripts/normalized_qld.py).

    python scripts/base_query_losses.py            # fill data/base_query_losses.csv
    python scripts/base_query_losses.py --only gpt2/indist --force

The normalized figures plot

    (filtered - random) / (base - unfiltered)

-- the share of what training on the corpus taught the model about a query that
removing the proponents takes back. `base` is the model the run started from,
evaluated on the run's own query set, so it needs one number per (model, query
set, logit_scale) combination and per query, not the single mean that
plot_filter_absolute_losses.py draws as its reference line. Per query matters:
the denominator is a paired within-query difference exactly as the QLD is, and
pairing is the whole reason the QLD interval is ~12x tighter than the absolute
one (queries differ from each other by 1.4 nats, far more than any filter moves
them).

The loss matches bergson's `per_doc_query_losses` (bergson/validate.py): mean
cross-entropy over a document's label tokens, float32 logits, one document per
row (the query sets are `chunk_length: 0`, so there are no packed `doc_ids`).
logit_scale is applied where the row's training config sets one -- it is run
config state never persisted in a checkpoint, and the filter's own baseline_loss
carries it (the scale-0.25 row reads 3.4998 unfiltered, not the ~4.6 an unscaled
evaluation gives), so a base loss without it would not be the same quantity.

Cheap: 20 documents x 512 tokens per combination, seconds on one GPU. Writes only
data/base_query_losses.csv (D23: never write to /mnt/ssd-1).
"""
import argparse
import csv
import os
import pathlib

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "base_query_losses.csv"
DL = "/mnt/ssd-2/lucia/datasets_local"
QWEN = "/mnt/ssd-2/lucia/models/Qwen2.5-1.5B"

# key -> (base model, query set, logit_scale). The key is what a plot script
# names; see normalized_qld.py for which figure row uses which.
COMBOS = {
    "gpt2/indist":        ("gpt2",        f"{DL}/query_20.hf",         1.0),
    "gpt2/heldout":       ("gpt2",        f"{DL}/query_20_heldout.hf", 1.0),
    "gpt2-medium/indist": ("gpt2-medium", f"{DL}/query_20.hf",         1.0),
    "gpt2@0.5/indist":    ("gpt2",        f"{DL}/query_20.hf",         0.5),
    "gpt2@0.25/indist":   ("gpt2",        f"{DL}/query_20.hf",         0.25),
    "qwen15b/indist":     (QWEN,          f"{DL}/query_20_qwen.hf",    1.0),
}
FIELDS = ["key", "model", "queries", "logit_scale", "query", "loss"]


@torch.no_grad()
def query_losses(model_id, queries, logit_scale, device, batch_size=8):
    """[loss per query document], bergson per_doc_query_losses convention."""
    ds = load_from_disk(queries)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval().to(device)
    out = []
    for i in range(0, len(ds), batch_size):
        x = torch.tensor(ds[i:i + batch_size]["input_ids"], device=device)
        logits = model(input_ids=x).logits * logit_scale
        tok = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(),
                              x[:, 1:].flatten(), reduction="none").view(x.shape[0], -1)
        out += tok.mean(dim=1).tolist()
    model.to("cpu")
    del model
    torch.cuda.empty_cache()
    return out


def load(path=OUT):
    """{key: {query: loss}} from the cache, empty when it does not exist."""
    got = {}
    if os.path.isfile(path):
        for r in csv.DictReader(open(path, newline="")):
            got.setdefault(r["key"], {})[int(r["query"])] = float(r["loss"])
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help=f"subset of {sorted(COMBOS)}")
    ap.add_argument("--force", action="store_true", help="recompute cached keys")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()

    keys = a.only or list(COMBOS)
    bad = [k for k in keys if k not in COMBOS]
    assert not bad, f"unknown key(s) {bad}; have {sorted(COMBOS)}"

    have = load()
    rows = [dict(zip(FIELDS, (k, *COMBOS[k][:3], q, v)))
            for k, per_q in have.items() for q, v in sorted(per_q.items())
            if k in COMBOS]
    for k in keys:
        if k in have and not a.force:
            print(f"  {k:20s} cached ({len(have[k])} queries)")
            continue
        model_id, queries, scale = COMBOS[k]
        losses = query_losses(model_id, queries, scale, a.device, a.batch_size)
        rows = [r for r in rows if r["key"] != k]
        rows += [dict(zip(FIELDS, (k, model_id, queries, scale, q, v)))
                 for q, v in enumerate(losses)]
        print(f"  {k:20s} {len(losses)} queries  mean={sum(losses) / len(losses):.6f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["key"], int(r["query"]))))
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
