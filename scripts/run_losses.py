#!/usr/bin/env python3
"""Final train and held-out cross-entropy for each experiment row, from its own checkpoint.

Fills train_loss / heldout_loss, the last cells required by the per-run diagnostics rule
(see EXPERIMENTS_CSV.md). Evaluates the run's OWN final checkpoint rather than the tuning
winner's saved model: reuse rule 3 makes them the same weights, but sweep models get
cleaned up (both 16k anchors' are already gone) whereas the run checkpoints are kept.

    python run_losses.py --csv experiments.csv [--device cuda:0] [--only RUN_ID ...]

logit_scale is read per row and applied, because it is run-config state never persisted in
the checkpoint -- evaluating a scale-0.5 model unscaled reads ~4.6 against its true ~3.3
(see scripts/heldout_eval.py). Stdlib + torch/transformers; writes nothing.
"""
import argparse, csv, re, sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout_eval import heldout_loss

hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()

DATA = Path("/mnt/ssd-1/lucia/bergson-damping/runs/ekfac_vs_n/datasets")


def final_ckpt(run_dir: Path) -> Path:
    steps = {int(m.group(1)): p for p in (run_dir / "checkpoints").glob("step_*.ckpt")
             if (m := re.fullmatch(r"step_(\d+)\.ckpt", p.name))}
    if not steps:
        raise FileNotFoundError(f"{run_dir}: no step_*.ckpt")
    return steps[max(steps)]


def load_model(run_dir: Path, base: str):
    sd = _load_state_dict_from_keys(checkpoint_id=str(final_ckpt(run_dir)))
    # The final checkpoint also carries optimizer state (D8: save_optimizer_state=last)
    # and rng/batch bookkeeping; keep only the model params.
    sd = {k: v for k, v in sd.items()
          if isinstance(v, torch.Tensor) and not k.startswith(("opt_state/", "."))}
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(base))
    model = model.to(torch.float32)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # GPT-2 ties lm_head to wte, so lm_head.weight is legitimately absent from the ckpt.
    real = [k for k in missing if k != "lm_head.weight"]
    assert not real and not unexpected, (
        f"{run_dir}: checkpoint/model key mismatch; missing={real} unexpected={list(unexpected)}")
    model.tie_weights()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=Path(__file__).resolve().parent.parent / "experiments.csv")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--only", nargs="*", help="restrict to these run_ids")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv, newline="")) if r["status"] == "done"]
    if a.only:
        rows = [r for r in rows if r["run_id"] in set(a.only)]

    cache = {}
    def ids_for(name):
        if name not in cache:
            cache[name] = torch.tensor(load_from_disk(str(DATA / name))["input_ids"])
        return cache[name]

    print(f"{'run':34s} {'scale':>5s} {'train':>8s} {'heldout':>8s}")
    for r in rows:
        run_dir, n = Path(r["run_dir"]), int(float(r["n_docs"]))
        scale = float(r["logit_scale"] or 1.0)
        try:
            model = load_model(run_dir, r["model"])
            tr = heldout_loss(model, ids_for(f"train_{n // 1000}k.hf"), a.device, a.batch_size, scale)
            ho = heldout_loss(model, ids_for("heldout_4k.hf"), a.device, a.batch_size, scale)
        except Exception as e:
            print(f"{r['run_id']:34s} {type(e).__name__}: {e}", file=sys.stderr)
            continue
        print(f"{r['run_id']:34s} {scale:5.2f} {tr:8.4f} {ho:8.4f}", flush=True)
        del model


if __name__ == "__main__":
    main()
