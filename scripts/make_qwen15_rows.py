"""Configs for the Qwen2.5-1.5B replication of figure 1, both panels.

1.5B is the largest size that fits an A40 pair under plain DDP -- 3B OOMs at 48 GB
and needs FSDP, 7B needs a whole 8-GPU node. That is what makes this tractable:
a 2-GPU run means 60 concurrent jobs on this fleet instead of 15.

Per-rank micro-batch is pinned to 4, the value the 7B ran at, so grad_accum is
derived rather than inherited: micro = batch / (nproc * accum), and a config
carried over from a 2-GPU gpt2 row would silently quadruple the activation memory.

Usage:
    make_qwen15_rows.py sweep [N]          lr sweep for one N rung
    make_qwen15_rows.py rows <lr>          one train-only row per N, after heldout selection

Qwen chain dependency:
    1. Train the `tune_qwen15b_*` sweep configs.
    2. Run `scripts/heldout_eval.py --heldout /mnt/ssd-2/lucia/datasets_local/heldout_4k_qwen_v2.hf .../retrained/base`
       for every completed sweep point. The training logs only contain scheduler steps; they
       do not contain the heldout objective.
    3. Select lr by heldout CE, using the same rule as `build_tuning_csv.py`: if an endpoint
       wins, add one octave beyond it; only launch downstream Qwen rows once the winner is
       interior/bracketed.
    4. Record the measured heldout losses and selected lr before launching filter/EK-FAC phases.
"""
import sys
from pathlib import Path

import yaml

C = Path("/mnt/ssd-2/lucia/metasmoothness/configs/experiments")
DS = "/mnt/ssd-2/lucia/datasets_local"
MODEL = "/mnt/ssd-2/lucia/models/Qwen2.5-1.5B"
NS = [4000, 8000, 16000, 32000, 64000, 128000, 256000]
# micro-batch 2, not 4: at 4 the 1.5B OOMs on a 48GB A40 whenever the node is
# not otherwise empty (the standalone probe fit, the fleet-wide launch did not).
NPROC, MICRO = 2, 2


def shape(n, lr):
    doc = yaml.safe_load((C / "plan_adam_eps1e17_4k_bs256.yaml").read_text())
    m = doc["steps"][0]["magic"]
    m["model"] = MODEL
    m.pop("model_kwargs", None)              # GPT-2 dropout names; Qwen rejects them
    m["precision"] = "bf16"
    m["fsdp"] = False                        # 1.5B fits per-rank; FSDP is pure overhead
    m["distributed"]["nproc_per_node"] = NPROC
    m["grad_accum_steps"] = max(1, m["batch_size"] // (NPROC * MICRO))
    m["data"]["dataset"] = f"{DS}/train_{n//1000}k_qwen.hf"
    m["query"]["dataset"] = f"{DS}/query_20_qwen.hf"
    m["lr_schedule"]["lr"] = lr
    m["lr_schedule"]["lr_end"] = lr / 10
    m["num_subsets"] = 0                     # no 100-retrain bank; filters need scores
    m["save_models"] = True
    m["skip_validation"] = True
    m["save_mode"] = "interval"
    m["save_interval"] = (n // m["batch_size"]) * m["num_epochs"]
    return doc, m


if sys.argv[1] == "sweep":
    # lr is tuned per N, as the gpt2 chain does (tuning.csv has a sweep_group per
    # rung). Pass an N to sweep that rung; default 4000.
    n_sweep = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    for lr in (2e-5, 5e-5, 1e-4, 2e-4):
        doc, m = shape(n_sweep, lr)
        tag = f"{n_sweep//1000}k"
        run = f"/mnt/ssd-2/lucia/paper_runs/tuning/tune_qwen15b_{tag}_bs256_lr{lr:g}_s42"
        m["run_path"] = run
        doc["run_path"] = run
        p = C / f"tune_qwen15b_{tag}_bs256_lr{lr:g}_s42.yaml"
        p.write_text(yaml.safe_dump(doc, sort_keys=False))
        print(f"  {p.name}  lr={lr:g} accum={m['grad_accum_steps']} steps={m['save_interval']}")
else:
    lr = float(sys.argv[2])
    for n in NS:
        doc, m = shape(n, lr)
        rid = f"qwen15b_{n//1000}k_bs256"
        run = f"/mnt/ssd-2/lucia/paper_runs/experiments/{rid}"
        m["run_path"] = run
        doc["run_path"] = run
        (C / f"{rid}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
        print(f"  {rid}.yaml  steps={m['save_interval']} accum={m['grad_accum_steps']}")
