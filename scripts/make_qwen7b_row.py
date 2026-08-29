"""Config for the 7B proponent-filter row, and its lr sweep.

Derived from plan_adam_eps1e17_4k_bs256 -- the smallest existing row, 32 steps --
because 23 retrains of a 7B is the cost that matters, not the step count.

Four things must change and each would break the run silently if missed:

  model_kwargs   resid_pdrop/attn_pdrop/embd_pdrop are GPT-2 argument names. Qwen
                 rejects them, and an earlier row lost its architecture entirely by
                 dropping `model` instead of fixing kwargs.
  precision      fp32 7B is ~28 GB of parameters before optimizer state. bf16.
  fsdp           without it every rank holds a full model and optimizer; on 48 GB
                 A40s that cannot fit at 7B. DDP is the default.
  lr             a 124M model's lr will destroy a 7B. Swept, selected on held-out.

Usage: make_qwen7b_row.py sweep|row [lr]
"""
import sys
from pathlib import Path

import yaml

C = Path("/mnt/ssd-2/lucia/metasmoothness/configs/experiments")
T = Path("/mnt/ssd-2/lucia/paper_runs/tuning")
DS = "/mnt/ssd-2/lucia/datasets_local"
MODEL = "/mnt/ssd-2/lucia/models/Qwen2.5-7B"
mode = sys.argv[1]


def base_doc():
    doc = yaml.safe_load((C / "plan_adam_eps1e17_4k_bs256.yaml").read_text())
    m = doc["steps"][0]["magic"]
    m["model"] = MODEL
    m.pop("model_kwargs", None)          # GPT-2 dropout names; Qwen rejects them
    m["precision"] = "bf16"
    m["fsdp"] = True
    m["distributed"]["nproc_per_node"] = 8
    m["data"]["dataset"] = f"{DS}/train_4k_qwen.hf"
    m["query"]["dataset"] = f"{DS}/query_20_qwen.hf"
    m["num_subsets"] = 0
    m["save_models"] = True
    m["skip_validation"] = True
    m["save_mode"] = "interval"
    m["save_interval"] = (4000 // m["batch_size"]) * m["num_epochs"]
    return doc, m


if mode == "sweep":
    for lr in (5e-6, 1e-5, 2e-5, 4e-5):
        doc, m = base_doc()
        m["lr_schedule"]["lr"] = lr
        m["lr_schedule"]["lr_end"] = lr / 10
        run = f"/mnt/ssd-2/lucia/paper_runs/tuning/tune_qwen7b_4k_bs256_lr{lr:g}_s42"
        m["run_path"] = run
        doc["run_path"] = run
        p = C / f"tune_qwen7b_4k_bs256_lr{lr:g}_s42.yaml"
        p.write_text(yaml.safe_dump(doc, sort_keys=False))
        print(f"  {p.name}  lr={lr:g}  steps={m['save_interval']}  fsdp={m['fsdp']} "
              f"precision={m['precision']} nproc={m['distributed']['nproc_per_node']}")
else:
    lr = float(sys.argv[2])
    doc, m = base_doc()
    m["lr_schedule"]["lr"] = lr
    m["lr_schedule"]["lr_end"] = lr / 10
    run = "/mnt/ssd-2/lucia/paper_runs/experiments/qwen7b_4k_bs256"
    m["run_path"] = run
    doc["run_path"] = run
    p = C / "qwen7b_4k_bs256.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"  wrote {p.name} lr={lr:g}")
