"""Check each ms probe trained the SAME configuration as the row's bank.

ms is only comparable to that row's LDS if the three probe trainings used the
row's training hyperparameters. gen_ms.py copies them from the row's own
experiment.yaml, but silently dropping one would leave the probe measuring a
different configuration while looking perfectly healthy.
"""
import glob
import os

import yaml

# Fields that define the training configuration and must match the magic step.
MUST_MATCH = [
    "model", "optimizer", "batch_size", "num_epochs", "seed", "weight_decay",
    "eps_root", "adam_beta1", "adam_beta2", "grad_accum_steps", "precision",
    "use_tf32_matmuls", "max_grad_norm", "logit_scale", "loss_reduction",
]

for root in ("/mnt/ssd-2/lucia/paper_runs/experiments",
             "/mnt/ssd-1/lucia/paper_runs/experiments"):
    for ms_cfg in sorted(glob.glob(f"{root}/*/ms/config.yaml")):
        run = os.path.basename(os.path.dirname(os.path.dirname(ms_cfg)))
        exp = os.path.join(os.path.dirname(os.path.dirname(ms_cfg)), "experiment.yaml")
        if not os.path.exists(exp):
            continue
        magic = next(s["magic"] for s in yaml.safe_load(open(exp))["steps"] if "magic" in s)
        ms = yaml.safe_load(open(ms_cfg))
        ms = ms["steps"][0]["metasmoothness"] if "steps" in ms else ms

        diffs = []
        for k in MUST_MATCH:
            a, b = magic.get(k), ms.get(k)
            if a != b:
                diffs.append(f"{k}: bank={a!r} ms={b!r}")
        lr_a = (magic.get("lr_schedule") or {}).get("lr")
        lr_b = (ms.get("lr_schedule") or {}).get("lr")
        if lr_a != lr_b:
            diffs.append(f"lr: bank={lr_a} ms={lr_b}")
        data_a = (magic.get("data") or {}).get("dataset")
        data_b = (ms.get("data") or {}).get("dataset")
        if data_a != data_b:
            diffs.append(f"dataset: bank={data_a} ms={data_b}")

        status = "OK" if not diffs else "MISMATCH"
        print(f"{run[:44]:44s} {status}")
        for d in diffs:
            print(f"    {d}")
