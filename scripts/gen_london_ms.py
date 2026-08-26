"""Generate london ms probes at 32k/64k/128k, following the existing precedent.

The london arm has no experiment rows and cannot get them: build_experiments_csv
asserts dataset == "smollm2" for every admitted row, which is a stated design
rule about what belongs in the paper grid. That ruling is Lucia's.

But london already runs OUTSIDE the grid. london16k_bs256_adamw and
london16k_bs256_muon exist as run directories with their own configs and produced
ms 0.9867 and 0.8547 without ever being rows. This follows that precedent rather
than changing the rule: same shape, same fd_step, same seed, at the lrs the
london sweeps actually chose.

ms needs three trainings and no bank, so it runs at sizes where 100 retrained
models are out of reach -- which is the whole reason the london comparison can be
carried to 128k at all.

lrs are the measured interior winners from tuning.csv:

    32k  8e-4      64k  1.6e-3      128k  1.6e-3

    python gen_london_ms.py            # write configs
    python gen_london_ms.py --list     # show what would be written
"""
import argparse
import copy
import os
import sys

import yaml

AP = argparse.ArgumentParser()
AP.add_argument("--list", action="store_true")
args = AP.parse_args()

TEMPLATE = ("/mnt/ssd-2/lucia/paper_runs/experiments/"
            "london16k_bs256_adamw/ms.yaml")
EXP = "/mnt/ssd-2/lucia/paper_runs/experiments"
MIRROR = "/mnt/ssd-2/lucia/datasets_local"

# (n_docs, optimizer, lr) -- lrs are the interior winners measured on
# london_heldout_4k, not retyped from the sweep design.
PLAN = [
    (32000, "adamw", 8e-4), (32000, "muon", 8e-4),
    (64000, "adamw", 1.6e-3), (64000, "muon", 1.6e-3),
    (128000, "adamw", 1.6e-3), (128000, "muon", 1.6e-3),
]

base = yaml.safe_load(open(TEMPLATE))


def setk(node, key, value):
    """Set every occurrence of key, wherever it sits in the config."""
    hit = False
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                node[k] = value
                hit = True
            elif setk(v, key, value):
                hit = True
    elif isinstance(node, list):
        for v in node:
            if setk(v, key, value):
                hit = True
    return hit


written = []
for n, opt, lr in PLAN:
    tag = "london%dk_bs256_%s" % (n // 1000, opt)
    root = os.path.join(EXP, tag)
    data = "%s/london_%dk.hf" % (MIRROR, n // 1000)
    if not os.path.isdir(data):
        print("SKIP %s: %s missing" % (tag, data), file=sys.stderr)
        continue

    cfg = copy.deepcopy(base)
    assert setk(cfg, "dataset", data), "no dataset key in template"
    assert setk(cfg, "optimizer", opt), "no optimizer key in template"
    assert setk(cfg, "lr", lr), "no lr key in template"
    ms_path = os.path.join(root, "ms")
    setk(cfg, "run_path", ms_path)
    cfg["run_path"] = ms_path

    out = os.path.join(root, "ms.yaml")
    if args.list:
        print("  %-30s lr=%-8g data=%s" % (tag, lr, os.path.basename(data)))
        continue
    if os.path.isfile(os.path.join(ms_path, "metasmoothness.json")):
        print("  %-30s already has a result, skipping" % tag)
        continue
    os.makedirs(root, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    written.append((tag, out, lr))
    print("  wrote %-30s lr=%-8g %s" % (tag, lr, os.path.basename(data)))

if written:
    print("\nlaunch each with EXACTLY 2 GPUs (world size is part of run identity):")
    print("  PYTHONPATH=/mnt/ssd-1/lucia/bergson-main-paper-429 \\")
    print("  python -s -P -m bergson <run>/ms.yaml")
