"""Generate a metasmoothness probe config for a row, from that row's own training config.

ms (arXiv 2503.13751 Def. 2) trains three models with data weights 1, 1+h*v and
1+2h*v and scores movement-weighted sign agreement of consecutive
finite-difference derivatives. Three trainings, NO retraining bank -- so it can
run on idle GPUs while a bank is unavailable or unfinished.

The training hyperparameters are taken from the row's own experiment.yaml magic
step, so the probe measures the configuration the bank was actually built with
rather than a retyped approximation. Only fields MetasmoothnessConfig actually
accepts are emitted: it is TrainingConfig plus fd_step and direction_seed.

NOTE the field is `fd_step`, not `h`. The older msfill_* templates on disk use
`h`, which current bergson rejects at parse time -- the same failure mode that
killed the gpt2-medium tuning sweep when a config carried an unknown field.

    python gen_ms.py <run_id> [--nproc 2] [--fd-step 0.1] [--direction-seed 0]
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import yaml

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--bergson", default="/mnt/ssd-1/lucia/bergson-main-paper-429")
_known, _ = _pre.parse_known_args()
# The field set is read from the SAME checkout the run will import, so a row whose
# config carries logit_scale must point at a checkout that has it -- otherwise the
# field is silently dropped and the probe measures an unscaled model.
sys.path.insert(0, _known.bergson)
from bergson.config.config import MetasmoothnessConfig  # noqa: E402

EXP = ["/mnt/ssd-2/lucia/paper_runs/experiments",
       "/mnt/ssd-1/lucia/paper_runs/experiments"]

ap = argparse.ArgumentParser(parents=[_pre])
ap.add_argument("run_id")
ap.add_argument("--nproc", type=int, default=2)
ap.add_argument("--fd-step", type=float, default=0.1)      # CONTROLS: fd_step 0.1
ap.add_argument("--direction-seed", type=int, default=0)   # CONTROLS: direction_seed 0
args = ap.parse_args()

root = None
for base in EXP:
    if (Path(base) / args.run_id).is_dir():
        root = Path(base) / args.run_id
        break
if root is None:
    sys.exit(f"run dir not found: {args.run_id}")

exp = yaml.safe_load((root / "experiment.yaml").read_text())
magic = next(s["magic"] for s in exp["steps"] if "magic" in s)

valid = {f.name for f in dataclasses.fields(MetasmoothnessConfig)}
# Carry over every training field the probe shares with the row, and nothing else.
skip = {"run_path", "query", "num_subsets", "subset_fraction", "query_method",
        "skip_validation", "save_models", "save_mode", "save_optimizer_state",
        "cleanup_ckpts", "resume", "double_backward_batch_size", "train_mode"}
cfg = {k: v for k, v in magic.items() if k in valid and k not in skip}

cfg["run_path"] = str(root / "ms")
cfg["overwrite"] = True
cfg["fd_step"] = args.fd_step
cfg["direction_seed"] = args.direction_seed
cfg.setdefault("distributed", {})
cfg["distributed"] = dict(cfg["distributed"], nproc_per_node=args.nproc, nnode=1)

dropped = sorted(set(magic) - set(cfg) - skip)
out = root / "ms.yaml"
out.write_text(yaml.safe_dump({"steps": [{"metasmoothness": cfg}]}, sort_keys=False))
print(f"wrote {out}")
print(f"  model={cfg.get('model')} optimizer={cfg.get('optimizer')} "
      f"bs={cfg.get('batch_size')} lr={(cfg.get('lr_schedule') or {}).get('lr')}")
print(f"  fd_step={cfg['fd_step']} direction_seed={cfg['direction_seed']} nproc={args.nproc}")
if dropped:
    print(f"  not accepted by MetasmoothnessConfig, omitted: {dropped}")
