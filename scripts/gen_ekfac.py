"""Generate an EK-FAC scoring config for a completed bank.

EK-FAC is scoring-only: reuse rule 1 says the bank is scorer-independent, so
this adds an ekfac_lds cell without rebuilding anything. The template is the
config lotus-0 used for the 4k pair -- the only accepted EK-FAC cells in the
grid -- so the D7 settings (kfac, ev_correction, damped_inverse, damping 0.1)
are inherited rather than retyped.

    python gen_ekfac.py <run_id> [--nproc 2]
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

# `python -P` keeps a bergson checkout's cwd off sys.path, but it also drops
# this script's own directory -- put just that one entry back for the import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_config  # noqa: E402

TEMPLATE = Path(
    "/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_4k_bs256"
    "/ekfac_scores/config.yaml"
)

ap = argparse.ArgumentParser()
ap.add_argument("run_id")
ap.add_argument("--nproc", type=int, default=2)
ap.add_argument("--no-bank", action="store_true",
    help="score a row that has no retrain bank, using base/model")
args = ap.parse_args()

root = None
for base in ("/mnt/ssd-2", "/mnt/ssd-1"):
    cand = Path(base) / "lucia/paper_runs/experiments" / args.run_id
    if cand.is_dir():
        root = cand
        break
if root is None:
    sys.exit(f"run dir not found: {args.run_id}")

# EK-FAC scoring needs a MODEL and the training data, not a bank -- the bank is
# only what an LDS is computed against afterwards. The step-ladder rows are
# registered bank-free on purpose (MAGIC would cost 150h+ of scoring at that N),
# and they still want scores so the proponent filter can rank documents. So
# --no-bank takes the model from base/model, produced by the row's base.yaml,
# and skips the bank precondition. It cannot produce an ekfac_lds, and is not
# meant to.
if args.no_bank:
    base_model = root / "base" / "model"
    if not base_model.is_dir():
        sys.exit(f"refusing: --no-bank needs a trained model at {base_model}; "
                 f"run the row's base.yaml first")
else:
    n_models = len(list((root / "retrained").glob("subset_*")))
    if n_models < 100:
        sys.exit(f"refusing: bank is {n_models}/100, EK-FAC needs the finished "
                 f"bank (or pass --no-bank to score a bank-free row)")
    base_model = root / "retrained" / "base"
    if not base_model.is_dir():
        sys.exit(f"refusing: no retrained/base in {root}")

# The training dataset and world size come from the row's own magic config, so
# EK-FAC scores the same data the bank was built on.
exp = run_config.load(root)
magic = next(s["magic"] for s in exp["steps"] if "magic" in s)
train_ds = magic["data"]["dataset"]
query_ds = magic["query"]["dataset"]

cfg = copy.deepcopy(yaml.safe_load(TEMPLATE.read_text()))
ek = cfg["steps"][0]["ekfac"]
idx = ek["index_cfg"]
idx["run_path"] = str(root / "ekfac_scores")
idx["model"] = str(base_model)
idx["data"]["dataset"] = train_ds
idx["distributed"]["nproc_per_node"] = args.nproc
ek["hessian_pipeline_cfg"]["query"]["dataset"] = query_ds

out = root / "ekfac.yaml"
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
inv = ek["preprocess_cfg"]["inversion_cfg"]
print(f"wrote {out}")
print(f"  model={base_model}")
print(f"  train={train_ds}")
print(f"  D7: inversion={inv['inversion']} damping={inv['damping_factor']} "
      f"method={ek['hessian_cfg']['method']} ev_correction={ek['hessian_cfg']['ev_correction']}")
