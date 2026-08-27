"""Emit bank_build.yaml for a row that has scores but no retrain bank.

Builds the bank the cheap way: `validate` with method: lds and save_models: true,
pointed at scores that already exist. That skips MAGIC scoring entirely, which is
serial and unshardable and costs 38-112 h on the larger rows.

Hyperparameters are copied from the row's OWN experiment.yaml magic step, never
retyped, so the retrains match the model the bank is meant to explain. This is the
same rule gen_ms.py follows and the reason its probes are comparable.

Two settings that are not negotiable and are forced here:

    save_mode: interval, save_interval: 10**9
        the default sqrt mode writes ~50 GB of trajectory checkpoints that a bank
        never reads
    method: lds, save_models: true
        save_models is what makes it a bank rather than a scoring pass

    python gen_bank.py <run_id> [--nproc N] [--out-root /mnt/ssd-2]

Prints the launch line. Does not launch: world size and GPU type are part of run
identity (D17), so the caller picks the node deliberately.
"""
import argparse
import copy
import os
import sys
from pathlib import Path

import yaml

AP = argparse.ArgumentParser()
AP.add_argument("run_id")
AP.add_argument("--nproc", type=int, default=None,
                help="default: whatever the experiment used")
AP.add_argument("--out-root", default=None,
                help="volume for the bank, e.g. /mnt/ssd-2 (default: beside the run)")
AP.add_argument("--num-subsets", type=int, default=100)
AP.add_argument("--subset-fraction", type=float, default=0.01)
AP.add_argument("--shard", nargs=2, type=int, default=None, metavar=("START", "STOP"),
                help="emit bank_shard_START_STOP.yaml that resumes into an existing "
                     "bank instead of bank_build.yaml. The bank_build run must have "
                     "created subsets.json first -- every shard must remove the SAME "
                     "documents, so they all share one subsets.json.")
args = AP.parse_args()

root = None
for b in ("/mnt/ssd-2", "/mnt/ssd-1"):
    p = Path(b) / "lucia/paper_runs/experiments" / args.run_id
    if p.is_dir():
        root = p
        break
if root is None:
    sys.exit("no run dir for %s" % args.run_id)

exp = root / "experiment.yaml"
if not exp.is_file():
    sys.exit("no experiment.yaml at %s" % exp)

scores = root / "ekfac_scores" / "scores"
if not (scores / "info.json").is_file():
    sys.exit("refusing: no finished EK-FAC scores at %s -- this route needs them" % scores)

cfg = yaml.safe_load(open(exp))


def find_magic_step(node):
    """The magic/validate step carries the training hyperparameters we need."""
    if isinstance(node, dict):
        if "batch_size" in node and ("optimizer" in node or "lr_schedule" in node):
            return node
        for v in node.values():
            r = find_magic_step(v)
            if r:
                return r
    elif isinstance(node, list):
        for v in node:
            r = find_magic_step(v)
            if r:
                return r
    return None


step = find_magic_step(cfg)
if step is None:
    sys.exit("could not find a training step in %s" % exp)

step = copy.deepcopy(step)

out_root = Path(args.out_root) / "lucia/paper_runs/experiments" / args.run_id \
    if args.out_root else root
bank = out_root / "bank_from_filter"

step["subset_fraction"] = args.subset_fraction
step["num_subsets"] = args.num_subsets
step["query_method"] = step.get("query_method", "none")
step["method"] = "lds"
step["save_models"] = True
step["scores"] = str(scores)
step["run_path"] = str(bank)
# A bank never reads the trajectory; the default sqrt mode would write ~50 GB of it.
step["save_mode"] = "interval"
step["save_interval"] = 10 ** 9
step.pop("resume", None)
step.pop("overwrite", None)

if args.nproc is not None:
    dist = step.get("dist") if isinstance(step.get("dist"), dict) else None
    if dist is not None:
        dist["nproc_per_node"] = args.nproc
    else:
        step.setdefault("dist", {})["nproc_per_node"] = args.nproc

# The experiment's step is a Magic config and carries fields Validate does not
# accept (cleanup_ckpts, skip_validation), which fails at instantiation rather
# than at parse time. Introspecting Validate is the obvious filter and it is
# WRONG here -- it drops `method` and `dist`, both of which the working template
# carries. So use a config that is known to run as the schema instead.
TEMPLATE = Path("/mnt/ssd-1/lucia/paper_runs/experiments/"
                "plan_adam_eps1e17_32k_bs32/bank_build.yaml")
if TEMPLATE.is_file():
    tmpl = yaml.safe_load(open(TEMPLATE))["steps"][0]["validate"]
    allowed = set(tmpl)
    dropped = sorted(set(step) - allowed)
    missing = sorted(allowed - set(step))
    step = {k: v for k, v in step.items() if k in allowed}
    # anything the template needs and the row did not supply, take from template
    for k in missing:
        step[k] = copy.deepcopy(tmpl[k])
    if dropped:
        print("dropped %d field(s) not in the template: %s"
              % (len(dropped), ", ".join(dropped)))
    if missing:
        print("filled %d field(s) from the template: %s"
              % (len(missing), ", ".join(missing)))
else:
    print("WARNING: template %s absent; emitting unfiltered" % TEMPLATE, file=sys.stderr)

# re-apply the bank-defining settings, in case the template supplied its own
step["subset_fraction"] = args.subset_fraction
step["num_subsets"] = args.num_subsets
step["method"] = "lds"
step["save_models"] = True
step["scores"] = str(scores)
step["run_path"] = str(bank)
step["save_mode"] = "interval"
step["save_interval"] = 10 ** 9
if args.nproc is not None and isinstance(step.get("dist"), dict):
    step["dist"]["nproc_per_node"] = args.nproc

# Point datasets at the ssd-2 mirror. The experiment configs reference
# /mnt/ssd-1/lucia/bergson-damping/runs/ekfac_vs_n/datasets/, and ssd-1 is full:
# a path lookup there stalls in the kernel (wchan=walk_component) with zero CPU,
# zero read_bytes and zero GPU. That looks exactly like a training hang and is
# not one -- `ls -d` on the same path times out too, so it is the filesystem.
# gen_filter.py and gen_ms.py already do this rewrite; this is the same.
MIRROR = "/mnt/ssd-2/lucia/datasets_local/"
for _key in ("data", "query"):
    _node = step.get(_key)
    if isinstance(_node, dict):
        _old = _node.get("dataset", "")
        if _old and not _old.startswith(MIRROR):
            _new = MIRROR + _old.rstrip("/").split("/")[-1]
            if os.path.isdir(_new):
                _node["dataset"] = _new
                print("  %s -> %s" % (_key, _new))
            else:
                print("  WARNING: no mirror for %s (%s); leaving as-is"
                      % (_key, _old), file=sys.stderr)


if args.shard is not None:
    a, b = args.shard
    # A shard is a *resume* into a bank some other process created. Without these
    # three keys the second shard dies with FileExistsError on the shared run_path;
    # and if it somehow did not, it would generate its own subsets and remove
    # different documents than its peers -- a silently corrupt bank.
    step["subset_start"] = a
    step["subset_stop"] = b
    step["subsets"] = str(bank / "subsets.json")
    step["resume"] = True
    step["overwrite"] = False

doc = {"steps": [{"validate": step}], "run_path": str(bank)}

out = root / ("bank_shard_%d_%d.yaml" % tuple(args.shard)
              if args.shard is not None else "bank_build.yaml")
if out.exists():
    sys.exit("refusing: %s already exists" % out)
with open(out, "w") as f:
    yaml.safe_dump(doc, f, sort_keys=False)


def deep_get(node, key):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                return v
            r = deep_get(v, key)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = deep_get(v, key)
            if r is not None:
                return r
    return None


nproc = deep_get(doc, "nproc_per_node")
print("wrote %s" % out)
print("  model=%s optimizer=%s bs=%s epochs=%s nproc=%s"
      % (deep_get(doc, "model"), doc.get("optimizer"), doc.get("batch_size"),
         doc.get("num_epochs"), nproc))
print("  subsets=%d frac=%s  bank -> %s" % (args.num_subsets, args.subset_fraction, bank))
print("  scores  <- %s" % scores)
print()
print("launch with EXACTLY %s GPU(s), on the SAME GPU TYPE the row ran on (D17):" % nproc)
print("  cd /tmp && setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\")
print("    CUDA_VISIBLE_DEVICES=<devs> MASTER_PORT=<port> PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \\")
print("    PYTHONPATH=/mnt/ssd-1/lucia/bergson-main-paper-429 \\")
print("    /home/lucia/envs/paper/bin/python -s -P -m bergson %s \\" % out)
print("    > %s/bank_build.log 2>&1 < /dev/null &" % root)
