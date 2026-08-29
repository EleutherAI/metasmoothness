"""Second-scorer config for the 256k row -- the last filter_method_appendix gap.

That row has a base model but NO trajectory, and this scorer replays the
trajectory, so training has to happen first. save_mode log is what writes the
trajectory; save_interval must not be 1e9 or the run keeps nothing usable.

Once the trajectory exists, shard_magic_queries.py splits the 20 query backwards
across the fleet -- unsharded they are one backward per query and run for days.
"""
import sys
from pathlib import Path

import yaml

ROOT = Path("/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_256k_bs256")
doc = yaml.safe_load((ROOT / "experiment.yaml").read_text())
step = doc["steps"][0]
m = step[next(iter(step))]
steps = (256000 // m["batch_size"]) * m["num_epochs"]

m["num_subsets"] = 0          # no bank; this figure needs scores, not an LDS
m["skip_validation"] = True
m["save_models"] = True
m["save_mode"] = "log"        # writes the trajectory the backward replays
m.pop("save_interval", None)
m["resume"] = True
m["overwrite"] = False
out = ROOT / "magic_scores"
m["run_path"] = str(out)
doc["run_path"] = str(out)
p = ROOT / "magic_scores.yaml"
if p.exists():
    sys.exit(f"refusing: {p.name} exists")
p.write_text(yaml.safe_dump(doc, sort_keys=False))
print(f"  wrote {p.name}: {steps} steps, save_mode=log, num_subsets=0")
print(f"  run_path {out}")
print(f"  nproc {m.get('distributed', {}).get('nproc_per_node')}")
