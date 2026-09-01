"""Timing probes: same row shape at 1.5B / 3B / 7B, to price the curve by model size.

Measure rather than extrapolate. Per-step time is only half the story -- the other
half is how many GPUs a run needs, because that sets how many rows can run at once.
A 7B needs a whole 8-GPU node under FSDP, so the fleet runs 15 concurrent jobs; if a
1.5B fits on 2 GPUs it runs 60. That ratio dominates wall-clock far more than s/it.

Same data, same batch, same steps for all three: Qwen2.5 shares one tokenizer across
sizes, so the datasets built for the 7B are reused byte-identically.

Usage: make_size_probe.py <model-dir> <nproc> <fsdp true|false> <tag>
"""
import sys
from pathlib import Path

import yaml

C = Path("/mnt/ssd-2/lucia/metasmoothness/configs/experiments")
DS = "/mnt/ssd-2/lucia/datasets_local"
model, nproc, fsdp, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "true", sys.argv[4]

doc = yaml.safe_load((C / "plan_adam_eps1e17_4k_bs256.yaml").read_text())
m = doc["steps"][0]["magic"]
m["model"] = model
m.pop("model_kwargs", None)          # GPT-2 dropout names; Qwen rejects them
m["precision"] = "bf16"
m["fsdp"] = fsdp
m["distributed"]["nproc_per_node"] = nproc
# Hold the per-rank micro-batch at 4, which is what the 7B runs at on 8 GPUs.
# micro = batch / (nproc * grad_accum), so halving the GPUs doubles the micro-batch
# unless accum compensates -- that is what OOMed the first 1.5B probe on an 80GB
# A100 at micro-batch 16. Fixing it also makes the sizes comparable: same tokens
# per forward, so the timing difference is the model and not the batch shape.
MICRO = 4
m["grad_accum_steps"] = max(1, m["batch_size"] // (nproc * MICRO))
m["data"]["dataset"] = f"{DS}/train_4k_qwen.hf"
m["query"]["dataset"] = f"{DS}/query_20_qwen.hf"
m["lr_schedule"]["lr"] = 1e-5
m["num_subsets"] = 0
m["save_models"] = False             # a probe; nothing to keep, and ssd-2 is 99% full
m["skip_validation"] = True
m["save_mode"] = "interval"
m["save_interval"] = 10 ** 9
run = f"/mnt/ssd-2/lucia/paper_runs/experiments/_probe_{tag}"
m["run_path"] = run
doc["run_path"] = run
p = C / f"_probe_{tag}.yaml"
p.write_text(yaml.safe_dump(doc, sort_keys=False))
print(f"  {p.name}: model={Path(model).name} nproc={nproc} fsdp={fsdp}")
