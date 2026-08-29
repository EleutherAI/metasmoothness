"""Create the Muon 256k row -- the missing point on filter_scaling_appendix.

lr comes from this row's own sweep, which had already run: all six models were on
disk with no heldout_loss recorded, so the selection was done and never harvested.
2e-4 wins at 3.1857 with 1e-4 and 4e-4 either side, an interior minimum, and it
matches the AdamW 256k row's selection.

Train-only shape, as for the bs512 row: this figure needs an EK-FAC delta, not a
100-retrain bank. save_interval = the step count keeps the FINAL checkpoint, which
save_interval 1e9 silently discards -- that cost a re-run on bs512.
"""
import csv
import sys
from pathlib import Path

import yaml

C = Path("/mnt/ssd-2/lucia/metasmoothness/configs/experiments")
src, dst = C / "plan_adam_eps1e17_256k_bs256.yaml", C / "plan_muon_eps1e17_256k_bs256.yaml"
RUN = "/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_256k_bs256"
if dst.exists():
    sys.exit(f"refusing: {dst.name} exists")

doc = yaml.safe_load(src.read_text())
m = doc["steps"][0]["magic"]
assert m["optimizer"] == "adamw" and m["batch_size"] == 256, (m["optimizer"], m["batch_size"])
steps = (256000 // m["batch_size"]) * m["num_epochs"]

m["optimizer"] = "muon"
m["save_optimizer_state"] = "none"      # as the other muon rows
m["lr_schedule"]["lr"] = 2e-4
m["num_subsets"] = 0
m["save_models"] = True
m["skip_validation"] = True
m["save_mode"] = "interval"
m["save_interval"] = steps
m["run_path"] = RUN
doc["run_path"] = RUN
dst.write_text(yaml.safe_dump(doc, sort_keys=False))

# Record the sweep result that was already sitting on disk.
T = Path("/mnt/ssd-2/lucia/metasmoothness/tuning.csv")
rows = list(csv.DictReader(open(T)))
cols = list(rows[0].keys())
LOSS = {"5e-05": "3.2008", "0.0001": "3.1904", "0.0002": "3.1857",
        "0.0004": "3.1909", "0.0008": "3.2114", "0.0016": "3.2487"}
n = 0
for r in rows:
    for lr, v in LOSS.items():
        if r["run_id"] == f"tune_muon_256k_bs256_lr{lr}" and not (r.get("heldout_loss") or "").strip():
            r["heldout_loss"] = v
            n += 1
with open(T, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader(); w.writerows(rows)

print(f"  wrote {dst.name}: muon, lr {m['lr_schedule']['lr']}, {steps} steps, "
      f"save_interval={steps}")
print(f"  recorded {n} heldout loss(es) in tuning.csv")
