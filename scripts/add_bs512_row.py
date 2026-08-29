"""Register the 16k/bs512 Muon row in experiments.csv.

Its config, base, EK-FAC scores and filter all exist and the filter is running --
but the row was never added to experiments.csv, which is what the figures read. So
the delta would have landed on disk and the batch-sweep figure would still have
shown a gap, exactly the way the Muon 128k point stayed invisible.

Fields are copied from the Muon arm's own bs256 row and only what actually differs
is changed, so nothing is retyped from memory.
"""
import csv
import shutil

P = "/mnt/ssd-2/lucia/metasmoothness/experiments.csv"
rows = list(csv.DictReader(open(P)))
cols = list(rows[0].keys())
NEW = "plan_muon_eps1e17_16k_bs512"
if any(r["run_id"] == NEW for r in rows):
    raise SystemExit(f"  {NEW} already registered")

src = next(r for r in rows if r["run_id"] == "sm_muon_eps1e17_16k_bs256")
new = dict(src)
new["run_id"] = NEW
new["status"] = "running"
new["batch_size"] = "512"
new["grad_accum_steps"] = "16"
new["steps"] = str((16000 // 512) * int(src["num_epochs"]))
new["run_dir"] = "/mnt/ssd-2/lucia/paper_runs/experiments/" + NEW
new["source_doc"] = "planned"
new["notes"] = ("Batch axis at 16k, bs512, muon. lr 2e-4 as in the AdamW bs512 row and "
                "both bs256 rows -- the Muon arm has matched AdamW's lr at every batch "
                "size in this sweep. nproc 2 with grad_accum 16 keeps the per-GPU "
                "micro-batch at 16, as bs256 does with accum 8.")
# Results belong to this row's own runs, not the row it was copied from.
for c in cols:
    if c.startswith(("filter_", "magic_", "ekfac_", "ms_")) or c in (
            "metasmoothness", "train_loss", "heldout_loss", "delta_l1", "delta_l2", "bank_dir"):
        new[c] = ""
rows.append(new)
shutil.copy(P, P + ".bak")
with open(P, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"  registered {NEW}: bs={new['batch_size']} accum={new['grad_accum_steps']} "
      f"steps={new['steps']} lr={new['lr']}")
print("  result columns cleared -- they fill from this row's own runs")
