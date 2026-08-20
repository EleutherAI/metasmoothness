"""Build tuning.csv — stage-0 hyperparameter selection runs, registered as empty rows.

Per the tuning protocol in CONTROLS.md, every experiments.csv config whose optimization
problem differs from the anchor needs a held-out lr mini-sweep {0.5x, 1x, 2x} BEFORE its
bank is built. This file registers all of those runs so an agent with spare compute can
claim them: train the config, run `scripts/heldout_eval.py`, fill `train_loss` /
`heldout_loss`, edit the row here, regenerate, commit.

Selection rule (per sweep_group): lowest heldout_loss wins; if an endpoint of the 3-point
grid wins, ADD one more row an octave further out and re-check before freezing. When a
group is complete, write the winning lr into the matching experiments.csv row(s) — the
`gates` column names them.

These are train-only runs: no banks, no subsets, no MAGIC. Cost is ~steps x bs; at bs256
that is 31 steps (4k) to 2000 steps (256k).

status: measured = both loss columns filled; empty = to run; blocked = prerequisite
missing (gpt2_custom does not exist yet; warmup500 exceeds the 125-step anchor and needs
the extend-vs-all-warmup decision first).
"""

import csv
import os

COLUMNS = [
    "run_id", "sweep_group", "status", "priority", "gates",
    "model", "arch_mod", "optimizer", "n_docs", "batch_size", "grad_accum_steps",
    "num_epochs", "steps", "warmup", "logit_scale", "weight_decay", "max_grad_norm",
    "eps_root", "seed", "lr",
    "train_loss", "heldout_loss",
    "run_dir", "notes",
]

BASE = dict(model="gpt2", arch_mod="none", optimizer="adamw", n_docs=16000,
            batch_size=256, grad_accum_steps=16, num_epochs=2, warmup=0.25,
            logit_scale=1.0, weight_decay=0.01, max_grad_norm="", eps_root=1e-17,
            seed=42, train_loss="", heldout_loss="", run_dir="", notes="")

MINI = [1e-4, 2e-4, 4e-4]  # {0.5x, 1x, 2x} of the anchor optimum

rows = []


def sweep(group, gates, lrs=MINI, status="empty", priority=2, **cfg):
    for lr in lrs:
        r = dict(BASE)
        r.update(cfg)
        r.update(run_id=f"{group}_lr{lr:g}", sweep_group=group, gates=gates,
                 status=status, priority=priority, lr=lr)
        if not r.get("steps"):
            r["steps"] = round(int(r["n_docs"]) * int(r["num_epochs"]) / int(r["batch_size"]))
        rows.append(r)


# ---------------------------------------------------------------------------------
# 0. MEASURED — the anchor sweeps (16k, bs256, 2ep). Models in /mnt/ssd-2/lucia/s16k_lrsweep.
#    ga=4 on these runs (micro-batch rule met with more devices); ga is heldout-neutral.
# ---------------------------------------------------------------------------------
ANCHOR = {("adamw", 1e-4): (None, 3.2592), ("adamw", 2e-4): (None, 3.2572),
          ("adamw", 4e-4): (None, 3.2670), ("adamw", 8e-4): (None, 3.2990),
          ("adamw", 2e-3): (None, 3.3974),
          ("muon", 1e-4): (None, 3.2649), ("muon", 2e-4): (None, 3.2570),
          ("muon", 4e-4): (None, 3.2660), ("muon", 8e-4): (None, 3.3035),
          ("muon", 2e-3): (None, 3.4198)}
for (opt, lr), (tl, hl) in ANCHOR.items():
    r = dict(BASE)
    r.update(run_id=f"tune_{opt}_16k_anchor_lr{lr:g}", sweep_group=f"tune_{opt}_16k_anchor",
             gates=f"sm_{opt}_eps1e17_16k_bs256", status="measured", priority=1,
             optimizer=opt, grad_accum_steps=4, lr=lr, train_loss=tl or "",
             heldout_loss=hl, steps=125,
             run_dir=f"/mnt/ssd-2/lucia/s16k_lrsweep/s16k_{opt}_lr{lr:g}",
             notes="Anchor sweep, measured 2026-08-20; untrained gpt2 heldout = 3.4981. "
                   "2e-4 selected (interior optimum for both optimizers).")
    rows.append(r)

# ---------------------------------------------------------------------------------
# 1. Token axis (both optimizers). 16k is the measured anchor; every other rung
#    needs its own mini-sweep. ga follows the micro-batch-16 rule at bs256.
# ---------------------------------------------------------------------------------
for opt in ["adamw", "muon"]:
    for n in [4000, 8000, 32000, 64000, 128000, 256000]:
        k = f"{n // 1000}k"
        sweep(f"tune_{opt}_{k}", gates=f"plan_{'adam' if opt == 'adamw' else 'muon'}_eps1e17_{k}_bs256",
              priority=1, optimizer=opt, n_docs=n,
              notes="4k is 31 steps at bs256/2ep — expect noisy selection; see CONTROLS caveat."
                    if n == 4000 else "")

# ---------------------------------------------------------------------------------
# 2. Batch-size axis (adam). Steps co-vary with bs at fixed epochs; lr optimum is
#    expected to shift with bs — this sweep is what makes the axis fair.
# ---------------------------------------------------------------------------------
for bs in [16, 32, 64, 128]:
    sweep(f"tune_adamw_16k_bs{bs}", gates=f"plan_adam_eps1e17_16k_bs{bs}", priority=1,
          batch_size=bs, grad_accum_steps=max(1, bs // 16))

# ---------------------------------------------------------------------------------
# 3. Model size (adam, 16k). Micro-batch 16 may not fit larger models — record the
#    ga actually used; ga is heldout-neutral.
# ---------------------------------------------------------------------------------
for mdl in ["gpt2-medium", "gpt2-large"]:
    sweep(f"tune_adamw_16k_{mdl}", gates=f"plan_adam_eps1e17_16k_{mdl}", priority=2,
          model=mdl, notes="Adjust grad_accum to fit; record what was used.")

# ---------------------------------------------------------------------------------
# 4. Warm start (adam, 16k). warmup here is ABSOLUTE steps (>=1 per LRScheduleConfig).
# ---------------------------------------------------------------------------------
for w in [100, 200]:
    sweep(f"tune_adamw_16k_warmup{w}", gates=f"plan_adam_eps1e17_16k_warmup{w}",
          priority=2, warmup=w)
sweep("tune_adamw_16k_warmup500", gates="plan_adam_eps1e17_16k_warmup500",
      status="blocked", priority=2, warmup=500,
      notes="500 warmup steps exceed the 125-step run; decide extend-epochs vs all-warmup "
            "before running (CONTROLS: warmup units).")

# ---------------------------------------------------------------------------------
# 5. Logit scale (adam, 16k) — landscape-changing, so it gets a sweep.
# ---------------------------------------------------------------------------------
for s in [0.5, 0.25]:
    sweep(f"tune_adamw_16k_scale{s}", gates=f"plan_adam_eps1e17_16k_scale{s}",
          priority=2, logit_scale=s)

# ---------------------------------------------------------------------------------
# 6. Weight decay / clipping (adam, 16k). Likely lr-neutral (rep-era nulls), so
#    priority 3 — but registered so the claim "tuned" holds everywhere.
# ---------------------------------------------------------------------------------
for wd in [0.0, 0.1]:
    sweep(f"tune_adamw_16k_wd{wd}", gates=f"plan_adam_eps1e17_16k_wd{wd}", priority=3,
          weight_decay=wd)
sweep("tune_adamw_16k_clip1.0", gates="plan_adam_eps1e17_16k_clip1.0", priority=3,
      max_grad_norm=1.0)

# ---------------------------------------------------------------------------------
# 7. Architecture mods — blocked on the gpt2_custom implementation. The no-mod
#    custom control needs its own sweep too (it is not stock gpt2).
# ---------------------------------------------------------------------------------
for mod in ["none", "qk_norm", "preact_layernorm", "preact_batchnorm"]:
    tag = "arch_control" if mod == "none" else mod
    sweep(f"tune_adamw_16k_{tag}", gates=f"plan_adam_eps1e17_16k_{tag}",
          status="blocked", priority=3, model="gpt2_custom", arch_mod=mod,
          notes="Blocked: gpt2_custom model does not exist yet.")


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning.csv")
    for r in rows:
        for c in COLUMNS:
            r.setdefault(c, "")
    order = {"measured": 0, "empty": 1, "blocked": 2}
    rows.sort(key=lambda r: (r["priority"], order.get(r["status"], 3), r["sweep_group"], r["lr"]))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    ids = [r["run_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate run_id"
    n = {s: sum(1 for r in rows if r["status"] == s) for s in ("measured", "empty", "blocked")}
    groups = len({r["sweep_group"] for r in rows})
    print(f"wrote {out}: {len(rows)} rows in {groups} sweep groups "
          f"({n['measured']} measured, {n['empty']} empty, {n['blocked']} blocked)")


if __name__ == "__main__":
    main()
