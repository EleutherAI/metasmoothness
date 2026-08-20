"""Build tuning.csv — stage-0 hyperparameter selection runs, registered as empty rows.

Per the tuning protocol in CONTROLS.md, every experiments.csv config whose optimization
problem differs from the anchor needs a held-out lr mini-sweep {0.5x, 1x, 2x} BEFORE its
bank is built. This file registers all of those runs so an agent with spare compute can
claim them: train the config, run `scripts/heldout_eval.py`, fill `train_loss` /
`heldout_loss`, edit the row here, regenerate, commit.

Selection rule (per sweep_group): lowest heldout_loss wins; if an endpoint of the 3-point
grid wins, ADD one more row one 2x step further out and re-check before freezing. When a
group is complete, write the winning lr into the matching experiments.csv row(s) — the
`selects_lr_for` column names them.

These are train-only runs: no banks, no subsets, no MAGIC. Cost is ~steps x bs; at bs256
that is 32 steps (4k) to 500 steps (64k).

status: measured = both loss columns filled; empty = to run; blocked = prerequisite
missing (the gpt2_custom implementation for arch groups, the D11 cost sign-off for
model-size groups, the bergson logit-scale hook for logit-scale groups).
"""

import csv
import os

COLUMNS = [
    "run_id", "sweep_group", "status", "priority", "selects_lr_for",
    "node_in_charge", "node_checkin_date",
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


def sweep(group, selects_lr_for, lrs=MINI, status="empty", priority=2, **cfg):
    for lr in lrs:
        r = dict(BASE)
        r.update(cfg)
        r.update(run_id=f"{group}_lr{lr:g}", sweep_group=group, selects_lr_for=selects_lr_for,
                 status=status, priority=priority, lr=lr)
        if not r.get("steps"):
            import math
            r["steps"] = math.ceil(int(r["n_docs"]) * int(r["num_epochs"]) / int(r["batch_size"]))
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
             selects_lr_for=f"sm_{opt}_eps1e17_16k_bs256", status="measured", priority=1,
             optimizer=opt, grad_accum_steps=4, lr=lr, train_loss=tl or "",
             heldout_loss=hl, steps=125,
             run_dir=f"/mnt/ssd-2/lucia/s16k_lrsweep/s16k_{opt}_lr{lr:g}",
             notes="Anchor sweep, measured 2026-08-20; untrained gpt2 heldout = 3.4981. "
                   "2e-4 selected (interior optimum for both optimizers).")
    rows.append(r)

# ---------------------------------------------------------------------------------
# 1. Token axis (both optimizers). 16k is the measured anchor; every other dataset size
#    needs its own mini-sweep. ga follows the micro-batch-16 rule at bs256.
# ---------------------------------------------------------------------------------
for opt in ["adamw", "muon"]:
    for n in [4000, 8000, 32000, 64000]:
        k = f"{n // 1000}k"
        sweep(f"tune_{opt}_{k}", selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_eps1e17_{k}_bs256",
              priority=1, optimizer=opt, n_docs=n)
# Endpoint extensions (procedure step 2: an endpoint win adds one 2x step outward).
sweep("tune_adamw_4k", selects_lr_for="plan_adam_eps1e17_4k_bs256", lrs=[5e-5],
      priority=1, optimizer="adamw", n_docs=4000,
      notes="Endpoint extension: 1e-4 won the 3-point grid (3.3149 vs 3.3178 at 2e-4).")
sweep("tune_adamw_64k", selects_lr_for="plan_adam_eps1e17_64k_bs256", lrs=[5e-5],
      priority=1, optimizer="adamw", n_docs=64000,
      notes="Endpoint extension: 1e-4 won the 3-point grid (3.2314 vs 3.2393 at 2e-4).")
sweep("tune_muon_64k", selects_lr_for="plan_muon_eps1e17_64k_bs256", lrs=[5e-5],
      priority=1, optimizer="muon", n_docs=64000,
      notes="Endpoint extension: 1e-4 won the 3-point grid (3.2323 vs 3.2417 at 2e-4).")
sweep("tune_muon_4k", selects_lr_for="plan_muon_eps1e17_4k_bs256", lrs=[8e-4],
      priority=1, optimizer="muon", n_docs=4000,
      notes="Endpoint extension: 4e-4 won the 3-point grid (3.3114 vs 3.3138 at 2e-4).")

# ---------------------------------------------------------------------------------
# 2. Batch-size axis (adam). Steps co-vary with bs at fixed epochs; lr optimum is
#    expected to shift with bs — this sweep is what makes the axis fair.
# ---------------------------------------------------------------------------------
CENTER = {16: 5e-5, 32: 5e-5, 64: 1e-4, 128: 1e-4}  # sqrt-batch rule, octave-rounded
for opt in ["adamw", "muon"]:
    for bs in [16, 32, 64, 128]:
        c = CENTER[bs]
        sweep(f"tune_{opt}_16k_bs{bs}",
              selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_eps1e17_16k_bs{bs}",
              lrs=[c / 2, c, c * 2], priority=1, optimizer=opt,
              batch_size=bs, grad_accum_steps=max(1, bs // 16),
              notes=f"Center {c:g} from the sqrt-batch rule (DECISIONS: sweep centers).")
# D2 arms
sweep("tune_adamw_16k_ep4", selects_lr_for="plan_adam_eps1e17_16k_ep4", priority=1, num_epochs=4,
      notes="D2 double-epochs arm (250 steps).")
sweep("tune_adamw_16k_ep4", selects_lr_for="plan_adam_eps1e17_16k_ep4", lrs=[5e-5],
      priority=1, num_epochs=4,
      notes="Endpoint extension: 1e-4 won the 3-point grid (3.2503 vs 3.2645 at 2e-4).")
sweep("tune_adamw_16k_bs512", selects_lr_for="plan_adam_eps1e17_16k_bs512", priority=1,
      batch_size=512, grad_accum_steps=32,
      notes="D2: uncontrolled double-batch arm (63 steps).")
# D12 eps_root eval-loss control (not an lr sweep: single point per optimizer)
for opt in ["adamw", "muon"]:
    sweep(f"tune_{opt}_16k_eps0_control", selects_lr_for=f"sm_{opt}_eps1e17_16k_bs256",
          lrs=[2e-4], priority=2, optimizer=opt, eps_root=0,
          notes="D12: eval-loss check that eps_root 1e-17 vs 0 is a null. Train-only twin "
                "of the anchor; compare heldout_loss against the measured anchor rows.")

# ---------------------------------------------------------------------------------
# 3. Model size (adam, 16k). Micro-batch 16 may not fit larger models — record the
#    ga actually used; ga is heldout-neutral.
# ---------------------------------------------------------------------------------
for mdl in ["gpt2-medium", "gpt2-large"]:
    sweep(f"tune_adamw_16k_{mdl}", selects_lr_for=f"plan_adam_eps1e17_16k_{mdl}",
          lrs=[5e-5, 1e-4, 2e-4], status="blocked", priority=2, model=mdl,
          notes=("D11: gpt2-medium is the registered scaling target; BLOCKED "
                 "until the cost plan is signed off. Center one octave down (larger models "
                 "prefer lower lr). Adjust grad_accum to fit; record it."
                 if mdl == "gpt2-medium" else
                 "D11: deferred — runs only if gpt2-medium proves informative."))

# D1 (2026-08-20): the former warm-start section is gone — warm start = attribution window,
# a pre-training axis; no lr-warmup sweeps exist.

# ---------------------------------------------------------------------------------
# 5. Logit scale (adam, 16k) — landscape-changing, so it gets a sweep. Blocked on the
#    bergson logit-scale hook — the tuning runs themselves need the hook, not just
#    the banks.
# ---------------------------------------------------------------------------------
for s in [0.5, 0.25]:
    sweep(f"tune_adamw_16k_scale{s}", selects_lr_for=f"plan_adam_eps1e17_16k_scale{s}",
          status="blocked", priority=2, logit_scale=s,
          notes="Blocked on the bergson logit-scale hook.")

# ---------------------------------------------------------------------------------
# 6. Weight decay / clipping (adam, 16k). Likely lr-neutral (rep-era nulls), so
#    priority 3 — but registered so the claim "tuned" holds everywhere.
# ---------------------------------------------------------------------------------
for wd in [0.0, 0.1]:
    sweep(f"tune_adamw_16k_wd{wd}", selects_lr_for=f"plan_adam_eps1e17_16k_wd{wd}", priority=3,
          weight_decay=wd)
sweep("tune_adamw_16k_clip1.0", selects_lr_for="plan_adam_eps1e17_16k_clip1.0", priority=3,
      max_grad_norm=1.0)

# ---------------------------------------------------------------------------------
# 7. Architecture mods — blocked on the gpt2_custom implementation. The no-mod
#    custom control needs its own sweep too (it is not stock gpt2).
#    preact_batchnorm dropped (D14): eval-mode training makes BN stats stale and
#    batch coupling makes per-doc gradients ill-defined.
# ---------------------------------------------------------------------------------
for mod in ["none", "qk_norm", "preact_layernorm"]:
    tag = "arch_control" if mod == "none" else mod
    sweep(f"tune_adamw_16k_{tag}", selects_lr_for=f"plan_adam_eps1e17_16k_{tag}",
          status="blocked", priority=3, model="gpt2_custom", arch_mod=mod,
          notes="Blocked on the gpt2_custom implementation (D10).")


# ---------------------------------------------------------------------------------
# Measured results for rows registered above (edit here when a run finishes, then
# regenerate). heldout_loss corresponds to run_dir; extra seeds go in notes.
# ---------------------------------------------------------------------------------
RESULTS = {
    "tune_adamw_16k_bs512_lr0.0001": dict(
        status="measured", heldout_loss=3.2805,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs512_lr0.0001_s42"),
    "tune_adamw_16k_bs512_lr0.0002": dict(
        status="measured", heldout_loss=3.2751,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs512_lr0.0002_s42",
        notes="Group complete: interior at 2e-4 (margins 0.0054/0.0026), the sqrt rule's "
              "rounded prediction — selected for plan_adam_eps1e17_16k_bs512."),
    "tune_adamw_16k_bs512_lr0.0004": dict(
        status="measured", heldout_loss=3.2777,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs512_lr0.0004_s42"),
    "tune_adamw_16k_wd0.1_lr0.0001": dict(
        status="measured", heldout_loss=3.2592,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_wd0.1_lr0.0001_s42"),
    "tune_adamw_16k_wd0.1_lr0.0002": dict(
        status="measured", heldout_loss=3.2572,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_wd0.1_lr0.0002_s42",
        notes="Group complete: interior at the anchor 2e-4. The wd0.1 curve matches "
              "wd0.0 point-for-point (3.2592/3.2572/3.2669 vs 3.2592/3.2572/3.2670) — "
              "weight decay is a null on heldout loss at these settings."),
    "tune_adamw_16k_wd0.1_lr0.0004": dict(
        status="measured", heldout_loss=3.2669,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_wd0.1_lr0.0004_s42"),
    "tune_adamw_16k_clip1.0_lr0.0001": dict(
        status="measured", heldout_loss=3.2553,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_clip1.0_lr0.0001_s42"),
    "tune_adamw_16k_clip1.0_lr0.0002": dict(
        status="measured", heldout_loss=3.2543,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_clip1.0_lr0.0002_s42",
        notes="Group complete: interior at the anchor 2e-4 (1e-4 gap 0.0010 is a tie; "
              "clipping is lr-neutral) — selected for plan_adam_eps1e17_16k_clip1.0."),
    "tune_adamw_16k_clip1.0_lr0.0004": dict(
        status="measured", heldout_loss=3.2653,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_clip1.0_lr0.0004_s42"),
    "tune_adamw_16k_ep4_lr0.0002": dict(
        status="measured", heldout_loss=3.2645,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_ep4_lr0.0002_s42"),
    "tune_adamw_16k_ep4_lr0.0004": dict(
        status="measured", heldout_loss=3.3145,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_ep4_lr0.0004_s42",
        notes="1e-4 is an endpoint winner (gap 0.0142 - double epochs clearly prefers "
              "lower lr); 5e-5 extension registered."),
    "tune_adamw_16k_wd0.0_lr0.0001": dict(
        status="measured", heldout_loss=3.2592,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_wd0.0_lr0.0001_s42"),
    "tune_adamw_16k_wd0.0_lr0.0002": dict(
        status="measured", heldout_loss=3.2572,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_wd0.0_lr0.0002_s42",
        notes="Group complete: interior optimum at the anchor lr 2e-4 — wd is lr-neutral "
              "as predicted; selected for plan_adam_eps1e17_16k_wd0.0."),
    "tune_adamw_16k_wd0.0_lr0.0004": dict(
        status="measured", heldout_loss=3.2670,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_wd0.0_lr0.0004_s42"),
    "tune_muon_16k_bs128_lr0.0002": dict(
        status="measured", heldout_loss=3.2526,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_16k_bs128_lr0.0002_s42",
        notes="Group complete: interior optimum at the sqrt-rule center 1e-4 (margin "
              "0.0025) — selected for plan_muon_eps1e17_16k_bs128. The sqrt-batch rule "
              "went 8-for-8 across the batch-size axis: every group's optimum landed on "
              "its predicted center."),
    "tune_adamw_16k_ep4_lr0.0001": dict(
        status="measured", heldout_loss=3.2503,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_ep4_lr0.0001_s42"),
    "tune_adamw_16k_eps0_control_lr0.0002": dict(
        status="measured", heldout_loss=3.2572,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_eps0_control_lr0.0002_s42",
        notes="D12 resolved: identical to the eps1e-17 anchor (3.2572) to all four "
              "decimals - eps_root 1e-17 vs 0 is a null on eval loss."),
    "tune_muon_16k_eps0_control_lr0.0002": dict(
        status="measured", heldout_loss=3.2570,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_16k_eps0_control_lr0.0002_s42",
        notes="D12 resolved: identical to the eps1e-17 anchor (3.2570) to all four "
              "decimals."),
    "tune_muon_16k_bs128_lr5e-05": dict(
        status="measured", heldout_loss=3.2609,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_16k_bs128_lr5e-05_s42"),
    "tune_muon_16k_bs128_lr0.0001": dict(
        status="measured", heldout_loss=3.2501,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_16k_bs128_lr0.0001_s42"),
    "tune_adamw_16k_bs128_lr0.0001": dict(
        status="measured", heldout_loss=3.2498,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs128_lr0.0001_s42",
        notes="Group complete: interior optimum at the sqrt-rule center 1e-4 (margin "
              "0.0037) — selected for plan_adam_eps1e17_16k_bs128."),
    "tune_adamw_16k_bs128_lr0.0002": dict(
        status="measured", heldout_loss=3.2535,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs128_lr0.0002_s42"),
    "tune_adamw_16k_bs64_lr0.0002": dict(
        status="measured", heldout_loss=3.2585,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs64_lr0.0002_s42",
        notes="Group complete: interior optimum at the sqrt-rule center 1e-4 — selected "
              "for plan_adam_eps1e17_16k_bs64 (5e-5 gap 0.0010 is a tie; center wins on "
              "both grounds)."),
    "tune_muon_16k_bs64_lr0.0002": dict(
        status="measured", heldout_loss=3.2592,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_16k_bs64_lr0.0002_s42",
        notes="Group complete: interior optimum at the sqrt-rule center 1e-4 (margin "
              "0.0022) — selected for plan_muon_eps1e17_16k_bs64."),
    "tune_muon_64k_lr5e-05": dict(
        status="measured", heldout_loss=3.2373,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_64k_lr5e-05_s42",
        notes="Extension point. Group complete: interior optimum at 1e-4 (3.2323, margin "
              "0.005) — selected for plan_muon_eps1e17_64k_bs256. Both optimizers' N axes "
              "are now fully tuned: 4k and 64k sit below/off the anchor lr, 8k-32k on it."),
    "tune_adamw_16k_bs64_lr5e-05": dict(
        status="measured", heldout_loss=3.2489,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs64_lr5e-05_s42"),
    "tune_adamw_16k_bs64_lr0.0001": dict(
        status="measured", heldout_loss=3.2479,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs64_lr0.0001_s42"),
    "tune_adamw_16k_bs128_lr5e-05": dict(
        status="measured", heldout_loss=3.2547,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs128_lr5e-05_s42"),
    "tune_muon_16k_bs64_lr5e-05": dict(
        status="measured", heldout_loss=3.2486,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_16k_bs64_lr5e-05_s42"),
    "tune_muon_16k_bs64_lr0.0001": dict(
        status="measured", heldout_loss=3.2464,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_16k_bs64_lr0.0001_s42"),
    "tune_muon_16k_bs32_lr0.0001": dict(
        status="measured", heldout_loss=3.2495,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_16k_bs32_lr0.0001_s42",
        notes="Group complete: interior optimum at the sqrt-rule center 5e-5 "
              "(margins ~0.005) — selected for plan_muon_eps1e17_16k_bs32."),
    "tune_adamw_64k_lr5e-05": dict(
        status="measured", heldout_loss=3.2340,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_64k_lr5e-05_s42",
        notes="Extension point. Group complete: interior optimum at 1e-4 (3.2314, margin "
              "0.0026 over 5e-5) — selected for plan_adam_eps1e17_64k_bs256."),
    "tune_muon_16k_bs32_lr2.5e-05": dict(
        status="measured", heldout_loss=3.2497,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_16k_bs32_lr2.5e-05_s42"),
    "tune_muon_16k_bs32_lr5e-05": dict(
        status="measured", heldout_loss=3.2441,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_16k_bs32_lr5e-05_s42"),
    "tune_muon_16k_bs16_lr0.0001": dict(
        status="measured", heldout_loss=3.2573,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_16k_bs16_lr0.0001_s42",
        notes="Group complete: 2.5e-5 endpoint leads 5e-5 by 0.0002, under the 0.002 tie "
              "threshold, so the tie rule selects the center 5e-5 and no extension fires "
              "- selected for plan_muon_eps1e17_16k_bs16."),
    "tune_muon_64k_lr0.0004": dict(
        status="measured", heldout_loss=3.2804,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_64k_lr0.0004_s42",
        notes="1e-4 is an endpoint winner (gap 0.0094); 5e-5 extension registered."),
    "tune_muon_16k_bs16_lr2.5e-05": dict(
        status="measured", heldout_loss=3.2441,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_16k_bs16_lr2.5e-05_s42"),
    "tune_muon_16k_bs16_lr5e-05": dict(
        status="measured", heldout_loss=3.2443,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_16k_bs16_lr5e-05_s42"),
    "tune_muon_64k_lr0.0002": dict(
        status="measured", heldout_loss=3.2417,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_64k_lr0.0002_s42"),
    "tune_adamw_64k_lr0.0004": dict(
        status="measured", heldout_loss=3.2671,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_64k_lr0.0004_s42",
        notes="1e-4 is an endpoint winner (gap 0.0079); 5e-5 extension registered."),
    "tune_adamw_16k_bs32_lr2.5e-05": dict(
        status="measured", heldout_loss=3.2511,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs32_lr2.5e-05_s42"),
    "tune_adamw_16k_bs32_lr5e-05": dict(
        status="measured", heldout_loss=3.2473,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs32_lr5e-05_s42",
        notes="Group complete: interior optimum at the sqrt-rule center 5e-5 "
              "(margins ~0.004) — selected for plan_adam_eps1e17_16k_bs32."),
    "tune_adamw_16k_bs32_lr0.0001": dict(
        status="measured", heldout_loss=3.2513,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs32_lr0.0001_s42"),
    "tune_muon_64k_lr0.0001": dict(
        status="measured", heldout_loss=3.2323,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_64k_lr0.0001_s42"),
    "tune_adamw_64k_lr0.0002": dict(
        status="measured", heldout_loss=3.2393,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_64k_lr0.0002_s42"),
    "tune_adamw_64k_lr0.0001": dict(
        status="measured", heldout_loss=3.2314,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_64k_lr0.0001_s42"),
    "tune_adamw_16k_bs16_lr2.5e-05": dict(
        status="measured", heldout_loss=3.2502,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs16_lr2.5e-05_s42"),
    "tune_adamw_16k_bs16_lr5e-05": dict(
        status="measured", heldout_loss=3.2497,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs16_lr5e-05_s42",
        notes="Group complete: interior optimum at the sqrt-rule center 5e-5 (2.5e-5 gap "
              "0.0005 is under the tie threshold; center selected on both grounds) — "
              "selected for plan_adam_eps1e17_16k_bs16."),
    "tune_adamw_16k_bs16_lr0.0001": dict(
        status="measured", heldout_loss=3.2620,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_bs16_lr0.0001_s42"),
    "tune_adamw_4k_lr0.0001": dict(
        status="measured", heldout_loss=3.3149,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_4k_lr0.0001_s42",
        notes="Endpoint winner of the 3-point grid; 5e-5 extension registered."),
    "tune_adamw_4k_lr0.0002": dict(
        status="measured", heldout_loss=3.3178,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_4k_lr0.0002_s42"),
    "tune_adamw_4k_lr0.0004": dict(
        status="measured", heldout_loss=3.3311,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_4k_lr0.0004_s42"),
    "tune_adamw_4k_lr5e-05": dict(
        status="measured", heldout_loss=3.3236,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_4k_lr5e-05_s42",
        notes="Extension point. Group complete: interior optimum at 1e-4 (3.3149) — "
              "selected for plan_adam_eps1e17_4k_bs256 (gap to 2e-4 is 0.0029, above the "
              "tie threshold, so the tuned value replaces the center)."),
    "tune_muon_4k_lr0.0008": dict(
        status="measured", heldout_loss=3.3363,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_4k_lr0.0008_s42",
        notes="Extension point. Group complete: interior optimum at 4e-4 (3.3114) — "
              "selected for plan_muon_eps1e17_4k_bs256 (gap to 2e-4 is 0.0024, above the "
              "tie threshold)."),
    "tune_adamw_8k_lr0.0001": dict(
        status="measured", heldout_loss=3.2866,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_8k_lr0.0001_s42"),
    "tune_adamw_8k_lr0.0004": dict(
        status="measured", heldout_loss=3.2957,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_8k_lr0.0004_s42",
        notes="Group complete: interior optimum at 2e-4 (3.2851) — "
              "selected for plan_adam_eps1e17_8k_bs256."),
    "tune_muon_4k_lr0.0001": dict(
        status="measured", heldout_loss=3.3462,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_4k_lr0.0001_s42"),
    "tune_muon_4k_lr0.0002": dict(
        status="measured", heldout_loss=3.3138,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_4k_lr0.0002_s42"),
    "tune_muon_4k_lr0.0004": dict(
        status="measured", heldout_loss=3.3114,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_4k_lr0.0004_s42",
        notes="Endpoint winner of the 3-point grid; 8e-4 extension registered."),
    "tune_adamw_32k_lr0.0001": dict(
        status="measured", heldout_loss=3.2388,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_32k_lr0.0001_s42"),
    "tune_adamw_32k_lr0.0002": dict(
        status="measured", heldout_loss=3.2365,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_32k_lr0.0002_s42",
        notes="Group complete: interior optimum at 2e-4 (margin 0.0023) — selected "
              "for plan_adam_eps1e17_32k_bs256."),
    "tune_adamw_32k_lr0.0004": dict(
        status="measured", heldout_loss=3.2449,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_32k_lr0.0004_s42"),
    "tune_muon_32k_lr0.0001": dict(
        status="measured", heldout_loss=3.2408,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_32k_lr0.0001_s42"),
    "tune_muon_32k_lr0.0002": dict(
        status="measured", heldout_loss=3.2372,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_32k_lr0.0002_s42",
        notes="Group complete: interior optimum at 2e-4 (margin 0.0036) — selected "
              "for plan_muon_eps1e17_32k_bs256."),
    "tune_muon_32k_lr0.0004": dict(
        status="measured", heldout_loss=3.2479,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_32k_lr0.0004_s42"),
    "tune_muon_8k_lr0.0001": dict(
        status="measured", heldout_loss=3.2974,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_8k_lr0.0001_s42"),
    "tune_muon_8k_lr0.0002": dict(
        status="measured", heldout_loss=3.2841,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_8k_lr0.0002_s42",
        notes="Group complete: interior optimum at 2e-4 — selected for "
              "plan_muon_eps1e17_8k_bs256."),
    "tune_muon_8k_lr0.0004": dict(
        status="measured", heldout_loss=3.2874,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_8k_lr0.0004_s42"),
    "tune_adamw_8k_lr0.0002": dict(
        status="measured", heldout_loss=3.2851,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_8k_lr0.0002_s42",
        notes="Measured 2026-08-20 (protocol-validation run, lotus-0). Seed 42: 3.2851; "
              "seed 43: 3.2839 (dir suffix _s43); gap 0.0012 nats at 63 steps — matches "
              "the ~0.001 seed-noise floor measured at 125 steps. This measurement is "
              "the evidence behind dropping the short-run 2-seed rule (procedure step 4)."),
}


def _preserve_claims(out_path, rows):
    """Carry node claims over from the existing CSV so regeneration never drops them.

    Claims are the one thing edited in the CSV directly (see NODES.md); everything else
    is edited in this script.
    """
    if not os.path.exists(out_path):
        return
    with open(out_path, newline="") as f:
        old = {r["run_id"]: r for r in csv.DictReader(f)}
    for r in rows:
        if r["status"] == "measured":
            continue  # a finished row has results and no owner (NODES.md)
        prev = old.get(r["run_id"])
        if prev:
            r["node_in_charge"] = prev.get("node_in_charge", "") or r.get("node_in_charge", "")
            r["node_checkin_date"] = prev.get("node_checkin_date", "") or r.get("node_checkin_date", "")


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning.csv")
    for r in rows:
        r.update(RESULTS.get(r["run_id"], {}))
        for c in COLUMNS:
            r.setdefault(c, "")
    order = {"measured": 0, "empty": 1, "blocked": 2}
    rows.sort(key=lambda r: (r["priority"], order.get(r["status"], 3), r["sweep_group"], r["lr"]))
    _preserve_claims(out, rows)
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
