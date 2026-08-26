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
# STEP-SCALING axis, ms-ONLY (2026-08-25). The question is how metasmoothness
# behaves as the number of OPTIMISER STEPS grows, so batch size is held at 32 --
# the config that gives 1000 steps and ms 0.9800 at 16k -- and the corpus grows:
#
#     bs32, 2 epochs:  32k -> 2000 steps, 64k -> 4000, 128k -> 8000
#
# No retrain bank: MAGIC is one reverse pass per query over the whole corpus, so
# these are 150h+ per row to score. ms needs no bank and costs three trainings.
#
# Centre is 5e-5, the CONTROLS centre for batch size 16-32 (sqrt(32/256) of the
# 2e-4 reference), NOT the 2e-4 used for the bs256 token axis.
# DRIFT RE-CENTRING (2026-08-25). The lr optimum halves as steps double:
#   1000 steps -> 5e-5, 2000 -> 5e-5 (interior), 4000 -> 2.5e-5 (low endpoint).
# So 8000 steps is predicted at ~1.25e-5 and a grid centred at 5e-5 would land on
# its low endpoint and need an extension run -- 10h at 128k, 20h at 256k. The
# 128k grid therefore drops one step to {1.25e-5, 2.5e-5, 5e-5}.
#
# Longer training preferring a lower lr is an established trend, not a reading
# off one lucky run, so centring on it is allowed (Lucia, 2026-08-25): it is the
# endpoint extension we already know is coming, applied up front. If 1.25e-5
# wins its own endpoint, extend as normal.
#
# MEASURED, AND IT WENT THE OTHER WAY (2026-08-25). Both 128k arms won at 5e-5,
# the HIGH endpoint, so the grid needed extending UP to 1e-4, not down:
#
#     adamw   1.25e-5 3.2219   2.5e-5 3.2150   5e-5 3.2129
#     muon    1.25e-5 3.2183   2.5e-5 3.2110   5e-5 3.2079
#
# So 8000 steps wants the same 5e-5 that 2000 steps did, and MORE than the 4000
# step arms took. The halving trend was three points -- 5e-5, 5e-5, 2.5e-5 --
# and the 64k step down is worth 0.003 nats against a 0.009 nat spread across
# the whole 128k grid. That is inside the noise this axis is known to have, and
# re-centring on it cost a run per arm rather than saving one.
#
# The lesson is not "never centre on a trend", it is that this axis is too flat
# to read a trend off adjacent rungs. Centre on the CONTROLS rule and extend
# from what the grid reports.
GRID = {32000: [2.5e-5, 5e-5, 1e-4],
        64000: [2.5e-5, 5e-5, 1e-4],
        128000: [1.25e-5, 2.5e-5, 5e-5]}
for opt in ["adamw", "muon"]:
    for n in [32000, 64000, 128000]:
        k = f"{n // 1000}k"
        sweep(f"tune_{opt}_{k}_bs32",
              selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_eps1e17_{k}_bs32",
              lrs=GRID[n], priority=2, optimizer=opt, n_docs=n,
              batch_size=32, grad_accum_steps=2,
              notes="ms-only step-scaling point at fixed bs32; no retrain bank planned.")

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
sweep("tune_adamw_16k_scale0.25", selects_lr_for="plan_adam_eps1e17_16k_scale0.25",
      lrs=[8e-4], priority=2, logit_scale=0.25,
      notes="Endpoint extension: 4e-4 won by 0.064; optimum above the grid.")
sweep("tune_adamw_16k_scale0.25", selects_lr_for="plan_adam_eps1e17_16k_scale0.25",
      lrs=[1.6e-3], priority=2, logit_scale=0.25,
      notes="Second endpoint extension (procedure limit): 8e-4 won by 0.0395.")
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
          lrs=[5e-5, 1e-4, 2e-4],
          status="" if mdl == "gpt2-medium" else "blocked",
          priority=2, model=mdl,
          notes=("D11: gpt2-medium is the registered scaling target. Cost plan "
                 "signed off 2026-08-23 (see DECISIONS D11), so this group is now "
                 "claimable. Center one octave down (larger models prefer lower lr). "
                 "Adjust grad_accum to fit; record it."
                 if mdl == "gpt2-medium" else
                 "D11: deferred — runs only if gpt2-medium proves informative."))

# Measured 2026-08-23 on the borrowed A100 pods (marisa-0 / maria-1 / shivam2-0),
# nproc 2, pinned venv, 125 steps. Held-out CE on the fixed 4k set.
#
# The interior point wins, so the optimum is bracketed and the sweep does not need
# extending. But the whole 4x range spans 0.0066 nats -- gpt2-medium is far flatter
# in lr than gpt2-small was (the anchor sweep moved 0.14 nats over the same span),
# so treat 1e-4 as "no worse than its neighbours" rather than a sharp optimum.
GPT2_MEDIUM_HELDOUT = {5e-5: 3.0062, 1e-4: 3.0019, 2e-4: 3.0085}
for r in rows:
    if r["sweep_group"] == "tune_adamw_16k_gpt2-medium":
        r["heldout_loss"] = GPT2_MEDIUM_HELDOUT[r["lr"]]
        r["status"] = "measured"
        r["run_dir"] = f"/mnt/ssd-2/lucia/paper_runs/tuning/{r['run_id']}_s42"

# D1 (2026-08-20): the former warm-start section is gone — warm start = attribution window,
# a pre-training axis; no lr-warmup sweeps exist.

# ---------------------------------------------------------------------------------
# 5. Logit scale (adam, 16k) — landscape-changing, so it gets a sweep. Blocked on the
#    bergson logit-scale hook — the tuning runs themselves need the hook, not just
#    the banks.
# ---------------------------------------------------------------------------------
for s in [0.5, 0.25]:
    sweep(f"tune_adamw_16k_scale{s}", selects_lr_for=f"plan_adam_eps1e17_16k_scale{s}",
          status="empty", priority=2, logit_scale=s,
          notes="Hook exists (bergson feat/logit-scale, PR #433); run from that "
                "worktree until merge.")

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
# The whole architecture axis is cut per D16, qk_norm included. These rows stay
# registered as future work, but "cut" rather than "blocked": nothing is pending
# that would open them. preact_layernorm is the same fine-tune-graft design the
# ruling rejects, and arch_control exists only to control the arch_mod rows.
for mod in ["none", "preact_layernorm"]:
    tag = "arch_control" if mod == "none" else mod
    sweep(f"tune_adamw_16k_{tag}", selects_lr_for=f"plan_adam_eps1e17_16k_{tag}",
          status="cut", priority=3, model="gpt2_custom", arch_mod=mod,
          notes="Cut with the architecture axis (D16, applied 2026-08-24). "
                "Future work: the native question needs the modification "
                "pre-trained in, not grafted into a pretrained model.")


# ---------------------------------------------------------------------------------
# Measured results for rows registered above (edit here when a run finishes, then
# regenerate). heldout_loss corresponds to run_dir; extra seeds go in notes.
# ---------------------------------------------------------------------------------
RESULTS = {
    "tune_adamw_256k_bs32_lr6.25e-06": dict(status="measured", heldout_loss=3.2185,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_256k_bs32_lr6.25e-06_s42",
        notes="Step-scaling sweep at fixed bs32 (16000 steps), nproc 2, A100, maria-1. Group INCOMPLETE: the 2.5e-05 point died at rc=124 under the old flat 2h slot deadline and has not been re-measured, so the winner so far (1.25e-05) is still an ENDPOINT of the three-point sweep and the selection rule says extend rather than select."),
    "tune_adamw_256k_bs32_lr1.25e-05": dict(status="measured", heldout_loss=3.2087,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_256k_bs32_lr1.25e-05_s42",
        notes="Step-scaling sweep at fixed bs32 (16000 steps), nproc 2, A100, maria-1. Group INCOMPLETE: the 2.5e-05 point died at rc=124 under the old flat 2h slot deadline and has not been re-measured, so the winner so far (1.25e-05) is still an ENDPOINT of the three-point sweep and the selection rule says extend rather than select."),
    "tune_muon_256k_bs32_lr6.25e-06": dict(status="measured", heldout_loss=3.215,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_256k_bs32_lr6.25e-06_s42",
        notes="Step-scaling sweep at fixed bs32 (16000 steps), nproc 2, A100, maria-1. Group INCOMPLETE: the 2.5e-05 point died at rc=124 under the old flat 2h slot deadline and has not been re-measured, so the winner so far (1.25e-05) is still an ENDPOINT of the three-point sweep and the selection rule says extend rather than select."),
    "tune_muon_256k_bs32_lr1.25e-05": dict(status="measured", heldout_loss=3.2039,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_256k_bs32_lr1.25e-05_s42",
        notes="Step-scaling sweep at fixed bs32 (16000 steps), nproc 2, A100, maria-1. Group INCOMPLETE: the 2.5e-05 point died at rc=124 under the old flat 2h slot deadline and has not been re-measured, so the winner so far (1.25e-05) is still an ENDPOINT of the three-point sweep and the selection rule says extend rather than select."),
    "tune_adamw_64k_bs32_lr1.25e-05": dict(status="measured", heldout_loss=3.2365,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_64k_bs32_lr1.25e-05_s42",
        notes="Step-scaling sweep at fixed bs32 (4000 steps), nproc 1 ga 2, A100, lotus-0. GROUP COMPLETE, clean interior minimum: 1.25e-5 3.2365 / 2.5e-5 3.2322 / 5e-5 3.2355 / 1e-4 3.2552. 2.5e-5 selected for plan_adam_eps1e17_64k_bs32; both neighbours are higher, so no endpoint extension is needed."),
    "tune_adamw_64k_bs32_lr2.5e-05": dict(status="measured", heldout_loss=3.2322,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_64k_bs32_lr2.5e-05_s42",
        notes="Step-scaling sweep at fixed bs32 (4000 steps), nproc 1 ga 2, A100, lotus-0. GROUP COMPLETE, clean interior minimum: 1.25e-5 3.2365 / 2.5e-5 3.2322 / 5e-5 3.2355 / 1e-4 3.2552. 2.5e-5 selected for plan_adam_eps1e17_64k_bs32; both neighbours are higher, so no endpoint extension is needed."),
    "tune_adamw_64k_bs32_lr5e-05": dict(status="measured", heldout_loss=3.2355,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_64k_bs32_lr5e-05_s42",
        notes="Step-scaling sweep at fixed bs32 (4000 steps), nproc 1 ga 2, A100, lotus-0. GROUP COMPLETE, clean interior minimum: 1.25e-5 3.2365 / 2.5e-5 3.2322 / 5e-5 3.2355 / 1e-4 3.2552. 2.5e-5 selected for plan_adam_eps1e17_64k_bs32; both neighbours are higher, so no endpoint extension is needed."),
    "tune_adamw_64k_bs32_lr0.0001": dict(status="measured", heldout_loss=3.2552,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_64k_bs32_lr0.0001_s42",
        notes="Step-scaling sweep at fixed bs32 (4000 steps), nproc 1 ga 2, A100, lotus-0. GROUP COMPLETE, clean interior minimum: 1.25e-5 3.2365 / 2.5e-5 3.2322 / 5e-5 3.2355 / 1e-4 3.2552. 2.5e-5 selected for plan_adam_eps1e17_64k_bs32; both neighbours are higher, so no endpoint extension is needed."),
    "tune_adamw_256k_bs32_lr6.25e-06": dict(status="measured", heldout_loss=3.2185,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_256k_bs32_lr6.25e-06_s42",
        notes="256k bs32 step-scaling sweep, measured on maria-1 (2-GPU slots), recorded here from the shared slot logs after sitting unrecorded ~7h; lotus-0 did not run it. The slot log shows earlier rc=137 and rc=124 attempts for this row BEFORE the successful one -- the recorded number is the final, completed attempt, whose run_dir holds both config.yaml and model. GROUP INCOMPLETE: the 2.5e-05 point is claimed by marisa-0 and still pending, so no lr is selected yet."),
    "tune_adamw_256k_bs32_lr1.25e-05": dict(status="measured", heldout_loss=3.2087,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_256k_bs32_lr1.25e-05_s42",
        notes="256k bs32 step-scaling sweep, measured on maria-1 (2-GPU slots), recorded here from the shared slot logs after sitting unrecorded ~7h; lotus-0 did not run it. The slot log shows earlier rc=137 and rc=124 attempts for this row BEFORE the successful one -- the recorded number is the final, completed attempt, whose run_dir holds both config.yaml and model. GROUP INCOMPLETE: the 2.5e-05 point is claimed by marisa-0 and still pending, so no lr is selected yet."),
    "tune_muon_256k_bs32_lr6.25e-06": dict(status="measured", heldout_loss=3.215,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_256k_bs32_lr6.25e-06_s42",
        notes="256k bs32 step-scaling sweep, measured on maria-1 (2-GPU slots), recorded here from the shared slot logs after sitting unrecorded ~7h; lotus-0 did not run it. The slot log shows earlier rc=137 and rc=124 attempts for this row BEFORE the successful one -- the recorded number is the final, completed attempt, whose run_dir holds both config.yaml and model. GROUP INCOMPLETE: the 2.5e-05 point is claimed by marisa-0 and still pending, so no lr is selected yet."),
    "tune_muon_256k_bs32_lr1.25e-05": dict(status="measured", heldout_loss=3.2039,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_256k_bs32_lr1.25e-05_s42",
        notes="256k bs32 step-scaling sweep, measured on maria-1 (2-GPU slots), recorded here from the shared slot logs after sitting unrecorded ~7h; lotus-0 did not run it. The slot log shows earlier rc=137 and rc=124 attempts for this row BEFORE the successful one -- the recorded number is the final, completed attempt, whose run_dir holds both config.yaml and model. GROUP INCOMPLETE: the 2.5e-05 point is claimed by marisa-0 and still pending, so no lr is selected yet."),
    "tune_muon_64k_bs32_lr0.0001": dict(status="measured", heldout_loss=3.2579,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_64k_bs32_lr0.0001_s42",
        notes="Step-scaling sweep at fixed bs32 (4000 steps), nproc 1 ga 2, A100, lotus-0. COMPLETES THE MUON 64k GROUP: 1.25e-5 3.2337 / 2.5e-5 3.2290 / 5e-5 3.2333 / 1e-4 3.2579. Confirmatory as expected -- 1e-4 is the worst of the four and the winner 2.5e-5 was already bracketed on both sides, so no endpoint extension is needed. Measured on the pinned node-local env (python 3.11.16 / torch 2.13.0+cu126 / nccl 2.29.3, leak check clean) built on lotus-0 today; earlier lotus-0 launches that ran outside it were killed before recording anything."),
    "tune_muon_64k_bs32_lr1.25e-05": dict(status="measured", heldout_loss=3.2337,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_64k_bs32_lr1.25e-05_s42",
        notes="Step-scaling sweep at fixed bs32 (4000 steps), nproc 1 ga 2, A100, lotus-0. Interior minimum bracketed: 1.25e-5 3.2337 / 2.5e-5 3.2290 / 5e-5 3.2333. 2.5e-5 selected for plan_muon_eps1e17_64k_bs32; the pending 1e-4 point is confirmatory only, since both neighbours of the winner are already higher."),
    "tune_muon_64k_bs32_lr5e-05": dict(status="measured", heldout_loss=3.2333,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_64k_bs32_lr5e-05_s42",
        notes="Step-scaling sweep at fixed bs32 (4000 steps), nproc 1 ga 2, A100, lotus-0. Interior minimum bracketed: 1.25e-5 3.2337 / 2.5e-5 3.2290 / 5e-5 3.2333. 2.5e-5 selected for plan_muon_eps1e17_64k_bs32; the pending 1e-4 point is confirmatory only, since both neighbours of the winner are already higher."),
    "tune_muon_64k_bs32_lr2.5e-05": dict(status="measured", heldout_loss=3.229,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_muon_64k_bs32_lr2.5e-05_s42",
        notes="Step-scaling sweep at fixed bs32 (4000 steps), nproc 1 ga 2, A100, lotus-0. Interior minimum bracketed: 1.25e-5 3.2337 / 2.5e-5 3.2290 / 5e-5 3.2333. 2.5e-5 selected for plan_muon_eps1e17_64k_bs32; the pending 1e-4 point is confirmatory only, since both neighbours of the winner are already higher."),
    "tune_adamw_16k_scale0.25_lr0.0016": dict(status="measured", heldout_loss=3.4341,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_scale0.25_lr0.0016_s42",
        notes="Group complete: 3.4341 vs 8e-4's 3.4338 is a 0.0003 tie - the curve "
              "flattened; 8e-4 selected (best and nearer center) for "
              "plan_adam_eps1e17_16k_scale0.25. Full curve 1e-4..1.6e-3: "
              "3.6272/3.5374/3.4733/3.4338/3.4341 - the tuned optimum sits 4x above the "
              "anchor lr; strong logit scaling flattens head gradients."),
    "tune_adamw_16k_scale0.25_lr0.0008": dict(status="measured", heldout_loss=3.4338,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_scale0.25_lr0.0008_s42",
        notes="Second endpoint win (0.0395 over 4e-4); 1.6e-3 extension is the LAST allowed "
              "before the procedure mandates investigation (optimum >4x from center). Now "
              "clears untrained by 0.064."),
    "tune_adamw_16k_scale0.5_lr0.0001": dict(status="measured", heldout_loss=3.3168,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_scale0.5_lr0.0001_s42"),
    "tune_adamw_16k_scale0.5_lr0.0002": dict(status="measured", heldout_loss=3.3020,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_scale0.5_lr0.0002_s42",
        notes="Group complete: 4e-4 leads by 0.0008 (tie); center 2e-4 selected. Heldout "
              "evaluated WITH --logit-scale 0.5 (raw logits are miscalibrated by design)."),
    "tune_adamw_16k_scale0.5_lr0.0004": dict(status="measured", heldout_loss=3.3012,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_scale0.5_lr0.0004_s42"),
    "tune_adamw_16k_scale0.25_lr0.0001": dict(status="measured", heldout_loss=3.6272,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_scale0.25_lr0.0001_s42"),
    "tune_adamw_16k_scale0.25_lr0.0002": dict(status="measured", heldout_loss=3.5374,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_scale0.25_lr0.0002_s42"),
    "tune_adamw_16k_scale0.25_lr0.0004": dict(status="measured", heldout_loss=3.4733,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_scale0.25_lr0.0004_s42",
        notes="ENDPOINT win by 0.064 - optimum above the grid; 8e-4 extension registered. "
              "Note: only this point beats untrained (3.4981), and barely - strong scaling "
              "flattens head gradients, shifting the effective lr far up. CONTROLS rule 4 "
              "may bite this row."),
    "tune_adamw_16k_ep4_lr5e-05": dict(
        status="measured", heldout_loss=3.2521,
        run_dir="/mnt/ssd-2/lucia/paper_runs/tuning/tune_adamw_16k_ep4_lr5e-05_s42",
        notes="Extension point. Group complete: 1e-4 (3.2503) selected for "
              "plan_adam_eps1e17_16k_ep4. The 5e-5 gap (0.0018) is under the tie "
              "threshold, but the center 2e-4 is decisively excluded (0.0142 worse), so "
              "the tie resolves to the tied value nearest the anchor. THE RUNNABLE "
              "TUNING GRID IS COMPLETE: all 80 non-blocked rows measured."),
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


# Next rung of the step-scaling ladder: 256k at bs32 = 16000 steps. Registered
# now so the sweep can start the moment 128k reports; the goal is to keep
# climbing until ms crosses the 0.95 collapse boundary for one optimizer.
# Centre drops to 2.5e-5, following the measured endpoint extensions rather than
# the batch-size heuristic: 32k chose 5e-5 and both 64k arms won at 2.5e-5, so a
# grid centred at 5e-5 would almost certainly land on its low endpoint again.
for opt in ["adamw", "muon"]:
    sweep(f"tune_{opt}_512k_bs32",
          selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_eps1e17_512k_bs32",
          lrs=[3.125e-6, 6.25e-6, 1.25e-5], priority=3, optimizer=opt, n_docs=512000,
          batch_size=32, grad_accum_steps=2,
          notes="ms-only step-scaling point at fixed bs32 (32000 steps); no bank planned.")

for opt in ["adamw", "muon"]:
    sweep(f"tune_{opt}_256k_bs32",
          selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_eps1e17_256k_bs32",
          lrs=[6.25e-6, 1.25e-5, 2.5e-5], priority=3, optimizer=opt, n_docs=256000,
          batch_size=32, grad_accum_steps=2,
          notes="ms-only step-scaling point at fixed bs32 (16000 steps); no bank planned.")

# Endpoint extensions on the step-scaling axis (procedure step 2). Both 64k arms
# won at the LOW endpoint 2.5e-5, so each gets one 2x step outward. The optimum
# is drifting down as steps grow -- 32k chose 5e-5, 64k wants 2.5e-5 or less --
# which is the same downward drift the bs256 64k arms showed.
for opt in ["adamw", "muon"]:
    sweep(f"tune_{opt}_64k_bs32",
          selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_eps1e17_64k_bs32",
          lrs=[1.25e-5], priority=2, optimizer=opt, n_docs=64000,
          batch_size=32, grad_accum_steps=2,
          notes="Endpoint extension: 2.5e-5 won the 3-point grid.")

# Pre-emptive low extension on the 128k arms. The 128k grid was already
# re-centred once on the halving trend (1000 steps -> 5e-5, 4000 -> 2.5e-5), and
# the trend has since held at 256k, whose two reported points make 1.25e-5 beat
# 6.25e-6. So 8000 steps is predicted at 1.25e-5, the LOW ENDPOINT of its own
# grid -- the extension the comment above the GRID says to run "as normal" once
# that happens. Running it up front costs nothing here: it went onto A40s that
# had no queued work, and it removes an hour and a half of serial latency from
# the rung if the prediction holds. If 2.5e-5 wins interior instead, this row is
# simply surplus.
for opt in ["adamw"]:
    sweep(f"tune_{opt}_128k_bs32",
          selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_eps1e17_128k_bs32",
          lrs=[6.25e-6], priority=2, optimizer=opt, n_docs=128000,
          batch_size=32, grad_accum_steps=2,
          notes="Pre-emptive LOW extension, and it backed the wrong direction: "
                "both arms then won at the high endpoint 5e-5. Kept as a "
                "registered row because the run was launched, but it was "
                "cancelled once the grid reported and never measured.")

# The extension the grid actually asked for: 5e-5 won the high endpoint on BOTH
# arms, so each gets one 2x step outward.
for opt in ["adamw", "muon"]:
    sweep(f"tune_{opt}_128k_bs32",
          selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_eps1e17_128k_bs32",
          lrs=[1e-4], priority=2, optimizer=opt, n_docs=128000,
          batch_size=32, grad_accum_steps=2,
          notes="Endpoint extension: 5e-5 won the 3-point grid.")

# Next rung of the TOKEN axis (bs256, 2 epochs), which currently runs 4k, 8k,
# 16k, 32k with 64k banks building. 128k at bs256 is 1000 optimiser steps, so it
# is cheap to tune -- the cost of this rung is the bank, not the sweep.
#
# Centre 1e-4: both 64k arms won there, interior in {5e-5, 1e-4, 2e-4, 4e-4}.
# Two independent arguments agree on it rather than on a lower centre. The step
# drift says the optimum falls as steps grow, and 128k doubles 64k to 1000 steps.
# The CONTROLS batch rule says lr scales as sqrt(batch), and the bs32 ladder
# chose 5e-5 at 1000 steps, which is 1.4e-4 after sqrt(256/32) -- just above 1e-4.
# The drift pulls down, the batch rule pushes up, and 1e-4 sits between them, so
# unlike the bs32 ladder there is no case for centring lower up front. Extend as
# normal if 5e-5 wins its own endpoint.
for opt in ["adamw", "muon"]:
    sweep(f"tune_{opt}_128k_bs256",
          selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_eps1e17_128k_bs256",
          lrs=[5e-5, 1e-4, 2e-4], priority=2, optimizer=opt, n_docs=128000,
          batch_size=256, grad_accum_steps=16,
          notes="Token axis at bs256, 2 epochs (1000 steps). Centre 1e-4 = the 64k winner.")

# High extension on the adamw arm only: 2e-4 edged 1e-4 by 0.0001 nats, which the
# selection rule treats as an endpoint win. muon took 1e-4 interior and needs no
# extension. 64k measured 4e-4 at 3.2671 against its 1e-4 winner at 3.2314, so
# this one is expected to come back clearly worse -- but it is 1000 steps on an
# otherwise idle pair, and measuring is cheaper than arguing.
for opt in ["adamw"]:
    sweep(f"tune_{opt}_128k_bs256",
          selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_eps1e17_128k_bs256",
          lrs=[4e-4], priority=2, optimizer=opt, n_docs=128000,
          batch_size=256, grad_accum_steps=16,
          notes="Endpoint extension: 2e-4 edged the 3-point grid by 0.0001 nats.")

# Step-scaling sweep results, measured 2026-08-25 (bs32, ga 2, nproc 2, pinned venv).
# Both 32k arms win on the INTERIOR point, so the CONTROLS batch-16-32 centre of
# 5e-5 was right and no endpoint extension is needed. Note how flat they are --
# 0.004 nats across a 4x lr range -- so 5e-5 is "no worse than its neighbours"
# rather than a sharp optimum, the same pattern gpt2-medium showed.
# ---------------------------------------------------------------------------------
# DISTRIBUTION SHIFT: london-llm-1800 at the anchor setting.
#
# Every row in the grid reads ms 0.98+, which does not match Lucia's WikiText
# results, and held-out loss falls only ~0.1 nats over a whole run. The suspicion
# is that smollm2 sits too close to GPT-2's pre-training distribution for
# fine-tuning to move the model enough to test anything -- if so these numbers
# describe the corpus, not metasmoothness.
#
# london_16k.hf is built by scripts/prep_london.py from a pre-1931 corpus, packed
# to the identical shape: gpt2 tokenizer, 512-token chunks, nested. So this sweep
# is the anchor config with ONLY the text changed -- same model, batch, epochs and
# seed as sm_adamw_eps1e17_16k_bs256, 125 steps.
#
# Grid centred on 2e-4, which both anchor arms chose on smollm2. A corpus this far
# from pre-training may well want a different lr, so the endpoints matter here more
# than usual; extend as normal if one wins.
for opt in ["adamw", "muon"]:
    sweep(f"tune_{opt}_london16k_bs256",
          selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_london16k_bs256",
          lrs=[1e-4, 2e-4, 4e-4], priority=2, optimizer=opt, n_docs=16000,
          batch_size=256, grad_accum_steps=16,
          notes="Distribution-shift control: anchor config, pre-1931 corpus.")

# Upward extension: adamw won at 4e-4, the top of its grid. A corpus this far from
# pre-training plausibly wants a larger step than smollm2 did, so this goes two
# octaves rather than one.
for opt in ["adamw", "muon"]:
    sweep(f"tune_{opt}_london16k_bs256",
          selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_london16k_bs256",
          lrs=[8e-4, 1.6e-3], priority=2, optimizer=opt, n_docs=16000,
          batch_size=256, grad_accum_steps=16,
          notes="Endpoint extension: 4e-4 won the 3-point london grid.")

# london at bs16 (2000 steps). Batch is the one axis known to move ms hard on
# smollm2 -- 0.9930 at bs256 down to 0.9133 at bs16 -- and london came in at
# 0.9867 at bs256, so this is the sharp probe for whether the corpus and the
# batch interact. Grid centred on 2e-4 = 8e-4 * sqrt(16/256).
for opt in ["adamw", "muon"]:
    sweep(f"tune_{opt}_london16k_bs16",
          selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_london16k_bs16",
          lrs=[1e-4, 2e-4, 4e-4], priority=2, optimizer=opt, n_docs=16000,
          batch_size=16, grad_accum_steps=1,
          notes="Distribution-shift control at small batch, pre-1931 corpus.")

# london at larger N, same bs256 anchor config: does ms hold as the corpus grows
# when the corpus is far from pre-training?
#
# This was adamw-only, on the grounds that muon deadlocked on london above 16k.
# It never did. The hang was a damaged datasets .map() cache file sitting beside
# london_32k.hf and london_64k.hf -- pyarrow blocked memory-mapping it inside
# attach_doc_ids_if_missing, for adamw and muon alike. Moving the caches aside
# fixed both, and muon london 32k has since completed all three lrs. See
# messages/2026-08-26-error-analysis-log.md; notes/muon32k_hang.md is superseded.
for n_docs, tag in ((32000, "london32k"), (64000, "london64k"),
                    (128000, "london128k")):
    for opt in ("adamw", "muon"):
        sweep(f"tune_{opt}_{tag}_bs256",
              selects_lr_for=f"plan_{'adam' if opt == 'adamw' else 'muon'}_{tag}_bs256",
              lrs=[4e-4, 8e-4, 1.6e-3], priority=2, optimizer=opt, n_docs=n_docs,
              batch_size=256, grad_accum_steps=16,
              notes="Distribution-shift N-scaling, pre-1931 corpus.")

# London sweep, measured 2026-08-26 on A40, nproc 2, evaluated on
# london_heldout_4k.hf -- NOT the smollm2 heldout, which would select whichever lr
# best fits the wrong distribution. The held-out set is packed from source rows
# 40000+, verified zero-overlap against london_128k.
#
#     adamw   1e-4 3.8821   2e-4 3.8641   4e-4 3.8471
#
# 4e-4 wins the HIGH endpoint, so the grid extends up; 8e-4 and 1.6e-3 are running.
#
# The baseline this exists to establish, stock gpt2 against each held-out set:
#
#     smollm2   3.4981 -> 3.2572 (anchor)   drop 0.241
#     london    4.0181 -> 3.8471 (4e-4)     drop 0.171
#
# So london IS further from pre-training -- gpt2 starts half a nat worse on it --
# but at the anchor's 125-step budget fine-tuning moves the model LESS, not more.
# Together with the winner sitting at the top of its grid, that reads as the lr
# being under-tuned for this corpus rather than the corpus failing to shift the
# distribution. The extension settles it.
LONDON_HELDOUT = {
    # First london results ABOVE 16k, measured 2026-08-26 against
    # london_heldout_4k (4000 docs x 512 tok). These exist only because the
    # stale datasets .map() cache was the hang -- every 32k/64k london run had
    # failed to start until the cache files were moved aside.
    #
    # muon london 32k bs256: 8e-4 wins and wins INTERIOR, so no endpoint
    # extension is needed. Same winner as london 16k for both optimizers, so the
    # london lr optimum is stable across a doubling of the corpus.
    # adamw london 32k bs256, measured 2026-08-26 alongside the muon arm. 8e-4
    # wins INTERIOR here too, so BOTH optimizers pick 8e-4 at 32k and both picked
    # 8e-4 at 16k. The london lr optimum does not move with corpus size over this
    # range, which is worth knowing before spending a sweep at 64k.
    #
    # muon is very slightly ahead at every lr (3.7842 vs 3.7873 at the winner,
    # 0.003 nats). That is far too small to call a difference, and it matches the
    # 16k finding that the two optimizers are indistinguishable on this corpus --
    # unlike the smollm2 grid, where they separate.
    "tune_adamw_london32k_bs256_lr0.0004": 3.8029,
    "tune_adamw_london32k_bs256_lr0.0008": 3.7873,
    "tune_adamw_london32k_bs256_lr0.0016": 3.7893,
    "tune_adamw_london64k_bs256_lr0.0004": 3.6737,
    "tune_muon_london32k_bs256_lr0.0004": 3.8005,
    "tune_muon_london32k_bs256_lr0.0008": 3.7842,
    "tune_muon_london32k_bs256_lr0.0016": 3.7915,
    # muon london 64k bs256, first point only -- 8e-4 and 1.6e-3 are still
    # training, so this is NOT yet a winner, just the one measurement.
    "tune_muon_london64k_bs256_lr0.0004": 3.6616,
    "tune_adamw_london16k_bs256_lr0.0001": 3.8821,
    "tune_adamw_london16k_bs256_lr0.0002": 3.8641,
    "tune_adamw_london16k_bs256_lr0.0004": 3.8471,
    # Extension measured 2026-08-26. 8e-4 wins INTERIOR, so the london adamw lr is
    # settled at 8e-4 -- four times the smollm2 winner, which is the size of lr
    # correction this corpus wanted.
    "tune_adamw_london16k_bs256_lr0.0008": 3.8397,
    "tune_adamw_london16k_bs256_lr0.0016": 3.8551,
    # muon london, measured 2026-08-26 against london_heldout_4k. Winner 8e-4,
    # INTERIOR -- the same lr adamw settled on, and to within 0.0003 the same loss
    # (3.8394 vs 3.8397). The two optimizers are indistinguishable on this corpus
    # at bs256, which is not what the smollm2 grid shows.
    "tune_muon_london16k_bs256_lr0.0001": 3.8975,
    "tune_muon_london16k_bs256_lr0.0002": 3.8724,
    "tune_muon_london16k_bs256_lr0.0004": 3.8490,
    "tune_muon_london16k_bs256_lr0.0008": 3.8394,
    "tune_muon_london16k_bs256_lr0.0016": 3.8593,
    # london at bs16 (2000 steps), measured 2026-08-26 against london_heldout_4k.
    # Both arms win 2e-4 INTERIOR, which is exactly 8e-4 * sqrt(16/256) -- the
    # CONTROLS batch rule predicting the bs256 winner's rescaling on the nose.
    #
    # The optimizers separate here in a way they do not at bs256. There they were
    # identical (3.8397 adamw, 3.8394 muon); at bs16 muon is 0.022 better.
    "tune_adamw_london16k_bs16_lr0.0001": 3.8487,
    "tune_adamw_london16k_bs16_lr0.0002": 3.8463,
    "tune_adamw_london16k_bs16_lr0.0004": 3.8641,
    "tune_muon_london16k_bs16_lr0.0001":  3.8283,
    "tune_muon_london16k_bs16_lr0.0002":  3.8240,
    "tune_muon_london16k_bs16_lr0.0004":  3.8451,
    # london at 128k, bs256 (1000 steps), measured 2026-08-26 against
    # london_heldout_4k. 1.6e-3 still running.
    #
    # Note how much more N buys on this corpus than on smollm2. london goes
    # 3.8397 at 16k to 3.5264 at 128k, a 0.31 nat gain from 8x the data, while
    # the smollm2 token axis moves ~0.02 over the same range. A distant corpus is
    # data-hungry in a way the near one is not, which is worth keeping in mind
    # when reading any london result against a smollm2 one at matched N.
    "tune_adamw_london128k_bs256_lr0.0004": 3.5725,
    "tune_adamw_london128k_bs256_lr0.0008": 3.5264,
    "tune_adamw_london128k_bs256_lr0.0016": 3.4992,
}

# london 128k wins at 1.6e-3, the HIGH endpoint, so the grid extends up: 3.2e-3
# and 6.4e-3 are running.
#
# The direction is the interesting part. On london the optimum RISES with N --
# 8e-4 at 16k, at least 1.6e-3 at 128k -- while at fixed bs256 more N means more
# steps, and on the smollm2 ladder more steps meant a LOWER optimum (5e-5 at 1000
# steps, 2.5e-5 at 4000). So the lr-versus-steps trend reverses with the corpus,
# the same way the optimizer ranking did. Anything extrapolated from the smollm2
# grid needs re-checking on london rather than assumed.

BS32_STEP_HELDOUT = {
    # 128k at bs256 (1000 steps), measured 2026-08-25 on A40, nproc 2, pinned venv.
    # Centring on 1e-4 was right for muon, which wins it interior. adamw is a tie
    # rather than a win: 2e-4 beats 1e-4 by 0.0001 nats, which is an order of
    # magnitude below the 0.0075 spread of its own grid and far below anything
    # this axis resolves. The selection rule still reads it as a high endpoint,
    # so the extension to 4e-4 is registered and running rather than argued away
    # -- the last time a prediction was substituted for the measurement on this
    # axis it went the wrong way.
    "tune_adamw_128k_bs256_lr5e-05":  3.2167,
    "tune_adamw_128k_bs256_lr0.0001": 3.2107,
    "tune_adamw_128k_bs256_lr0.0002": 3.2106,
    "tune_muon_128k_bs256_lr5e-05":   3.2181,
    "tune_muon_128k_bs256_lr0.0001":  3.2108,
    "tune_muon_128k_bs256_lr0.0002":  3.2114,
    # 128k, measured 2026-08-25 on A40 (allium-0 adamw, iris-0/secret-ord-0/
    # bellflower-0 muon), nproc 2, pinned venv. Both arms win the HIGH endpoint.
    "tune_adamw_128k_bs32_lr1.25e-05": 3.2219,
    "tune_adamw_128k_bs32_lr2.5e-05":  3.2150,
    "tune_adamw_128k_bs32_lr5e-05":    3.2129,
    "tune_muon_128k_bs32_lr1.25e-05":  3.2183,
    "tune_muon_128k_bs32_lr2.5e-05":   3.2110,
    "tune_muon_128k_bs32_lr5e-05":     3.2079,
    # Endpoint extension, measured 2026-08-25 on A40 secret-ord-0, nproc 2.
    # 1e-4 comes in WORSE than 5e-5, so 5e-5 is now an interior winner and the
    # adamw 128k lr is settled at 5e-5 -- no further extension.
    "tune_adamw_128k_bs32_lr0.0001":   3.2224,
    "tune_adamw_32k_bs32_lr2.5e-05": 3.2380,
    "tune_adamw_32k_bs32_lr5e-05":   3.2342,
    "tune_adamw_32k_bs32_lr0.0001":  3.2380,
    "tune_muon_32k_bs32_lr2.5e-05":  3.2351,
    "tune_muon_32k_bs32_lr5e-05":    3.2310,
    "tune_muon_32k_bs32_lr0.0001":   3.2359,
}
# One loop per results dict. Each sets status and run_dir as well as the loss --
# splitting a loop and leaving those lines behind silently demotes every row it
# covers back to "empty", which is what happened when the london loop was first
# inserted here: 16 measured 32k/128k rows went empty in one rebuild.
_heldout_matched = set()
for _table in (BS32_STEP_HELDOUT, LONDON_HELDOUT):
    for _r in rows:
        _hl = _table.get(_r["run_id"])
        if _hl is not None:
            _heldout_matched.add(_r["run_id"])
            _r["heldout_loss"] = _hl
            _r["status"] = "measured"
            _r["run_dir"] = f"/mnt/ssd-2/lucia/paper_runs/tuning/{_r['run_id']}_s42"

# These tables UPDATE rows; they never create them. A key naming a run with no
# sweep() row above matches nothing and its measurement is dropped without a
# word. Four muon london values sat in this file for hours doing exactly that,
# because the muon london 32k/64k sweeps had never been defined. Fail loudly.
_heldout_keys = set(BS32_STEP_HELDOUT) | set(LONDON_HELDOUT)
_heldout_orphans = sorted(_heldout_keys - _heldout_matched)
if _heldout_orphans:
    raise SystemExit(
        "%d heldout result(s) name a run with no sweep() row, so they would be "
        "silently dropped:\n  %s\nAdd the sweep, or remove the entry."
        % (len(_heldout_orphans), "\n  ".join(_heldout_orphans)))

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
