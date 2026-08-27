"""Build experiments.csv — the source-of-truth grid for the metasmoothness paper.

One row per experiment: a training configuration plus the attribution metrics measured on it.
Parameter columns first, result columns last. Empty result cells are *not-yet-run*, not
failures: an agent with spare compute can select rows where a result column is empty and fill
it. Regenerate with `python build_experiments_csv.py`; edit the row tables here, never the CSV.

Admission policy (2026-08-20)
-----------------------------
Only measurements made under the CURRENT implementation are admitted:

1. **Per-epoch shuffle everywhere.** Training, the leave-k-out bank retrains, and the
   metasmoothness probe must all reshuffle each epoch (commit 1e6eea7f, PR #352). Runs from the
   older shuffle-once-then-`.repeat` ("rep") code are excluded — including every WikiText run,
   the original headline SmolLM2 grid MAGIC values, and every Shampoo/SOURCE/TrackStar/baseline
   scoring, all of which used rep banks. The excluded numbers remain in LDS_RESULTS.md /
   BASELINE_LDS.md / SHAMPOO_RESULTS.md; see EXPERIMENTS_CSV.md for the exclusion table.
   Exception: `num_epochs == 1` runs are shuffle-agnostic (one pass, one order) and are admitted
   with shuffle="agnostic_1ep".

2. **No active dropout.** Every admitted row trained with dropout effectively off. Most GPT-2
   rows *configure* the HF default 0.1 but run with `train_mode=false`, where the trainer calls
   `model.eval()` and the rate is inert — recorded as dropout_cfg=0.1 / dropout_effective=0.0.
   The two WikiText runs with dropout genuinely active (train_mode=true) are excluded, and would
   fall to filter 1 anyway. All *planned* rows set dropout_cfg=0.0 explicitly so the cfg/effective
   split cannot recur.

Conventions
-----------
status      done    = metasmoothness, magic_lds and ekfac_lds all measured
            partial = at least one of the three measured
            planned = nothing measured; the row exists to be claimed
steps       ceil(n_docs * num_epochs / batch_size). Exact: the trainer concatenates all
            epochs into one shuffled sequence (shuffled_epochs) and pads that sequence once,
            at its end, with zero-weight copies of the last document — so a fractional config
            (4k/8k at bs256, 16k at bs512) has exactly one partial batch in the whole run.
eps_root    epsilon inside the AdamW sqrt: m / (sqrt(v + eps_root) + adam_eps). Non-standard.
            For muon it reaches only the AdamW-fallback params (121,344 of 163M = 0.07%).
warmup      lr-warmup as a fraction of total steps (values below 1; at 1 or above the trainer
            reads absolute steps). Fixed control 0.25; no row varies it. The attribution-window
            axis ("warm start") is a pre-training experiment — see EXPERIMENTS_CSV.md.
ckpt_avg_k  last-k checkpoints the QUERY GRADIENT is averaged over (1 = none). Needs
            cleanup_ckpts=false at train time.
heldout_loss  mean per-token CE on heldout_4k (scripts/heldout_eval.py). The tuning protocol in
            CONTROLS.md requires it for every row; empty = not yet measured. Anchor-config
            reference (from the lr-sweep twin at ga4): adamw 3.2572, muon 3.2570,
            untrained gpt2 3.4981.
reusable    bank+scores = retrained models on disk; re-score a new method without retraining
            bank        = retrained models only
            ms_only     = metasmoothness.json only
            none        = artifacts deleted; re-run from the parameter columns
"""

import csv
import os

COLUMNS = [
    # --- identity ---
    "run_id", "status", "family",
    "node_in_charge", "node_checkin_date",
    # --- model ---
    "model", "model_init", "arch_mod", "n_params_m", "logit_scale",
    # --- data ---
    "dataset", "n_docs", "chunk_length",
    # --- optimizer ---
    "optimizer", "lr", "lr_scheduler", "warmup", "eps_root", "adam_eps",
    "beta1", "beta2", "weight_decay", "max_grad_norm",
    # --- training ---
    "batch_size", "grad_accum_steps", "num_epochs", "steps",
    "dropout_cfg", "dropout_effective", "shuffle", "seed", "precision", "ckpt_avg_k",
    # --- attribution / estimator ---
    "attr_window_frac", "n_subsets", "subset_fraction", "n_queries",
    # --- results ---
    "metasmoothness", "ms_direction_seed", "ms_fd_step", "ms_total_movement_l1",
    # Tail-filter ("filter power"): per query, remove the top filter_frac of
    # documents the scorer ranks most influential, retrain once, measure that
    # query's loss change against the row's own bank subsets (random removals
    # of the same size). Raw loss deltas shrink as the training set grows, so
    # they are NOT comparable across dataset sizes -- the normalized columns
    # are the ones to compare. percentile is nonparametric and the safest.
    # Per-query raw data lives in data/filter_raw/ and data/filter_power.csv;
    # these columns are a summary of a score design that may still change.
    # Tail filter: the mean query-loss delta under each way of choosing which 1%
    # of the training data to remove, in nats, with a 10k bootstrap CI over
    # queries. Three directly comparable quantities rather than a derived score --
    # z was dropped because within a run both scorers share the same bank as
    # their control, so random_sd is identical and z is just the raw delta times
    # a constant. Raw per-query data: data/filter_raw/, data/filter_deltas.csv.
    "filter_frac", "filter_n_queries",
    "filter_random_delta", "filter_random_lo", "filter_random_hi",
    "filter_ekfac_delta", "filter_ekfac_lo", "filter_ekfac_hi",
    "filter_magic_delta", "filter_magic_lo", "filter_magic_hi",
    "magic_lds", "magic_ci_lo", "magic_ci_hi", "magic_n_queries",
    "ekfac_lds", "ekfac_ci_lo", "ekfac_ci_hi", "ekfac_n_subsets",
    "train_loss", "heldout_loss", "delta_l1", "delta_l2",
    # --- provenance ---
    "run_dir", "bank_dir", "code_commit", "reusable", "source_doc", "notes",
]

GPT2_FT = dict(
    family="gpt2_ft", model="gpt2", model_init="pretrained", arch_mod="none",
    n_params_m=124, logit_scale=1.0, dataset="smollm2", chunk_length=0,
    optimizer="adamw", lr=8e-4, lr_scheduler="polynomial", warmup=0.25,
    adam_eps=1e-8, beta1=0.95, beta2=0.975, weight_decay=0.01, max_grad_norm="",
    batch_size=64, grad_accum_steps=1, num_epochs=2,
    dropout_cfg=0.1, dropout_effective=0.0, shuffle="per_epoch", seed=42,
    precision="fp32", ckpt_avg_k=1, attr_window_frac=0.0, subset_fraction=0.01,
    n_subsets=50, ms_direction_seed=0, ms_fd_step=0.1, source_doc="LDS_RESULTS.md",
)

OLMO2 = dict(
    family="olmo2_scratch", model="olmo2_reinit", model_init="scratch", arch_mod="none",
    n_params_m=124, logit_scale=1.0, dataset="smollm2", chunk_length=0,
    optimizer="muon", lr=9e-3, lr_scheduler="polynomial", warmup=0.25,
    eps_root=1e-6, adam_eps=1e-8, beta1=0.95, beta2=0.975, weight_decay=0.1,
    max_grad_norm="", batch_size=128, grad_accum_steps=1, num_epochs=6,
    dropout_cfg=0.0, dropout_effective=0.0, shuffle="per_epoch", seed=42,
    precision="fp32", ckpt_avg_k=1, attr_window_frac=0.0, subset_fraction=0.01,
    n_subsets=50, ms_direction_seed=0, ms_fd_step=0.1, source_doc="LDS_RESULTS.md",
)

rows = []


def add(base, **kw):
    r = dict(base)
    r.update(kw)
    if "steps" not in r and r.get("n_docs") and r.get("batch_size") and r.get("num_epochs"):
        import math
        r["steps"] = math.ceil(r["n_docs"] * r["num_epochs"] / r["batch_size"])
    rows.append(r)


PEREPOCH_RUNS = "/mnt/ssd-1/lucia/perepoch/runs"

# =====================================================================================
# 1. GPT-2 ft / SmolLM2 — per-epoch grid (banks + ms at /mnt/ssd-1/lucia/perepoch/runs).
#    These replace the rep-era headline grid. MAGIC has NOT been run on any per-epoch
#    bank config yet (the rep-era MAGIC values do not transfer: the metagradient replays
#    the training order). train_loss was not recorded in the per-epoch replication —
#    backfill from the bank checkpoints.
# =====================================================================================
GRID = [
    # run_id                      n      opt     lr    eps    bs  ep  ms      ekfac  lo     hi     dl1     dl2    bankdir
    ("sm_adam_eps0_16k",       16000, "adamw", 8e-4, 0,      64, 2, 0.4269, 0.1186, 0.074, 0.164, 0.0938, 0.0964, "adam_eps0_16k"),
    ("sm_adam_eps0_8k",         8000, "adamw", 8e-4, 0,      64, 2, 0.6226, 0.1555, 0.121, 0.190, 0.0714, 0.0742, "adam_eps0_8k"),
    ("sm_adam_eps0_4k",         4000, "adamw", 8e-4, 0,      64, 2, 0.7724, 0.1540, 0.106, 0.202, 0.0565, 0.0587, "adam_eps0_4k"),
    ("sm_adam_eps1e10_4k",      4000, "adamw", 8e-4, 1e-10,  64, 2, 0.7883, 0.2020, 0.164, 0.240, 0.0272, 0.0283, "adam_eps1e10_4k"),
    ("sm_adam_eps1e8_4k",       4000, "adamw", 8e-4, 1e-8,   64, 2, 0.8755, 0.3095, 0.269, 0.351, 0.0068, 0.0084, "adam_eps1e8_4k"),
    ("sm_adam_eps1e8_4k_rep2",  4000, "adamw", 8e-4, 1e-8,   64, 2, 0.8755, 0.3048, 0.264, 0.346, 0.0068, 0.0084, "adam_eps1e8_4k_drop0"),
    ("sm_adam_eps1e8_4k_bs128", 4000, "adamw", 8e-4, 1e-8,  128, 4, 0.9822, 0.3576, 0.317, 0.397, 0.0065, 0.0080, "adam_eps1e8_bs128_4k"),
    ("sm_adam_eps1e6_8k",       8000, "adamw", 8e-4, 1e-6,   64, 2, 0.9786, 0.2950, 0.253, 0.338, 0.0024, 0.0028, "adam_eps1e6_8k"),
    ("sm_adam_eps1e6_4k",       4000, "adamw", 8e-4, 1e-6,   64, 2, 0.9952, 0.3076, 0.268, 0.346, 0.0015, 0.0021, "adam_eps1e6_4k"),
    ("sm_muon_eps0_5e5_4k",     4000, "muon",  5e-5, 0,      64, 4, 0.9960, 0.4648, 0.422, 0.504, 0.0053, 0.0061, "muon_eps0_4k"),
    ("sm_muon_eps1e6_5e5_4k",   4000, "muon",  5e-5, 1e-6,   64, 4, 0.9962, 0.4630, 0.420, 0.503, 0.0053, 0.0061, "muon_eps1e6_4k"),
]
# train_loss backfilled 2026-08-20: eval-mode mean CE of each bank's saved base model
# over its own train set (scripts/heldout_eval.py --heldout <train set>), GPU eval.
TRAIN_LOSS = {"adam_eps0_16k": 2.6482, "adam_eps0_4k": 2.5846, "adam_eps0_8k": 2.5983,
              "adam_eps1e10_4k": 2.6896, "adam_eps1e6_4k": 3.1849, "adam_eps1e6_8k": 3.1836,
              "adam_eps1e8_4k": 3.0175, "adam_eps1e8_4k_drop0": 3.0175,
              "adam_eps1e8_bs128_4k": 2.9837, "muon_eps0_4k": 3.0901, "muon_eps1e6_4k": 3.0912}

# heldout_loss backfilled the same way against heldout_4k.hf; all beat untrained
# gpt2 (3.4981; CONTROLS rule 4). Note the memorisation trade-off: eps0/8e-4 rows
# have the lowest train CE and the worst heldout.
HELDOUT = {"adam_eps0_16k": 3.3517, "adam_eps0_4k": 3.4230, "adam_eps0_8k": 3.3856,
           "adam_eps1e10_4k": 3.3833, "adam_eps1e6_4k": 3.2946, "adam_eps1e6_8k": 3.2716,
           "adam_eps1e8_4k": 3.3065, "adam_eps1e8_4k_drop0": 3.3065,
           "adam_eps1e8_bs128_4k": 3.3153, "muon_eps0_4k": 3.2796, "muon_eps1e6_4k": 3.2798}

# Legacy eps-root-damping family: OUT of the paper CSV (ruling 2026-08-22);
# values archived in LDS_RESULTS.md.
for rid, n, opt, lr, eps, bs, ep, ms, ek, lo, hi, l1, l2, bd in []:
    extra = {}
    if bd in TRAIN_LOSS:
        extra["train_loss"] = TRAIN_LOSS[bd]
    if bd in HELDOUT:
        extra["heldout_loss"] = HELDOUT[bd]
    if rid == "sm_adam_eps1e8_4k_rep2":
        extra = dict(extra, dropout_cfg=0.0,
                     notes="Replicate of sm_adam_eps1e8_4k: identical effective training "
                           "(its rep-era twin differed only in the INERT dropout cfg). The "
                           "EK-FAC gap to its twin (0.3048 vs 0.3095) is bank-level noise.")
    add(GPT2_FT, run_id=rid, status="partial", n_docs=n, optimizer=opt, lr=lr, eps_root=eps,
        batch_size=bs, num_epochs=ep, metasmoothness=ms, ekfac_lds=ek, ekfac_ci_lo=lo,
        ekfac_ci_hi=hi, ekfac_n_subsets=50, delta_l1=l1, delta_l2=l2,
        bank_dir=f"{PEREPOCH_RUNS}/{bd}/bank", run_dir=f"{PEREPOCH_RUNS}/{bd}",
        code_commit="1e6eea7f+", reusable="bank+scores", **extra)

# ms-only per-epoch points (no bank built, or the only bank is rep-era and excluded)
MS_ONLY = [
    ("sm_adam_eps1e6_16k",      16000, "adamw", 8e-4, 1e-6,  64, 2, 0.9954,
     "rep-era bank exists (EK-FAC 0.3815, excluded); rebuild the bank per-epoch."),
    ("sm_adam_eps1e6_32k",      32000, "adamw", 8e-4, 1e-6,  64, 2, 0.9979,
     "rep-era bank exists (EK-FAC 0.3575, excluded); rebuild the bank per-epoch."),
    ("sm_adam_eps1e6_4k_ep4",    4000, "adamw", 8e-4, 1e-6,  64, 4, 0.9989, ""),
    ("sm_adam_eps0_4k_ep4",      4000, "adamw", 8e-4, 0,     64, 4, 0.6836,
     "Data fixed, steps 125->250: ms 0.7724 -> 0.6836."),
    ("sm_adam_eps1e8_4k_bs256",  4000, "adamw", 8e-4, 1e-8, 256, 8, 0.9984,
     "LDS was blocked on OOM pre-grad_accum; grad_accum_steps now exists, so it is unblocked."),
    ("sm_muon_eps1e8_5e5_4k",    4000, "muon",  5e-5, 1e-8,  64, 4, 0.9961, ""),
    ("sm_muon_eps1e6_5e5_8k",    8000, "muon",  5e-5, 1e-6,  64, 4, 0.9957, ""),
    ("sm_muon_eps1e6_5e5_16k",  16000, "muon",  5e-5, 1e-6,  64, 4, 0.9952, ""),
    ("sm_muon_eps1e6_5e5_32k",  32000, "muon",  5e-5, 1e-6,  64, 4, 0.9947,
     "Muon ms is flat over 8x steps (0.9965 -> 0.9947)."),
    ("sm_muon_eps0_5e5_8k",      8000, "muon",  5e-5, 0,     64, 4, 0.9956, ""),
    ("sm_muon_eps0_5e5_16k",    16000, "muon",  5e-5, 0,     64, 4, 0.9951, ""),
    ("sm_muon_eps0_5e5_4k_bs16", 4000, "muon",  5e-5, 0,     16, 4, 0.9932,
     "Batch size does not un-saturate muon; adam collapses to 0.500 at bs16 (rep-era, excluded)."),
]
for rid, n, opt, lr, eps, bs, ep, ms, note in []:
    add(GPT2_FT, run_id=rid, status="partial", n_docs=n, optimizer=opt, lr=lr, eps_root=eps,
        batch_size=bs, num_epochs=ep, metasmoothness=ms, reusable="ms_only",
        run_dir="/mnt/ssd-2/lucia/muon_ms_steps" if opt == "muon" and n > 4000 else "",
        notes=note)

# epochs=1: shuffle-agnostic, so the rep-era measurements remain valid. The only row
# with all three metrics.
# [removed 2026-08-22: legacy eps-damping row, not this paper's controls; ms archived in LDS_RESULTS.md]
# add(GPT2_FT, run_id="sm_adam_eps1e8_4k_bs32_ep1", status="done", n_docs=4000, eps_root=1e-8,
#     batch_size=32, num_epochs=1, shuffle="agnostic_1ep", metasmoothness=0.837,
#     ekfac_lds=0.1781, ekfac_n_subsets=50, magic_lds=0.05, magic_ci_lo=-0.054,
#     magic_ci_hi=0.145, magic_n_queries=20, train_loss=3.07, delta_l1=0.0087,
#     delta_l2=0.0100, run_dir="/mnt/ssd-2/lucia/muon4k/magicroll_bs32",
#     reusable="bank+scores",
#     notes="1 epoch = one pass over one order, so the shuffle implementation cannot matter; "
#           "admitted although measured on the older code. MAGIC on the FIXED metagrad code "
#           "(c0f11ba8); CI spans zero.")

# eps_root 1e-17 pair (scaling_magic, 2026-08-07): per-epoch trained, MAGIC measured.
# ms + EK-FAC filled by the fill_* work rows: D7-canonical EK-FAC scoring + ms probe,
# both on code 10874f93 (scores/ms under /mnt/ssd-2/lucia/s16k_<opt>/). EK-FAC estimator:
# scripts/ekfac_lds.py (loss-signed, validated on the per-epoch grid).
ANCHOR_FILL = {
    "adamw": dict(ekfac_lds=0.4251, ekfac_ci_lo=0.3772, ekfac_ci_hi=0.4693,
                  ekfac_n_subsets=100, metasmoothness=0.9928, ms_direction_seed=0,
                  ms_fd_step=0.1),
    "muon": dict(ekfac_lds=0.4285, ekfac_ci_lo=0.3856, ekfac_ci_hi=0.4674,
                 ekfac_n_subsets=100, metasmoothness=0.9963, ms_direction_seed=0,
                 ms_fd_step=0.1),
}
for opt, lds, lo, hi in [("adamw", 0.9333, 0.9186, 0.9448), ("muon", 0.8470, 0.8274, 0.8685)]:
    add(GPT2_FT, run_id=f"sm_{opt}_eps1e17_16k_bs256",
        status="done", n_docs=16000,
        optimizer=opt, lr=2e-4, eps_root=1e-17, batch_size=256, grad_accum_steps=16,
        num_epochs=2, magic_lds=lds, magic_ci_lo=lo, magic_ci_hi=hi, magic_n_queries=20,
        n_subsets=100, n_queries=20, run_dir=f"/mnt/ssd-2/lucia/s16k_{opt}",
        bank_dir=f"/mnt/ssd-2/lucia/s16k_{opt}/merged", code_commit="docs-4",
        reusable="bank", source_doc="examples/scaling_magic/LDS_RESULTS.md",
        **ANCHOR_FILL.get(opt, {}),
        notes="Paired diff adamw-muon = +0.0863 [+0.0670, +0.1052], 19/20 per-query wins; "
              "identical subset lists. EK-FAC scored on code 10874f93 (D7 canonical); "
              "EK-FAC cannot separate the optimizers (0.4251 vs 0.4285) while MAGIC "
              "does (+0.086). ms probe: total_movement_l1 34147 (adamw). Base-training "
              "ckpts deleted (retrains reproduce deterministically)." +
              (" adamw scores rebuilt from per-query .pt files (padded-query bug on docs-4)."
               if opt == "adamw" else ""))

# [moved out 2026-08-22: OLMo2 family is a different study; values in LDS_RESULTS.md]
# =====================================================================================
# 2. OLMo2 from-scratch (dropout genuinely 0.0; per-epoch rows only). Different model
#    family — kept for the pre-training endpoint; filter on `family` for GPT-2-only plots.
# =====================================================================================
# add(OLMO2, run_id="olmo2_muon_16k_full", status="partial", n_docs=16000,
#     metasmoothness=-0.000, train_loss=2.92, delta_l1=4.56, delta_l2=4.10,
#     run_dir="/mnt/ssd-2/lucia/scratch_olmo", reusable="bank",
#     notes="Dead endpoint: ms ~= 0 (below the ~0.02 information floor). The rep-era full-run "
#           "EK-FAC (0.0175, CI spans 0) is excluded; the rep bank exists if a per-epoch rebuild "
#           "is ever wanted, but the per-epoch ms says the answer is already 'unattributable'.")
# add(OLMO2, run_id="olmo2_muon_16k_tail083", status="partial", n_docs=16000,
#     attr_window_frac=0.833, metasmoothness=0.984, ekfac_lds=0.161, ekfac_ci_lo=0.123,
#     ekfac_ci_hi=0.198, ekfac_n_subsets=50, n_queries=50, train_loss=3.23,
#     run_dir="runs/tail_bank_083_full", code_commit="5833a9b3", reusable="bank",
#     notes="MAIN RESULT: attributing only the last epoch makes pre-training scoreable (9x, "
#           "disjoint CIs) at the model's full loss. Tail-MAGIC was in progress, not recorded.")
# for frac, ms in [(0.25, 0.025), (0.5, 0.355), (0.6, 0.669), (0.75, 0.793),
#                  (0.896, 0.986), (0.95, 0.993), (0.99, 0.990)]:
#     add(OLMO2, run_id=f"olmo2_muon_16k_window{frac}", status="partial", n_docs=16000,
#         attr_window_frac=frac, metasmoothness=ms, reusable="ms_only",
#         notes="Window sweep, ms only. frac>0.833 loses doc coverage (bank would be invalid).")
# for n, st, ms, loss in [(4000, 188, 0.0095, 4.98), (8000, 375, 0.0177, 3.95),
#                         (32000, 1500, 0.0051, 3.09)]:
#     add(OLMO2, run_id=f"olmo2_muon_{n//1000}k_full", status="partial", n_docs=n, steps=st,
#         metasmoothness=ms, train_loss=loss, reusable="ms_only",
#         notes="Full-run attribution flat at ~0 across 1.9 nats of loss.")
# for knob, ms, loss, kw in [
#         ("opt_adamw", 0.647, 6.18, dict(optimizer="adamw", lr=8e-4)),
#         ("lr3e-3", 0.019, 2.69, dict(lr=3e-3)),
#         ("wd0", 0.003, 3.27, dict(weight_decay=0.0)),
#         ("epsroot1e-4", 0.004, 3.34, dict(eps_root=1e-4)),
#         ("bs64", 0.005, 4.31, dict(batch_size=64, num_epochs=3)),
#         ("bs256", 0.006, 1.34, dict(batch_size=256, num_epochs=12))]:
#     add(OLMO2, run_id=f"olmo2_muon_16k_{knob}", status="partial", n_docs=16000,
#         metasmoothness=ms, train_loss=loss, reusable="ms_only",
#         notes="opt_adamw's 0.647 is at loss 6.18 — unusable." if knob == "opt_adamw" else "",
#         **kw)

# =====================================================================================
# 3. FILL rows — measurements missing on configs whose artifacts already exist (cheap).
# =====================================================================================
# fill_* rows removed per D15 final ruling: their per-epoch banks are invalid
# (pre-venv) and deleted; the configs re-run fresh if the paper needs them.
for rid, bd in [(r[0], r[13]) for r in GRID if False]:
    add(GPT2_FT, run_id=f"fill_{rid}_magic", status="planned",
        n_docs=dict(GRID_N := {g[0]: g[1] for g in GRID})[rid],
        optimizer={g[0]: g[2] for g in GRID}[rid], lr={g[0]: g[3] for g in GRID}[rid],
        eps_root={g[0]: g[4] for g in GRID}[rid],
        batch_size={g[0]: g[5] for g in GRID}[rid],
        num_epochs={g[0]: g[6] for g in GRID}[rid],
        bank_dir=f"{PEREPOCH_RUNS}/{bd}/bank", source_doc="planned",
        notes=f"MAGIC per-query rollout for {rid}: the per-epoch bank exists, so only the "
              "metagradient run + validate --retrained_dir is needed. Budget one reverse pass "
              "per query. MUST run at the bank's world size (nproc 8; banks built on "
              "37d7b386): a nproc-4 retrain of the base on that same commit diverges 7.7e-3 "
              "— replay is only faithful at matched nproc. Code choice (current vs 37d7b386) "
              "needs a nproc-8 bit-gate when 8 GPUs are free.")
# EK-FAC (D7 canonical: damped_inverse 0.1, kfac+ev_correction, query_20) measured by
# scripts/ekfac_lds.py against each bank's validation.csv; scores at
# /mnt/ssd-2/lucia/s16k_<opt>/ekfac_scores, code commit 10874f93 (main-parent worktree).
EKFAC_FILL = {}  # results live on the sm_* parent rows (ANCHOR_FILL above)
# Anchor fill tickets removed with the invalid banks (D15 final ruling); the
# sm_ rows themselves are the re-run vehicles.
for opt in []:
    add(GPT2_FT, run_id=f"fill_sm_{opt}_eps1e17_16k_bs256_ms_ekfac",
        status="done",
        n_docs=16000, optimizer=opt, lr=2e-4, eps_root=1e-17, batch_size=256,
        grad_accum_steps=16, source_doc="planned", n_queries=20,
        bank_dir=f"/mnt/ssd-2/lucia/s16k_{opt}/merged",
        **EKFAC_FILL.get(opt, {}),
        notes=f"Work ticket complete: results are recorded on sm_{opt}_eps1e17_16k_bs256 "
              "(ms, EK-FAC, and MAGIC all measured for both optimizers).")

# =====================================================================================
# 4. PLANNED — one-factor deviations from the scaling_magic anchor (GPT-2, SmolLM2 16k,
#    bs256/ga16, ep2 = 125 steps, eps_root 1e-17, lr 2e-4). All planned rows configure
#    dropout_cfg=0.0 explicitly (policy: no inert-0.1 ambiguity in new runs) and keep
#    checkpoints (ckpt averaging + future windows need them).
# =====================================================================================
BASE17 = dict(GPT2_FT, n_docs=16000, lr=2e-4, eps_root=1e-17, batch_size=256,
              grad_accum_steps=16, num_epochs=2, status="planned", dropout_cfg=0.0,
              n_subsets=100, n_queries=20, source_doc="planned")

for bs in [16, 32, 64, 128]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_bs{bs}", batch_size=bs,
        grad_accum_steps=max(1, bs // 16),
        notes="Batch-size axis. bs256 measured (MAGIC 0.9333).")
for bs in [16, 32, 64, 128]:
    add(BASE17, run_id=f"plan_muon_eps1e17_16k_bs{bs}", optimizer="muon", batch_size=bs,
        grad_accum_steps=max(1, bs // 16),
        notes="D5: muon twin of the batch-size axis. bs256 measured (MAGIC 0.8470).")
# Measured results for completed clean-env banks (env = the pinned paper env,
# ENVIRONMENT.md; identity recorded per row in notes).
BANK_RESULTS = {
    # First 500-step bank to complete. Built as 12 shards across marisa-0 and
    # shivam2-0 (both A100, so D17 holds) with the tail, subsets 37-39, finished
    # on lotus-0 after marisa-0 wedged. merge_bank.py reports the full
    # 100 subsets x 20 queries = 2000 rows with no INCOMPLETE line.
    #
    # This is the first (LDS, filter-delta) pair we have had at 500 steps -- the
    # rest of the correlation sits at 125. Its EK-FAC filter delta is retraining
    # now on secret-ord-0; the MAGIC side needs MAGIC scores, which this row
    # does not have yet.
    "plan_adam_eps1e17_64k_bs256": dict(
        ekfac_lds=0.4336, ekfac_ci_lo=0.3826, ekfac_ci_hi=0.4796,
        ekfac_n_subsets=100,
    ),

    # Second gpt2-medium point, at bs32 rather than bs256 -- the config D11 was
    # meant to register, and an exact match to plan_adam_eps1e17_16k_bs32 (gpt2,
    # ms 0.9800, MAGIC 0.9201) with only the model swapped. ms came in at 0.7402,
    # LOWER than the bs256 run 0.8580, so lowering batch moves ms the wrong way on
    # this model just as it does on gpt2 (0.9930 bs256 -> 0.9800 bs32 -> 0.9133
    # bs16). Recovering ms on gpt2-medium is not a batch-size lever.
    # Bank and MAGIC scoring for this row are running on lotus-0; scoring alone is
    # ~47h at 8.84 s/step over 1000 steps x 20 queries, so MAGIC LDS lands well
    # after the bank does.
    "gpt2medium_16k_bs32": dict(
        status="planned",
        metasmoothness=0.7402,
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/gpt2medium_16k_bs32",
        notes="gpt2-medium at bs32, 1000 steps, lr 5e-05, world size 4 on A100 "
              "(lotus-0/maria-1). ms 0.7402 measured with fd_step 0.1, seed 0, "
              "total_movement_l1 68137.34."),
    # FIRST COMPLETED 2000-STEP BANK. 100/100 subsets, built by shards and merged
    # with scripts/merge_bank.py, scored against the row's own EK-FAC scores.
    #
    # 0.4146 [0.3719, 0.4541] sits inside the 0.404-0.473 band every other EK-FAC
    # row occupies, so EK-FAC flatness now extends to 2000 steps instead of
    # breaking there. notes/ekfac_floor.md stays open.
    #
    # I reported 0.1085 for this row from a 92/100 bank and it was wrong -- not
    # because the partial data was corrupt, but because dropping those same 8
    # subsets from the FINISHED file reproduces 0.1085 exactly. Eight subsets in a
    # hundred move this estimate from 0.11 to 0.41, and the 92-subset interval
    # [0.0476, 0.1655] was tight enough to look trustworthy. A partial bank is not
    # a noisy version of the finished one; it can be precisely wrong. See
    # notes/ekfac_2000step_collapse.md.
    #
    # MAGIC for this row is still scoring, so magic_lds stays empty.
    # SECOND COMPLETED 2000-STEP BANK, and the muon twin of the adam row below.
    # 100/100 subsets, sharded across A40 nodes, merged with merge_bank.py.
    #
    # 0.4281 [0.3833, 0.4695] -- inside the same 0.404-0.473 band as every other
    # EK-FAC row, and agreeing with its adam twin at 0.4146. So EK-FAC flatness
    # holds to 2000 steps for BOTH optimizers, and notes/ekfac_floor.md stays
    # open: nothing measured so far distinguishes robustness from a floor.
    #
    # The incomplete-bank trap fired again here and harder than the first time.
    # At 99/100 subsets this row read 0.1659 [0.1036, 0.2288]. ONE subset moved
    # it to 0.4281. The adam row did the same thing at 92/100 (0.1085) against
    # 0.4146 at 100. Two independent rows, both looking like a collapse, both
    # ordinary once complete -- and in both cases the partial interval was tight
    # enough to look trustworthy. Never read an LDS off a bank that
    # merge_bank.py reports INCOMPLETE.
    "plan_muon_eps1e17_32k_bs32": dict(
        ekfac_lds=0.4281, ekfac_ci_lo=0.3833, ekfac_ci_hi=0.4695,
        n_subsets=100, n_queries=20,
        reusable="bank+scores",
        run_dir="/mnt/ssd-1/lucia/paper_runs/experiments/plan_muon_eps1e17_32k_bs32",
        bank_dir="/mnt/ssd-1/lucia/paper_runs/experiments/plan_muon_eps1e17_32k_bs32/bank_from_filter",
        notes="Token axis at 2000 steps, muon, bs32, 32k docs, 2 epochs. WORLD "
              "SIZE 2, NVIDIA A40 -- base build on lucia-ord-0 with shards on "
              "allium-0, bellflower-0, shared-ord-0 and iris-0, all A40, so the "
              "bank is hardware-homogeneous under D17. Built via validate with "
              "method: lds and save_models: true against existing EK-FAC scores, "
              "skipping the unshardable MAGIC scoring stage. 100 subsets, 20/20 "
              "queries. Pairs with plan_adam_eps1e17_32k_bs32 (0.4146): the two "
              "2000-step rows agree, and both sit in the band the 250-step rows "
              "occupy. EK-FAC does not degrade with step count over this range. "
              "Subset 39 was lost when the ssd-1 quota killed its shard mid-write "
              "and was rebuilt separately; MAGIC scoring for this row is still "
              "running, so magic_lds is empty."),
    "plan_adam_eps1e17_32k_bs32": dict(
        ekfac_lds=0.4146, ekfac_ci_lo=0.3719, ekfac_ci_hi=0.4541,
        n_subsets=100, n_queries=20,
        reusable="bank+scores",
        run_dir="/mnt/ssd-1/lucia/paper_runs/experiments/plan_adam_eps1e17_32k_bs32",
        bank_dir="/mnt/ssd-1/lucia/paper_runs/experiments/plan_adam_eps1e17_32k_bs32/bank_from_filter",
        notes="Token axis at 2000 steps, bs32, 32k docs, 2 epochs. WORLD SIZE 2, "
              "NVIDIA A40 -- base build on secret-ord-0 with shards on allium-0, "
              "bellflower-0, lucia-ord-0 and shared-ord-0, all A40, so the bank is "
              "hardware-homogeneous under D17. Bank built via validate with "
              "method: lds and save_models: true against existing EK-FAC scores, "
              "which skips the unshardable MAGIC scoring stage. 100 subsets, 20/20 "
              "queries. THE FIRST LDS ABOVE 250 STEPS: 0.4146 [0.3719, 0.4541], "
              "indistinguishable from the 16k and 32k bs256 rows (0.4127-0.4730). "
              "EK-FAC does not degrade with step count over this range."),
    # ms at N=128,000 -- the largest measured, and the first point on the ladder
    # that Lucia wants carried to N=1M. 8000 steps at bs32, 2 epochs.
    #
    #     16k 0.9800   32k 0.9866   64k 0.9869   128k 0.9741
    #
    # Still high, and the first drift downward: 128k is 0.013 below 64k, where
    # 16k->32k->64k moved by less than 0.007 in total. One point is not a trend,
    # and the muon twin plus 256k and 512k are running, but it is the first sign
    # that ms is not simply flat in N.
    # ms at N=256,000, 16000 steps. The ladder now reads:
    #
    #   16k 0.9800   32k 0.9866   64k 0.9869   128k 0.9741   256k 0.9637
    #
    # ms rises slightly to 64k and then declines steadily: -0.013 to 128k, a
    # further -0.010 to 256k. It stays high in absolute terms -- 0.96 is nowhere
    # near a collapse -- but it is no longer flat, and the decline is monotone
    # over the last three points rather than noise. Extrapolating that slope
    # naively puts N=1M near 0.94, still high.
    "plan_adam_eps1e17_256k_bs32": dict(
        status="planned",
        metasmoothness=0.9637,
        run_dir="/mnt/ssd-1/lucia/paper_runs/experiments/plan_adam_eps1e17_256k_bs32",
        notes="ms 0.9637260437011719, fd_step 0.1, direction_seed 0, world size 2 "
              "on A40. 256k docs, bs32, 16000 steps, lr 5e-05. Training probe "
              "only, no bank -- a 100-model bank at this size is out of reach, "
              "which is why ms is the only measure that reaches here."),
    "plan_adam_eps1e17_128k_bs32": dict(
        status="planned",
        metasmoothness=0.9741,
        run_dir="/mnt/ssd-1/lucia/paper_runs/experiments/plan_adam_eps1e17_128k_bs32",
        notes="ms 0.9741322994232178 with fd_step 0.1, direction_seed 0, "
              "world size 2 on NVIDIA A100 (shivam2-0). 128k docs, bs32, 8000 "
              "steps, lr 5e-05. No bank: this is a training probe only, which is "
              "why it can run at a size where a 100-model bank is out of reach."),
    "gpt2medium_64k_bs32": dict(
        status="planned",
        metasmoothness=0.8580,
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/gpt2medium_64k_bs32",
        notes="gpt2-medium at 64k, bs32, 4000 steps, lr 2.5e-05, world size 4 on A100 "
              "(marisa-0). ms 0.8579838871955872 with fd_step 0.1, seed 0, "
              "total_movement_l1 64432.09. Matches the 16k bs256 gpt2-medium row "
              "(0.8580217, movement 44512.30) to four decimals across 4x the data."),
    "plan_adam_eps1e17_16k_gpt2-medium": dict(
        status="done",
        magic_lds=-0.0407, magic_ci_lo=-0.0980, magic_ci_hi=0.0155,
        # EK-FAC on the SAME bank reads 0.4189 [0.3707, 0.4645] -- squarely in the
        # 0.404-0.473 band it produces on every other row, while MAGIC is zero
        # here. Either EK-FAC is robust where MAGIC is not, or 0.42 is its floor.
        # Its filter delta decides which; see notes/ekfac_floor.md.
        ekfac_lds=0.4189, ekfac_ci_lo=0.3707, ekfac_ci_hi=0.4645,
        magic_n_queries=20, n_subsets=100, n_queries=20,
        # scripts/param_delta.py, step 124: rel_l2 0.00940
        delta_l1=324769.17, delta_l2=20.7219,
        # scripts/run_losses.py against the run's own final checkpoint
        train_loss=2.8529, heldout_loss=3.0019,
        reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_gpt2-medium",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_gpt2-medium",
        notes="Model-size axis, D11's registered scaling target: gpt2-medium (355M) "
              "at 16k, bs256, 125 steps -- the anchor's config with only the model "
              "swapped. WORLD SIZE 4, NVIDIA A100 (bank finished on shivam2-0; the "
              "60-80 slice died partway and 64-79 was re-run). "
              "MAGIC LDS IS ZERO: -0.0407 [-0.0980, 0.0155], an interval that "
              "contains zero, against 0.9411 for the same config on gpt2. So MAGIC "
              "attribution carries no signal at all on the larger model, while the "
              "model itself trains fine -- held-out 3.0019 against the gpt2 anchor's "
              "3.2363, the best of any row in the grid. This is the ms prediction "
              "landing: ms reads 0.8580 here, below the 0.95 boundary that separates "
              "the rows where MAGIC works from the rows where it collapses, and it "
              "has the LOWEST ms of any non-cut row. It is not alone below the "
              "boundary and the ordering is not clean: ms 0.8580 -> -0.04 here, "
              "0.9133 -> 0.1796 (adam bs16), 0.9037 -> 0.3020 (muon 4k), 0.9930 -> "
              "0.9411 (the anchor). So a lower ms goes with a lower LDS across this "
              "set, but 0.9037 beats 0.9133, and ms does not order the two rows "
              "between them correctly. "
              "Bank overlap resolved: the main run overran into the slice ranges and "
              "wrote subsets 0-63 while slices covered 40-59 and 60-62. The 460 "
              "duplicated (subset, query) pairs agree to 2.1e-06 on diff and are "
              "bit-identical on score_sum, so the main run's copies of 40-62 were "
              "dropped and the original kept as validation.csv.premerge. Subset 63 "
              "exists only in the main run and was kept."),
    "plan_adam_eps1e17_16k_ep4": dict(
        status="done",
        magic_lds=0.9534, magic_ci_lo=0.9458, magic_ci_hi=0.9594,
        # scripts/ekfac_lds.py, D7 config, A40 allium-0 nproc 2, scored after the
        # mirror fix cleared the stall. Same 100-subset bank as MAGIC.
        ekfac_lds=0.4730, ekfac_ci_lo=0.4310, ekfac_ci_hi=0.5110,
        magic_n_queries=20, n_subsets=100, n_queries=20,
        # scripts/param_delta.py against this run's own checkpoints/step_0.ckpt
        delta_l1=245405.56, delta_l2=25.1617,
        # scripts/run_losses.py against the run's own final checkpoint
        train_loss=3.0838, heldout_loss=3.2505,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_ep4",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_ep4",
        notes="D2: double epochs (250 steps, batch unchanged) -- isolates step count "
              "from batch size. lr 1e-4 from tuning.csv sweep_group tune_adamw_16k_ep4. "
              "nproc 2, NVIDIA A40, bellflower-0; retrains 30-100 sharded four ways on "
              "shared-ord-0, also A40, so the bank is hardware-homogeneous under D17. "
              "100 subsets, 20/20 queries, CI half-width 0.0068. HIGHEST MAGIC IN THE "
              "GRID: 0.9534 [0.9458, 0.9594] against the anchor 0.9411 [0.9326, 0.9477], "
              "intervals not overlapping. Doubling epochs at fixed batch RAISES "
              "attribution, and ms rises with it (0.9959 vs the anchor 0.9930), so on "
              "this axis smoothness and attributability move together. Read against "
              "bs512 (0.9233, 63 steps, the uncontrolled double batch): more steps at "
              "fixed batch helps, more batch at fixed data does not. The main process "
              "overran its stop and recomputed subsets 30-44, which slice_30_48 also "
              "covered; the two copies agree to 3.8e-06 on diff, the main copies were "
              "dropped and the original kept as validation.csv.premerge."),
    "plan_muon_eps1e17_32k_bs256": dict(
        status="done",
        magic_lds=0.8715, magic_ci_lo=0.8575, magic_ci_hi=0.8835,
        # scripts/ekfac_lds.py, D7 config, A40 allium-0 nproc 2, same 100-subset bank
        ekfac_lds=0.4044, ekfac_ci_lo=0.3584, ekfac_ci_hi=0.4483,
        magic_n_queries=20, n_subsets=100, n_queries=20,
        # scripts/param_delta.py, step 249: rel_l2 0.01825
        delta_l1=281856.91, delta_l2=31.9472,
        # scripts/run_losses.py against this run own final checkpoint; the
        # heldout it measures reproduces the independently recorded 3.2372.
        train_loss=3.1013, heldout_loss=3.2372,
        reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_32k_bs256",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_32k_bs256",
        notes="Token axis, muon at N=32k, bs256 (250 steps). lr 2e-4. WORLD SIZE 2 -- "
              "all six slice configs agree on nproc_per_node 2, so the bank is "
              "bit-exactly reusable at that world size. NVIDIA A40 (claimed by iris-0; "
              "the final subset 74 was retrained on secret-ord-0, also A40). Per-slice "
              "host is not written to the slice logs, so A40 rests on the claim record "
              "rather than a per-subset check -- worth re-reading if this row ever "
              "disagrees with its neighbours, since D17 puts a mixed-hardware bank "
              "0.055 low. COMPLETES THE MUON TOKEN AXIS: 0.3020 at 4k, 0.7712 at 8k, "
              "0.8379 at the 16k anchor, 0.8715 here. Monotone and still climbing, "
              "closing on the adamw arm (0.9295 / 0.9163 / 0.9411 / 0.9529) but 0.081 "
              "below it at 32k -- the two optimisers have not converged over an 8x "
              "token range. ms 0.9948, which is above the 0.95 boundary while LDS is "
              "well under the adamw arm, so ms and LDS disagree here."),
    "plan_adam_eps1e17_32k_bs256": dict(
        status="done",
        magic_lds=0.9529, magic_ci_lo=0.9470, magic_ci_hi=0.9577,
        # scripts/ekfac_lds.py, D7 config, A40 allium-0 nproc 2, same 100-subset bank
        ekfac_lds=0.4127, ekfac_ci_lo=0.3678, ekfac_ci_hi=0.4554,
        magic_n_queries=20, n_subsets=100, n_queries=20,
        delta_l1=353172.61, delta_l2=36.5131,
        train_loss=3.1074, heldout_loss=3.2363,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_32k_bs256",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_32k_bs256",
        notes="Token axis, adamw at N=32k, bs256 (250 steps). lr 2e-4. nproc 2, "
              "NVIDIA A40, secret-ord-0; retrains 25-100 sharded three ways on "
              "allium-0, also A40. 100 subsets, 20/20 queries, CI half-width 0.0054 -- "
              "the tightest row in the grid. EXTENDS THE ADAMW TOKEN AXIS: 0.9295 at "
              "4k, 0.9163 at 8k, 0.9411 at the 16k anchor, 0.9529 here. The axis is "
              "flat to mildly rising over an 8x range in tokens, against the muon arm "
              "which climbs steeply (0.3020 at 4k to 0.7712 at 8k). ms 0.9937. Main "
              "overran into the slice range on subsets 25-39; copies agree to 3.3e-06 "
              "on diff, main copies dropped, original kept as validation.csv.premerge."),
    "plan_muon_eps1e17_16k_bs16": dict(
        status="done",
        magic_lds=0.8850, magic_ci_lo=0.8708, magic_ci_hi=0.8967,
        magic_n_queries=20,
        # scripts/ekfac_lds.py, D7 config, scored on A100 (lotus-0) nproc 2 after
        # the mirror fix cleared the EK-FAC stall. Same 100-subset bank as MAGIC.
        ekfac_lds=0.4276, ekfac_ci_lo=0.3815, ekfac_ci_hi=0.4686,
        # scripts/param_delta.py against this run's own checkpoints/step_0.ckpt
        delta_l1=226800.32, delta_l2=26.1277,
        # scripts/run_losses.py against the run's own final checkpoint
        train_loss=3.0518, heldout_loss=3.2440,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_16k_bs16",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_16k_bs16",
        notes="Batch-size axis, muon at bs16. Bank built by lotus-0 (A100), tail "
              "sharded 92-95/95-97/97-100; validation_merged.csv is 100 subsets "
              "exactly once. "
              "RESOLVES THE SECOND PRE-REGISTERED PREDICTION. Its ms (0.9939) was "
              "measured while the bank was at 73/100 and put it in the HEALTHY "
              "band, against adam bs16 at ms 0.9133 (collapsed, MAGIC 0.1796). The "
              "prediction recorded before this bank finished was MAGIC above ~0.77; "
              "measured 0.8850. "
              "THE LARGEST OPTIMIZER GAP IN THE GRID BY FAR: paired adamw-muon = "
              "-0.7054 [-0.7995, -0.6014], and adamw wins 0/20 queries -- the only "
              "unanimous loss anywhere. For scale, the anchor pair is +0.1032 and "
              "the next largest is +0.1451 at 8k. "
              "The pair is hardware-matched (both arms lotus-0/A100), so D17 does "
              "not apply to this contrast. "
              "Read with the bs16 lr caveat: at bs16 the tuned lr is 5e-5, and the "
              "controlled probe shows adamw at bs16 scores ms 0.5127 at the anchor "
              "lr, so adamw's collapse here is the batch size with its low lr only "
              "partly rescuing it. Muon is evidently not harmed the same way."),
    "plan_adam_eps1e17_16k_bs512": dict(
        status="done",
        magic_lds=0.9233, magic_ci_lo=0.9115, magic_ci_hi=0.9331,
        magic_n_queries=20,
        ekfac_lds=0.4142, ekfac_ci_lo=0.366, ekfac_ci_hi=0.4587,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-1/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_bs512",
        bank_dir="/mnt/ssd-1/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_bs512",
        notes="Batch-size axis, adamw at bs512 (63 steps, D2 uncontrolled double "
              "batch). nproc 4, NVIDIA A100-80GB, maria-1; single node, no shards, "
              "so internally homogeneous. 100 models, 20/20 queries, half-width "
              "0.0108. Sits just below the anchor (0.9411) and bs128 (0.9441). "
              "D17: A100, unlike the A40 anchor -- see the batch-axis hardware note "
              "on bs16. Tuned lr 2e-4."),
    "plan_adam_eps1e17_16k_bs16": dict(
        status="done",
        magic_lds=0.1796, magic_ci_lo=0.1249, magic_ci_hi=0.2324,
        magic_n_queries=20,
        ekfac_lds=0.3872, ekfac_ci_lo=0.3424, ekfac_ci_hi=0.4305,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_bs16",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_bs16",
        notes="Batch-size axis, adamw at bs16. Bank built by lotus-0 (NVIDIA "
              "A100-80GB), which runs independently; scored here because the bank "
              "was complete and unrecorded. 100 models, 20/20 queries. "
              "ATTRIBUTION LARGELY COLLAPSES AT THE SMALL-BATCH END: 0.1796 "
              "[0.1249, 0.2324], against 0.9201 at bs32 and 0.9411 at the bs256 "
              "anchor. Second-largest collapse in the grid after scale0.25 (0.0456). "
              "D17 CAVEAT ON THE AXIS AS A WHOLE: the batch sweep now interleaves "
              "hardware -- bs16 A100 (lotus-0), bs32 A40, bs64 A40, bs128 A100, "
              "bs256 A40, bs512 A100 -- and GPU type is worth ~0.055 on its own. "
              "That is far too small to explain 0.18 vs 0.92, so the small-batch "
              "collapse is real, but the axis should not be read as a clean curve "
              "until the arms are on one GPU type. The bs16 PAIR is safe: its muon "
              "partner is also lotus-0/A100. Tuned lr 5e-5. "
              "SECOND INVERSION IN THE GRID: EK-FAC scores 0.3872 [0.3424, 0.4305] "
              "on this bank, well ABOVE MAGIC 0.1796 [0.1249, 0.2324]. And EK-FAC "
              "barely notices the small batch at all -- 0.3872 against its usual "
              "0.42-0.46 elsewhere -- while MAGIC drops from 0.94 to 0.18. "
              "Together with scale0.25 (EK-FAC 0.1733 over MAGIC 0.0456) this is a "
              "consistent pattern rather than one oddity: MAGIC is far better in "
              "the normal regime (~0.94 vs ~0.42) but fails hard in specific "
              "regimes, and in exactly those regimes EK-FAC degrades gracefully "
              "and overtakes it. That is a robustness claim, distinct from the "
              "accuracy claim, and the two inversions are the evidence for it."),
    "plan_adam_eps1e17_16k_wd0.0": dict(
        status="done",
        magic_lds=0.941, magic_ci_lo=0.9326, magic_ci_hi=0.9476,
        magic_n_queries=20,
        ekfac_lds=0.4235, ekfac_ci_lo=0.375, ekfac_ci_hi=0.4684,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_wd0.0",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_wd0.0",
        notes="Weight-decay axis, wd=0.0. nproc 2, NVIDIA A40, allium-0. 100 models, "
              "20/20 queries. WEIGHT DECAY IS A NULL, as pre-registered: "
              "wd 0.0 -> 0.9410, wd 0.01 (the anchor) -> 0.9411, wd 0.1 -> "
              "0.9414. All three within 0.0004 of each other, against CI "
              "half-widths of ~0.008. Tuned lr 2e-4."),
    "plan_adam_eps1e17_16k_wd0.1": dict(
        status="done",
        magic_lds=0.9414, magic_ci_lo=0.933, magic_ci_hi=0.9482,
        magic_n_queries=20,
        ekfac_lds=0.4244, ekfac_ci_lo=0.3763, ekfac_ci_hi=0.4689,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_wd0.1",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_wd0.1",
        notes="Weight-decay axis, wd=0.1. nproc 2, NVIDIA A40, iris-0. 100 models, "
              "20/20 queries. See wd0.0: the three weight-decay points span "
              "0.0004 and the axis is a null. Tuned lr 2e-4."),
    "plan_adam_eps1e17_16k_clip1.0": dict(
        status="done",
        magic_lds=0.8982, magic_ci_lo=0.8757, magic_ci_hi=0.9169,
        magic_n_queries=20,
        ekfac_lds=0.4176, ekfac_ci_lo=0.3695, ekfac_ci_hi=0.4622,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_clip1.0",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_clip1.0",
        notes="Grad-clip axis, max_grad_norm=1.0. nproc 2, NVIDIA A40, allium-0. 100 "
              "models, 20/20 queries. Drops attribution from the anchor 0.9411 "
              "to 0.8982; the intervals do not overlap (anchor [0.9326, 0.9477] "
              "vs [0.8757, 0.9169]), so clipping costs a real ~0.043 -- about "
              "the size of the bs32 optimizer effect. Tuned lr 2e-4."),
    "plan_adam_eps1e17_16k_scale0.5": dict(
        status="done",
        magic_lds=0.9448, magic_ci_lo=0.9353, magic_ci_hi=0.9521,
        magic_n_queries=20,
        ekfac_lds=0.176, ekfac_ci_lo=0.1092, ekfac_ci_hi=0.2379,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_scale0.5",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_scale0.5",
        notes="Logit-scale axis, logit_scale=0.5. nproc 2, NVIDIA A40, secret-ord-0. "
              "Ran against bergson feat/logit-scale (PR #433), NOT the pinned "
              "-429 worktree, which has no logit_scale field. 100 models, 20/20 "
              "queries. NO EFFECT: 0.9448 vs the anchor 0.9411, intervals "
              "overlapping. Halving the logits is harmless TO MAGIC. Tuned lr 2e-4. "
              "BUT NOT TO EK-FAC: 0.1760 [0.1092, 0.2379] here, against 0.4253 at "
              "the scale-1.0 anchor. So the two scorers fail at different points "
              "on this axis. EK-FAC has already lost more than half its "
              "correlation at scale 0.5, where MAGIC is untouched (0.9448 vs "
              "0.9411); at scale 0.25 EK-FAC is essentially unchanged again "
              "(0.1733) while MAGIC collapses to 0.0456. EK-FAC degrades EARLIER "
              "and MAGIC degrades LATER BUT HARDER. "
              "This also corrects an over-broad reading of the EK-FAC numbers: "
              "EK-FAC is flat near 0.42 across weight decay, gradient clipping, "
              "batch size and token count, but it is NOT flat across logit scale, "
              "so flatness is an axis-specific observation and not a property of "
              "the scorer."),
    "plan_adam_eps1e17_16k_scale0.25": dict(
        status="done",
        magic_lds=0.0456, magic_ci_lo=0.0038, magic_ci_hi=0.0861,
        magic_n_queries=20,
        ekfac_lds=0.1733, ekfac_ci_lo=0.1115, ekfac_ci_hi=0.232,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_scale0.25",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_scale0.25",
        notes="Logit-scale axis, logit_scale=0.25. nproc 2, NVIDIA A40, allium-0. Ran "
              "against bergson feat/logit-scale (PR #433). 100 models, 20/20 "
              "queries. ATTRIBUTION COLLAPSES: 0.0456 [0.0038, 0.0861], against "
              "0.9448 at scale 0.5 and 0.9411 at the anchor. The per-query "
              "Spearmans scatter around zero (-0.106 to +0.285), so this is a "
              "uniform loss of signal, not one broken query. "
              "THE GROUND TRUTH IS HEALTHY: the diff std is 2.4e-3, LARGER than "
              "the anchor 1.39e-3, so removing data still moves the model a lot "
              "-- MAGIC simply stops predicting which data. "
              "CAVEAT, do not report the scale effect alone: this row is the "
              "only one in the grid at tuned lr 8e-4 (every other adamw row is "
              "2e-4 or lower), because the sweep picked 8e-4 for this config. "
              "Logit scale and learning rate are therefore not separable from "
              "this single row; a scale-0.25 run at 2e-4, or a scale-1.0 run at "
              "8e-4, would separate them. "
              "FIRST INVERSION IN THE GRID: EK-FAC scores 0.1733 [0.1115, 0.2320] "
              "on this same bank, ABOVE MAGIC 0.0456 [0.0038, 0.0861] with "
              "non-overlapping intervals. Everywhere else MAGIC leads EK-FAC by "
              "~0.5. Logit scaling degrades both scorers but hits MAGIC far "
              "harder (-0.895 from its anchor, against about -0.25 for EK-FAC), "
              "which is what you would expect if the damage is to the gradient "
              "trajectory MAGIC linearises around rather than to the data. "
              "CORRECTED 2026-08-24: that reading attributed the damage to logit "
              "scaling, but this row also carries a 4x learning rate (tuned 8e-4 "
              "vs the anchor 2e-4). An ms probe at logit_scale 0.25 with lr held "
              "at 2e-4 scores 0.9812, i.e. healthy, so the pathology on this row "
              "is the learning rate and not the logit scale. scale0.5, which runs "
              "at the anchor lr, has MAGIC 0.9448 -- unharmed. The EK-FAC drop at "
              "scale 0.5 (0.4253 -> 0.1760) is NOT explained by this, since that "
              "row holds lr fixed, and remains a genuine logit-scale effect on "
              "EK-FAC alone."),
    "plan_adam_eps1e17_16k_bs128": dict(
        status="done",
        magic_lds=0.9441, magic_ci_lo=0.9334, magic_ci_hi=0.9523,
        magic_n_queries=20,
        ekfac_lds=0.4551, ekfac_ci_lo=0.4085, ekfac_ci_hi=0.4973,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_bs128",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_bs128",
        notes="Batch-size axis, adamw at bs128. nproc 4, NVIDIA A100-80GB, marisa-0 -- "
              "the ONLY adamw batch-size point not measured on A40. 100 models, "
              "20/20 queries. "
              "D17 CONFOUND, read before using the pair: its partner "
              "plan_muon_eps1e17_16k_bs128 (0.8480) was measured on A40, and "
              "GPU type demonstrably changes the retrained models (6.9e-4 vs "
              "2.5e-7 within-hardware, against a 1.1e-3 signal). The apparent "
              "+0.096 optimizer gap here is confounded with hardware and must "
              "not be quoted as an optimizer effect until one arm is re-run. "
              "The row is internally valid. Tuned lr 1e-4."),
    "sm_adamw_eps1e17_16k_bs256": dict(
        status="done",
        magic_lds=0.9411, magic_ci_lo=0.9326, magic_ci_hi=0.9477,
        magic_n_queries=20,
        ekfac_lds=0.4253, ekfac_ci_lo=0.3772, ekfac_ci_hi=0.4697,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/sm_adamw_eps1e17_16k_bs256",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/sm_adamw_eps1e17_16k_bs256",
        notes="THE ADAMW ANCHOR. nproc 2, NVIDIA A40, bellflower-0; pinned venv. "
              "100 models, 20/20 queries, CI half-width 0.0076 -- the tightest row "
              "in the grid. Per-query Spearman 0.90-0.99. "
              "HARDWARE: every retrain ran on A40. Two A100 slices (subsets 40-70 "
              "and 70-100) were computed during sharding and are QUARANTINED as "
              "validation_*.csv.a100, NOT merged -- see the GPU-type finding below. "
              "The A40 main process completed all 100 subsets by itself, so no "
              "recomputation was needed. "
              "EK-FAC 0.4253 [0.3772, 0.4697], and this is a REPRODUCTION: "
              "ANCHOR_FILL already recorded 0.4251 [0.3772, 0.4693] for the adamw "
              "anchor, measured on a different bergson commit (10874f93) against "
              "the separate s16k_adamw assets. Fresh bank, pinned env 79c08dce, "
              "same number to 2e-4. "
              "Contrast MAGIC on the same comparison: the anchor was previously "
              "quoted as 0.9333 (see the bs256/16k references in other row notes) and measures 0.9411 here, a shift of 0.008 -- just outside this row half-width of 0.0076. EK-FAC depends only on the final model, while MAGIC depends on the whole training trajectory, which is rebuilt per bank; the asymmetry in how well each reproduces across banks is expected, and worth stating explicitly for D15."),
    "sm_muon_eps1e17_16k_bs256": dict(
        status="done",
        magic_lds=0.8379, magic_ci_lo=0.8205, magic_ci_hi=0.8519,
        magic_n_queries=20,
        ekfac_lds=0.4237, ekfac_ci_lo=0.3814, ekfac_ci_hi=0.4616,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/sm_muon_eps1e17_16k_bs256",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/sm_muon_eps1e17_16k_bs256",
        notes="THE MUON ANCHOR. nproc 2, NVIDIA A40; pinned venv. 100 models, "
              "20/20 queries, CI half-width 0.0157. validation_merged.csv is "
              "subsets 0-17 from the original bellflower-0 run plus 18-45/45-73/ "
              "73-100 from three A40 slices. "
              "CLOSES THE ANCHOR PAIR, the reference contrast for the whole grid: "
              "paired adamw-muon = +0.1032 [+0.0851, +0.1213], half-width 0.0181, "
              "and adamw wins 20/20 queries -- the only unanimous contrast measured. "
              "GPU-TYPE FINDING (2026-08-23): this bank was accidentally sharded "
              "across A40 and A100 nodes, and the two hardware types do NOT produce "
              "the same retrained models. Where both sets ran on A40 they agree to "
              "2.5e-7 (matching the 8k shard boundary, i.e. retraining is "
              "deterministic on identical hardware); across A40 vs A100 they differ "
              "by 6.9e-4 mean / 2.1e-3 max, against a within-query spread of `diff` "
              "of only 1.1e-3 -- the disagreement is 43% of the signal LDS ranks. "
              "The consequence is not subtle: scoring this bank from the mixed set "
              "gives 0.7828, from the homogeneous A40 set 0.8379, a gap of 0.055 "
              "that is LARGER than most optimizer effects in the grid. A bank's "
              "retrains must all run on one GPU type. "
              "EK-FAC 0.4237 [0.3814, 0.4616], reproducing the 0.4285 in ANCHOR_FILL (older code 10874f93, separate s16k_muon assets) to 0.005 -- the adamw anchor reproduced to 0.0002. "
              "THE ANCHOR PAIR IS THE FOURTH AND CLEANEST SCORER CONTRAST: MAGIC paired adamw-muon = +0.1032 [+0.0851, +0.1213] with 20/20 query wins, while EK-FAC on the SAME two banks gives +0.0016 [-0.0248, +0.0258] with 12/20 -- a coin flip. Four pairs now: bs32, bs64, 8k and the anchor. MAGIC produces effects of +0.145, +0.103, +0.046 and -0.088; EK-FAC returns -0.001, +0.002, -0.004 and +0.002, never distinguishable from zero. "
              "SCALE CAVEAT (added after review, and it corrects an earlier version "
              "of this note): do NOT read this as EK-FAC being UNABLE to resolve "
              "the contrast. The scorers sit on different scales -- across the 19 "
              "measured rows MAGIC spans 0.046-0.945 and EK-FAC only 0.173-0.459, "
              "about a third the dynamic range -- so a contrast worth +0.10 in "
              "MAGIC units should not be expected at +0.10 in EK-FAC units. The "
              "earlier not-a-power-problem argument compared CI widths, which is "
              "estimate precision, and ignored this. Simple compression does not "
              "explain it either: on the logit-scale axis EK-FAC moves 0.25 "
              "(0.4253 -> 0.1760) where MAGIC does not move at all (0.9411 -> "
              "0.9448), so the two are not related by any monotone rescaling and "
              "cross-scorer effect sizes are not directly comparable in either "
              "direction. What the data supports is the OBSERVATION, not a "
              "mechanism: on identical ground truth MAGIC separates the optimizers "
              "in 4/4 pairs and EK-FAC does not, which flips the conclusion a "
              "reader would draw about which optimizer is more attributable. To "
              "settle the mechanism, normalise each scorer contrast by that "
              "scorer spread across the grid before comparing."),
    "plan_muon_eps1e17_16k_bs64": dict(
        status="done",
        magic_lds=0.8690, magic_ci_lo=0.8514, magic_ci_hi=0.8837,
        magic_n_queries=20,
        ekfac_lds=0.4284, ekfac_ci_lo=0.3868, ekfac_ci_hi=0.4646,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_16k_bs64",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_16k_bs64",
        notes="Batch-size axis, muon at bs64 (500 steps). nproc 2, NVIDIA A40, "
              "allium-0; pinned venv. 100 models, 20/20 queries. Half-width 0.0162. "
              "EK-FAC on the same bank: 0.4284 [0.3868, 0.4646]. "
              "SCORER FINDING (both bs32 and bs64 pairs, paired over queries, "
              "scripts/ekfac_paired.py): MAGIC resolves the optimizer contrast "
              "and EK-FAC does not. MAGIC bs32 +0.0464 [+0.0169, +0.0724] 17/20 "
              "wins; MAGIC bs64 -0.0879 [-0.1406, -0.0380] 7/20 -- two "
              "significant effects of OPPOSITE sign. EK-FAC bs32 +0.0019 "
              "[-0.0274, +0.0321] 9/20; EK-FAC bs64 -0.0045 [-0.0359, +0.0270] "
              "8/20 -- both straddle zero at coin-flip query wins. This is not "
              "a power problem: the EK-FAC half-widths (~0.030) match the MAGIC "
              "bs32 half-width (0.028), so EK-FAC has the resolution to see a "
              "+0.046 effect and does not. EK-FAC does track BATCH size "
              "(~0.457 at bs32 vs ~0.426 at bs64 in both arms), so it is "
              "responding to something -- just not the optimizer. "
              "CLOSES THE bs64 PAIR, and it REVERSES: paired adamw-muon = -0.0879 "
              "[-0.1406, -0.0380], adamw wins only 7/20 queries. This is the only "
              "negative optimizer gap in the grid -- everywhere else adamw leads "
              "(+0.6275 at 4k, +0.1451 at 8k, +0.0464 at bs32). "
              "Muon at bs64 is unremarkable (0.8690, between its bs32 0.8737 and "
              "bs128 0.8480); it is plan_adam_eps1e17_16k_bs64 that is anomalously "
              "LOW at 0.7811 with the widest per-query spread in the grid (0.53-0.97). "
              "The reversal is therefore evidence the adamw bs64 row is the outlier, "
              "not evidence of a batch-size effect on the optimizer contrast. Suggest "
              "a repeat of the adamw bs64 row before any batch-size claim. "
              "Tuned lr 1e-4."),
    "plan_muon_eps1e17_16k_bs32": dict(
        status="done",
        magic_lds=0.8737, magic_ci_lo=0.8574, magic_ci_hi=0.8869,
        magic_n_queries=20,
        ekfac_lds=0.4567, ekfac_ci_lo=0.4161, ekfac_ci_hi=0.4918,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_16k_bs32",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_16k_bs32",
        notes="Batch-size axis, muon at bs32 (1000 steps). nproc 2, NVIDIA A40, "
              "pinned venv. FIRST SHARDED BANK: subsets 0-21 from the original run, "
              "22-48/48-74/74-100 from three slices on lucia-ord-0, bellflower-0 and "
              "secret-ord-0; merged by magic_lds.py into validation_merged.csv, which "
              "asserts each subset appears exactly once. Sharding cut it from ~19 h to "
              "~6 h. CI half-width 0.0148. "
              "CLOSES THE bs32 OPTIMIZER PAIR: paired adamw-muon = +0.0464 "
              "[+0.0169, +0.0724], 17/20 query wins -- a real but small gap, and the "
              "narrowest optimizer effect measured anywhere in the grid so far. "
              "Tuned lr 5e-5. "
              "EK-FAC measured 2026-08-23 on the same bank: 0.4567 "
              "[0.4161, 0.4918], half-width 0.038, 100/100 subsets. "
              "READ THIS NEXT TO adam bs32 EK-FAC (0.4586): the two optimizers "
              "are indistinguishable under EK-FAC (0.4586 vs 0.4567, a gap of "
              "0.002 against half-widths of ~0.04), while MAGIC separates them "
              "on the SAME two banks (0.9201 vs 0.8737, paired +0.0464 with "
              "17/20 query wins). EK-FAC appears blind to the optimizer "
              "contrast this paper is about; a second pair should confirm "
              "before that is claimed. "
              "First value here was 0.4512 on 22 subsets -- ekfac_lds.py was "
              "reading the pre-shard validation.csv prefix rather than "
              "validation_merged.csv; fixed, and the 4k pair is unaffected "
              "because neither 4k bank was sharded."),
    "plan_adam_eps1e17_16k_bs32": dict(
        status="done",
        magic_lds=0.9201, magic_ci_lo=0.9088, magic_ci_hi=0.9296,
        magic_n_queries=20,
        ekfac_lds=0.4586, ekfac_ci_lo=0.4194, ekfac_ci_hi=0.4934,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_bs32",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_bs32",
        notes="Batch-size axis, adamw at bs32 (1000 steps). nproc 4, NVIDIA A40, "
              "lucia-ord-0; pinned venv. 100 models, 20/20 queries. CI half-width "
              "0.0104, the tightest adamw row so far; per-query Spearman 0.77-0.97. "
              "NON-MONOTONE against bs64 (0.7811, half-width 0.0512): attribution is "
              "HIGHER at the smaller batch, and the intervals do not overlap. bs64 is "
              "the outlier of the two -- low value and by far the widest per-query "
              "spread (0.53-0.97) of any row measured. Do not read a batch-size trend "
              "until bs128 and the re-measured bs256 anchor land. Tuned lr 5e-5. "
              "EK-FAC measured 2026-08-23 on the SAME bank (reuse rule 1; D7 "
              "config inherited from the accepted 4k template): 0.4586 "
              "[0.4194, 0.4934], half-width 0.037, 100/100 subsets, 20 queries, "
              "2216 s on marisa-0. MAGIC beats EK-FAC by 0.46 on identical "
              "ground truth -- the largest scorer gap measured so far, and far "
              "outside either interval. The EK-FAC per-query spread (0.23-0.65) "
              "is also much wider than the MAGIC spread on this row (0.77-0.97)."),
    "plan_adam_eps1e17_8k_bs256": dict(
        status="done",
        magic_lds=0.9163, magic_ci_lo=0.9013, magic_ci_hi=0.9280,
        magic_n_queries=20,
        ekfac_lds=0.3869, ekfac_ci_lo=0.3441, ekfac_ci_hi=0.4245,
        ekfac_n_subsets=100,
        code_commit="3c66bb51", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_8k_bs256",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_8k_bs256",
        notes="Token axis, adamw at N=8k. nproc 1, A100, lotus-0; pinned venv. "
              "Training and scoring at 3c66bb51, retrains at 5b03b7b1 (subset_start "
              "worktree). 100 models, 20/20 queries. Retrains sharded: subsets 0-86 "
              "from the main process, 87-99 from a slice; the main overran its "
              "intended stop at 72 so 72-86 were computed twice, the duplicates agree "
              "exactly (score_sum delta 0, diff delta < 3e-6) and validation.csv holds "
              "each subset once (merge by bellflower-0, original kept as "
              "validation.csv.premerge). CI half-width 0.0134. Token axis for adamw is "
              "flat: 0.9295 at 4k, 0.9163 at 8k, intervals nearly overlapping. "
              "Tuned lr 2e-4."),
    "plan_muon_eps1e17_8k_bs256": dict(
        status="done",
        magic_lds=0.7712, magic_ci_lo=0.7477, magic_ci_hi=0.7904,
        magic_n_queries=20,
        ekfac_lds=0.3881, ekfac_ci_lo=0.3403, ekfac_ci_hi=0.4333,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_8k_bs256",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_8k_bs256",
        notes="Token axis, muon at N=8k. nproc 2, NVIDIA A40, bellflower-0; pinned "
              "venv (python 3.11.15 / torch 2.13.0+cu126 / nccl 2.29.3). 100 models, "
              "20/20 queries. CI half-width 0.0214, inside the D6 threshold of 0.06. "
              "COMPLETES THE 8k OPTIMIZER PAIR against plan_adam_eps1e17_8k_bs256 "
              "(0.9163): unpaired difference adamw-muon = +0.145, far smaller than the "
              "+0.6275 measured at 4k. The muon token axis rises steeply, 0.3020 at 4k "
              "to 0.7712 at 8k, while adamw is flat (0.9295 to 0.9163) -- so the "
              "optimizer gap narrows with N rather than being constant. Tuned lr 2e-4."),
    "plan_muon_eps1e17_16k_bs128": dict(
        status="done",
        magic_lds=0.8480, magic_ci_lo=0.8307, magic_ci_hi=0.8620,
        magic_n_queries=20,
        ekfac_lds=0.4635, ekfac_ci_lo=0.4233, ekfac_ci_hi=0.4995,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_16k_bs128",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_16k_bs128",
        notes="Batch-size axis, muon at bs128. First muon row completed by the A40 "
              "fleet. nproc 4, NVIDIA A40, secret-ord-0; pinned venv (python 3.11.15 / "
              "torch 2.13.0+cu126 / nccl 2.29.3 / triton 3.7.1 / transformers 5.15.1 / "
              "datasets 5.0.1). 100 models, 20/20 queries, 2001-row validation.csv. "
              "CI half-width 0.0157, well inside the D6 threshold of 0.06 and much "
              "tighter than the adamw rows measured so far -- per-query Spearman spans "
              "only 0.78-0.97. Tuned lr 1e-4."),
    "plan_adam_eps1e17_16k_bs64": dict(
        status="done",
        magic_lds=0.7811, magic_ci_lo=0.7272, magic_ci_hi=0.8295,
        magic_n_queries=20,
        ekfac_lds=0.4239, ekfac_ci_lo=0.3854, ekfac_ci_hi=0.4593,
        ekfac_n_subsets=100,
        code_commit="79c08dce", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_bs64",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_16k_bs64",
        notes="Batch-size axis, bs64. First completed row of the A40 fleet and the "
              "first bank built entirely in the pinned venv. nproc 4, ga 1, "
              "NVIDIA A40 (47.5 GB), allium-0; env python 3.11.15 / torch 2.13.0+cu126 "
              "/ nccl 2.29.3 / triton 3.7.1 / transformers 5.15.1 / datasets 5.0.1. "
              "100 models, 20/20 queries, 2001-row validation.csv. "
              "CI half-width 0.051, inside the D6 threshold of 0.06 (raised from "
              "0.025 on 2026-08-22); reportable as measured. Per-query Spearman is "
              "widely spread (0.53 to 0.97), so a query_50 re-score would tighten it "
              "-- registered as future work, not required. "
              "Tuned lr 1e-4 per the completed sweep group."),
    "plan_adam_eps1e17_4k_bs256": dict(
        status="done", metasmoothness=0.9946, ms_direction_seed=0, ms_fd_step=0.1,
        magic_lds=0.9295, magic_ci_lo=0.9195, magic_ci_hi=0.9381,
        magic_n_queries=20, ekfac_lds=0.3975, ekfac_ci_lo=0.3521, ekfac_ci_hi=0.4391,
        ekfac_n_subsets=100, code_commit="3c66bb51", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_4k_bs256",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_adam_eps1e17_4k_bs256",
        notes="First clean-env bank of the campaign: pinned paper env, tuned lr 1e-4, "
              "nproc 2, A100-SXM4-80GB, lotus-0. 101 models, 20/20 queries. "
              "ms 0.9946 (movement_l1 7577). ALL THREE METRICS MEASURED, all clean-env."),
    "plan_muon_eps1e17_4k_bs256": dict(
        status="done", metasmoothness=0.9037, ms_direction_seed=0, ms_fd_step=0.1,
        magic_lds=0.3020, magic_ci_lo=0.2537, magic_ci_hi=0.3487,
        magic_n_queries=20, ekfac_lds=0.3031, ekfac_ci_lo=0.2545, ekfac_ci_hi=0.3468,
        ekfac_n_subsets=100, code_commit="3c66bb51", reusable="bank+scores",
        run_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_4k_bs256",
        bank_dir="/mnt/ssd-2/lucia/paper_runs/experiments/plan_muon_eps1e17_4k_bs256",
        notes="Clean-env bank: pinned paper env, tuned lr 4e-4, nproc 2, A100, lotus-0. "
              "Paired vs adamw 4k: adamw-muon = +0.6275 [+0.5033, +0.7517], 20/20 query "
              "wins - the optimizer gap is ~7x the anchor's +0.086 at this N. EK-FAC "
              "0.3031 is statistically identical to MAGIC 0.3020 on this bank - at "
              "muon-4k the methods agree. ms 0.9037 (movement_l1 19678; seed-1 confirmed: 0.9302, movement 20068) - smoothness HIGH where attribution collapsed. ALL THREE METRICS, all clean-env."),
}

# Tuned lr per completed sweep group (procedure step 5); 2e-4 until the group completes.
# ---------------------------------------------------------------------------------
# Metasmoothness probe, measured 2026-08-24 via scripts/gen_ms.py (bergson
# Metasmoothness command, three trainings per row; no retraining bank).
# CONTROLS: fd_step 0.1, direction_seed 0, always record total_movement_l1.
#
# The four rows that already had a value reproduce it: sm_adamw 0.9930 vs 0.9928
# recorded, sm_muon 0.996358 vs 0.9963, adam 4k 0.994613 vs 0.9946, muon 4k
# 0.903657 vs 0.9037 -- so this path agrees with the older msfill_* runs.
#
# CONTROLS also says to confirm any cell that is surprising or < 0.9 at
# direction_seed 1. Every value here is >= 0.9036, and muon 4k (0.9037) sits
# right at that line -- it is the one to re-confirm at seed 1.
MS_MEASURED = {
    "sm_adamw_eps1e17_16k_bs256":     (0.993036, 34078.5),
    "sm_muon_eps1e17_16k_bs256":      (0.996358, 29223.4),
    "plan_adam_eps1e17_8k_bs256":     (0.992400, 22937.3),
    "plan_muon_eps1e17_8k_bs256":     (0.996233, 16996.5),
    "plan_adam_eps1e17_16k_bs128":    (0.993474, 25212.8),
    "plan_muon_eps1e17_16k_bs128":    (0.994376, 24160.2),
    "plan_adam_eps1e17_16k_clip1.0":  (0.989642, 36087.6),
    "plan_adam_eps1e17_16k_wd0.0":    (0.993033, 34082.1),
    "plan_adam_eps1e17_16k_wd0.1":    (0.993050, 34045.5),
    "plan_adam_eps1e17_4k_bs256":     (0.994613,  7576.98),
    "plan_muon_eps1e17_4k_bs256":     (0.903657, 19677.5),
    # Step-scaling axis at fixed bs32, lr 5e-05 for both optimizers (the
    # interior minimum of the 32k bs32 sweep: adamw 3.2342 vs 3.238 either
    # side, muon 3.231 vs 3.2351). 2000 steps, nproc 2 (from ms.yaml).
    # GPU TYPE UNRECORDED: these runs leave no hardware trace in ms.log,
    # ms.yaml or ms/config.yaml. D17 makes GPU type part of run identity, so it
    # is left unstated rather than guessed -- an earlier version of this comment
    # asserted A40 on no evidence.
    "plan_adam_eps1e17_32k_bs32":     (0.986618, 34054.9),
    "plan_muon_eps1e17_32k_bs32":     (0.994085, 38738.9),
    # 64k rung, lr 2.5e-05 for both (the interior minimum of the 64k bs32
    # sweep). 4000 steps, nproc 2, A40, bellflower-0 -- the pair shares a node,
    # so D17 does not apply to the optimizer contrast between them.
    "plan_adam_eps1e17_64k_bs32":     (0.986871, 24047.7),
    "plan_muon_eps1e17_64k_bs32":     (0.993564, 27276.1),
    # Logit-scale axis, measured on A40 (matching these banks):
    #   scale 1.0  (lr 2e-4) -> ms 0.9930 / MAGIC 0.9411
    #   scale 0.5  (lr 2e-4) -> ms 0.9878 / MAGIC 0.9448
    #   scale 0.25 (lr 8e-4) -> ms 0.9150 / MAGIC 0.0456
    #
    # DIAGNOSTIC 2026-08-24, paper_runs/diagnostics/ms_scale0.25_lr2e-4:
    # the same logit_scale 0.25 with lr held at the ANCHOR 2e-4 gives
    # ms = 0.9812 -- comfortably inside the healthy band.
    #
    # So the scale0.25 row IS collapsed, but the cause is its tuned lr of
    # 8e-4, not the logit scaling. Holding lr fixed, halving the logits
    # costs only ~0.006 of ms per halving (0.9930 -> 0.9878 -> 0.9812).
    # Do NOT describe either the ms drop or the MAGIC collapse on this row
    # as a logit-scale effect; the row varies two things at once and the
    # diagnostic attributes the damage to the learning rate.
    "plan_adam_eps1e17_16k_scale0.5":  (0.987800, None),
    "plan_adam_eps1e17_16k_scale0.25": (0.915000, None),
    # Batch-size axis. Not monotone against MAGIC: bs64 has the higher ms
    # (0.9853 vs 0.9800) and the LOWER attributability (0.7811 vs 0.9201).
    "plan_adam_eps1e17_16k_bs32":     (0.980000, None),
    "plan_adam_eps1e17_16k_bs64":     (0.985300, None),
    "plan_muon_eps1e17_16k_bs64":     (0.993900, None),
    # OUT-OF-SAMPLE TEST for the ms<0.95 collapse boundary, which was drawn
    # from muon 4k and scale0.25 only. bs16 did not inform it: predicted low
    # before measurement (MAGIC 0.1796), measured 0.9133. Three for three.
    # NOTE hardware: this ms ran on A40 (secret-ord-0) while the bs16 bank is
    # A100 (lotus-0). Labelled per the D17 accept-and-label ruling.
    "plan_adam_eps1e17_16k_bs16":     (0.913300, None),
    "plan_muon_eps1e17_16k_bs32":    (0.995200, None),
    "plan_adam_eps1e17_16k_bs512":   (0.995000, None),
    # PRE-REGISTERED PREDICTION (2026-08-24, before its MAGIC exists).
    # gpt2-medium has the LOWEST ms in the grid, below all three known
    # collapsed rows (muon 4k 0.9037, bs16 0.9133, scale0.25 0.9150). Its
    # bank is still scoring (13/20 queries), so nothing about its
    # attributability informed this number.
    #   detector claim holds  => MAGIC should land BELOW ~0.31
    #   MAGIC lands above 0.77 => the claim is falsified on its first
    #                             forward prediction
    # ms ran on A100 (maria-1), matching its A100 bank -- no D17 mismatch.
    "plan_adam_eps1e17_16k_gpt2-medium": (0.858000, None),
    # SECOND PRE-REGISTERED PREDICTION (2026-08-24). muon bs16 sits in the
    # HEALTHY band (0.9939) while adam bs16 at the same batch size is
    # collapsed (0.9133, MAGIC 0.1796). Its bank is at 73/100 models, so its
    # attributability does not exist yet and cannot have informed this.
    #   detector claim holds => muon bs16 MAGIC should land ABOVE ~0.77,
    #                           i.e. a huge optimizer gap at bs16 that no
    #                           other batch size shows
    #   MAGIC lands below 0.31 => falsified in the other direction (a false
    #                           negative: collapsed but high ms)
    # Sharper than the gpt2-medium prediction because both arms share batch
    # size, data, seed and node (lotus-0/A100), differing only in optimizer.
    "plan_muon_eps1e17_16k_bs16":     (0.993900, None),
    # Token axis at bs256 and the 4-epoch row, measured 2026-08-25.
    # All well inside the healthy band, and note the token axis does NOT
    # collapse: 32k and 64k stay at 0.987-0.995 even though 64k is 500 steps.
    # That matters for the step-scaling ladder -- at bs256 more DATA does not
    # move ms, so if the bs32 ladder collapses it is the step count doing it.
    "plan_adam_eps1e17_16k_ep4":      (0.995916, 33964.1),
    "plan_adam_eps1e17_32k_bs256":    (0.993725, 49520.4),
    "plan_muon_eps1e17_32k_bs256":    (0.994801, 47704.0),
    "plan_adam_eps1e17_64k_bs256":    (0.987626, 37072.4),
    "plan_muon_eps1e17_64k_bs256":    (0.994736, 34602.8),
}

TUNED_LR = {"plan_adam_eps1e17_4k_bs256": 1e-4, "plan_adam_eps1e17_8k_bs256": 2e-4,
            "plan_muon_eps1e17_4k_bs256": 4e-4, "plan_muon_eps1e17_8k_bs256": 2e-4, "plan_adam_eps1e17_32k_bs256": 2e-4, "plan_muon_eps1e17_32k_bs256": 2e-4, "plan_adam_eps1e17_16k_bs16": 5e-5, "plan_adam_eps1e17_16k_bs32": 5e-5, "plan_muon_eps1e17_16k_bs16": 5e-5, "plan_adam_eps1e17_64k_bs256": 1e-4, "plan_muon_eps1e17_16k_bs32": 5e-5, "plan_muon_eps1e17_64k_bs256": 1e-4, "plan_muon_eps1e17_16k_bs64": 1e-4, "plan_adam_eps1e17_16k_bs64": 1e-4, "plan_adam_eps1e17_16k_bs128": 1e-4, "plan_muon_eps1e17_16k_bs128": 1e-4, "plan_adam_eps1e17_16k_ep4": 1e-4,
            # 64k bs32 sweep complete on lotus-0: 1.25e-5 3.2365 / 2.5e-5 3.2322 /
            # 5e-5 3.2355 / 1e-4 3.2552 -- interior minimum, no extension needed.
            "plan_adam_eps1e17_64k_bs32": 2.5e-5,
            # muon 64k bs32: 1.25e-5 3.2337 / 2.5e-5 3.2290 / 5e-5 3.2333 --
            # the winner is bracketed on both sides, so the pending 1e-4 point
            # cannot change the selection.
            "plan_muon_eps1e17_64k_bs32": 2.5e-5,
            # Selections that landed on the anchor value - recorded explicitly so
            # "completed sweep => entry here" holds without exception.
            "plan_adam_eps1e17_16k_bs512": 2e-4, "plan_adam_eps1e17_16k_wd0.0": 2e-4,
            "plan_adam_eps1e17_16k_wd0.1": 2e-4, "plan_adam_eps1e17_16k_clip1.0": 2e-4,
            "plan_adam_eps1e17_16k_scale0.5": 2e-4, "plan_adam_eps1e17_16k_scale0.25": 8e-4,
            # gpt2-medium sweep measured 2026-08-23: 5e-5 -> 3.0062,
            # 1e-4 -> 3.0019, 2e-4 -> 3.0085. Interior optimum, but a very flat
            # one (0.0066 nats across 4x).
            "plan_adam_eps1e17_16k_gpt2-medium": 1e-4,
            # Step-scaling axis, measured 2026-08-25. Both arms won on the
            # interior 5e-5, so no endpoint extension.
            "plan_adam_eps1e17_32k_bs32": 5e-5,
            "plan_muon_eps1e17_32k_bs32": 5e-5}
for n in [4000, 8000, 32000, 64000]:
    rid = f"plan_adam_eps1e17_{n//1000}k_bs256"
    add(BASE17, run_id=rid, n_docs=n, lr=TUNED_LR.get(rid, 2e-4),
        notes="N axis (nested chain, EleutherAI/bergson-smollm2-scaling). 16k measured "
              "(MAGIC 0.9333). lr comes from tuning.csv sweep_group "
              f"tune_adamw_{n//1000}k.")
for n in [4000, 8000, 32000, 64000]:
    rid = f"plan_muon_eps1e17_{n//1000}k_bs256"
    add(BASE17, run_id=rid, n_docs=n, optimizer="muon", lr=TUNED_LR.get(rid, 2e-4),
        notes="N axis, muon. 16k measured (MAGIC 0.8470). lr comes from tuning.csv "
              f"sweep_group tune_muon_{n//1000}k.")
# STEP-SCALING axis: METASMOOTHNESS ONLY (2026-08-25).
# How does ms behave as the number of OPTIMISER STEPS grows? Batch size is held
# at 32 -- the config with 1000 steps and ms 0.9800 at 16k -- and the corpus
# grows, so steps scale directly: 32k -> 2000, 64k -> 4000, 128k -> 8000.
# No bank: MAGIC would need one reverse pass per query over the whole corpus,
# 150h+ per row. magic_lds/ekfac_lds stay empty by design.
for n in [32000, 64000, 128000, 256000, 512000]:
    k = f"{n // 1000}k"
    for opt, pre in (("adamw", "adam"), ("muon", "muon")):
        rid = f"plan_{pre}_eps1e17_{k}_bs32"
        add(BASE17, run_id=rid, n_docs=n, optimizer=opt, batch_size=32,
            grad_accum_steps=2, lr=TUNED_LR.get(rid, 5e-5),
            notes=f"ms-only step-scaling point: bs32, {n // 32 * 2} steps. No bank "
                  "planned (MAGIC would cost 150h+ of scoring at this N). lr from "
                  f"tuning.csv sweep_group tune_{opt}_{k}_bs32.")

# D1 (2026-08-20): "warm start" = attribution window, a pre-training experiment —
# removed from the fine-tuning grid. See EXPERIMENTS_CSV.md "Planned pre-training experiments".
add(BASE17, run_id="plan_adam_eps1e17_16k_ep4", num_epochs=4,
    notes="D2: double epochs (250 steps, batch unchanged) — isolates step count. "
          "lr comes from tuning.csv sweep_group tune_adamw_16k_ep4. nproc 2, A40, "
          "bellflower-0; sharded 30-48/48-65/65-82/82-100 on shared-ord-0, also A40, "
          "so the bank is hardware-homogeneous under D17. 100 subsets, 20/20 queries, "
          "CI half-width 0.0068. THE HIGHEST MAGIC IN THE GRID: 0.9534 against the "
          "anchor 0.9411, whose interval [0.9326, 0.9477] it does not overlap. Doubling "
          "epochs at fixed batch RAISES attribution, and it raises ms too (0.9959 vs "
          "0.9930), so on this axis the two move together. The main process overran its "
          "stop and recomputed subsets 30-44, which the slices also covered; the two "
          "copies agree to 3.8e-06 (diff column) and the main copies were dropped, "
          "original kept as validation.csv.premerge.")
add(BASE17, run_id="plan_adam_eps1e17_16k_bs512", batch_size=512, grad_accum_steps=32,
    notes="D2: uncontrolled double batch (63 steps). lr comes from tune_adamw_16k_bs512.")
# The two gpt2-medium bs32 points. Both were measured, both were entered in
# BANK_RESULTS, and neither had a row here -- so neither reached the CSV. The
# 64k one also answers the dataset-size question on this model directly: ms
# 0.85798 at 64k against 0.85802 at 16k, two independent runs (total_movement_l1
# 64432.09 vs 44512.30) agreeing to four decimals. On gpt2-medium ms is flat in
# N over a 4x increase -- it is simply flat at 0.858, not at the 0.98 we want.
add(BASE17, run_id="gpt2medium_16k_bs32", model="gpt2-medium", n_params_m=355,
    batch_size=32, grad_accum_steps=2, lr=5e-5,
    notes="Model-size axis at bs32 (1000 steps), the config D11 registered: an exact "
          "match to plan_adam_eps1e17_16k_bs32 (gpt2, ms 0.9800) with only the model "
          "swapped. WORLD SIZE 4, NVIDIA A100.")
add(BASE17, run_id="gpt2medium_64k_bs32", model="gpt2-medium", n_params_m=355,
    n_docs=64000, batch_size=32, grad_accum_steps=2, lr=2.5e-5,
    notes="gpt2-medium at 64k, bs32, 4000 steps, 2 epochs. WORLD SIZE 4, NVIDIA A100 "
          "(marisa-0). Paired with gpt2medium_16k_bs32 for the dataset-size question "
          "on this model.")

for mdl, prm in [("gpt2-medium", 355), ("gpt2-large", 774)]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_{mdl}", model=mdl, n_params_m=prm,
        notes="Model-size axis. MAGIC is one reverse pass per query and scales with params. "
              + ("gpt2-medium is the registered scaling target (D11)."
                 if mdl == "gpt2-medium" else
                 "Deferred: runs only if gpt2-medium proves informative (D11)."))
add(BASE17, run_id="plan_adam_eps1e17_16k_ckptavg4", ckpt_avg_k=4,
    notes="D9: average the QUERY GRADIENT over the last 4 checkpoints; BOTH scorers (MAGIC "
          "and EK-FAC) use the averaged gradient. Replicate Louis's effect on the anchor "
          "first. D15: the stored anchor base is NOT bit-reachable from the current "
          "environment; a fresh deterministic base + trajectory exist at "
          "/mnt/ssd-2/lucia/paper_runs/d9_magic_base — the replication path awaits the "
          "D15 ruling. Eval-side: exempt from lr gating. NOT RUNNABLE as generated (2026-08-23): bergson has no query-gradient checkpoint-averaging code (no ckpt_avg_k/avg_k/averaged_gradient anywhere in the package) and gen_experiment_run.py never reads the ckpt_avg_k column, so the generated config is byte-equivalent to a plain bs256 anchor. Launching it spends ~69 GB and a full bank reproducing the anchor and answers nothing. Needs the bergson eval-side feature first.")
# preact_batchnorm dropped (D14) — see DECISIONS.md.
# The whole architecture axis is cut per D16 (graft-vs-pretrain design question;
# out of scope for this campaign):
#   qk_norm           cut, registered as future work
#   preact_layernorm  cut -- the same fine-tune-graft design D16 rejected
#   arch_control      cut -- it exists only to control the arch_mod rows above,
#                     and with those cut it controls nothing
# preact_batchnorm was already dropped separately under D14.
for s in [0.5, 0.25]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_scale{s}", logit_scale=s,
        notes="Unblocked 2026-08-23: the logit-scale hook exists on bergson feat/logit-scale "
              "(PR #433, worktree /mnt/ssd-1/lucia/bergson-logit-scale); this row runs "
              "against that checkout, NOT the pinned -429 worktree, so record code_commit. "
              "Rep-era "
              "4k/eps1e-8 data (excluded) moved ms 0.876->0.609 but also delta_l2 — "
              "not a clean isolate; re-measure here.")
for wd in [0.0, 0.1]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_wd{wd}", weight_decay=wd,
        notes="Rep-era data (excluded) suggested weight decay is a null on ms over 0-0.3.")
add(BASE17, run_id="plan_adam_eps1e17_16k_clip1.0", max_grad_norm=1.0,
    notes="Rep-era data (excluded) suggested clipping is a no-op (norms rarely exceed 1).")

# Reuse rule 3: the winning tuning run is the experiment's base model, so its measured
# heldout fills the row (values from tuning.csv winners; models under paper_runs/tuning).
HELDOUT_FROM_TUNING = {
    "plan_adam_eps1e17_4k_bs256": 3.3149, "plan_muon_eps1e17_4k_bs256": 3.3114,
    "plan_adam_eps1e17_8k_bs256": 3.2851, "plan_muon_eps1e17_8k_bs256": 3.2841,
    "plan_adam_eps1e17_32k_bs256": 3.2365, "plan_muon_eps1e17_32k_bs256": 3.2372,
    "plan_adam_eps1e17_64k_bs256": 3.2314, "plan_muon_eps1e17_64k_bs256": 3.2323,
    "plan_adam_eps1e17_16k_bs16": 3.2497, "plan_muon_eps1e17_16k_bs16": 3.2443,
    # Step-ladder rows at 2000 steps. Measured from each run own base/model with
    # heldout_eval.py and they reproduce their tuning winners exactly (adamw 5e-5
    # 3.2342, muon 5e-5 3.2310), which is reuse rule 3 doing what it claims.
    "plan_adam_eps1e17_32k_bs32": 3.2342, "plan_muon_eps1e17_32k_bs32": 3.2310,
    "plan_adam_eps1e17_16k_bs32": 3.2473, "plan_muon_eps1e17_16k_bs32": 3.2441,
    "plan_adam_eps1e17_16k_bs64": 3.2479, "plan_muon_eps1e17_16k_bs64": 3.2464,
    "plan_adam_eps1e17_16k_bs128": 3.2498, "plan_muon_eps1e17_16k_bs128": 3.2501,
    "plan_adam_eps1e17_16k_ep4": 3.2503, "plan_adam_eps1e17_16k_bs512": 3.2751,
    "plan_adam_eps1e17_16k_wd0.0": 3.2572, "plan_adam_eps1e17_16k_wd0.1": 3.2572,
    "plan_adam_eps1e17_16k_clip1.0": 3.2543,
    "sm_adamw_eps1e17_16k_bs256": 3.2572, "sm_muon_eps1e17_16k_bs256": 3.2570,
}
for r in rows:
    if r["run_id"] in HELDOUT_FROM_TUNING and not r.get("heldout_loss"):
        r["heldout_loss"] = HELDOUT_FROM_TUNING[r["run_id"]]

# Parameter-update norms ||theta_final - theta_0||, measured 2026-08-24 by
# scripts/param_delta.py off each run's own checkpoints/step_0.ckpt + final step.
# Backfills delta_l1/delta_l2, empty for the whole paper grid since the legacy
# eps-root-damping family (their only prior source) was excluded 2026-08-22.
PARAM_DELTA = {  # run_id: (delta_l1, delta_l2)
    'plan_adam_eps1e17_16k_bs128': (193213.92, 19.7564),
    'plan_adam_eps1e17_16k_bs16': (347689.76, 39.7684),
    'plan_adam_eps1e17_16k_bs32': (210602.27, 22.4467),
    'plan_adam_eps1e17_16k_bs512': (212701.43, 20.8275),
    'plan_adam_eps1e17_16k_bs64': (265521.33, 27.6404),
    'plan_adam_eps1e17_16k_clip1.0': (272329.81, 27.6562),
    'plan_adam_eps1e17_16k_scale0.25': (1202912.19, 123.6988),
    'plan_adam_eps1e17_16k_scale0.5': (311640.88, 30.7532),
    'plan_adam_eps1e17_16k_wd0.0': (269182.05, 27.2166),
    'plan_adam_eps1e17_16k_wd0.1': (272062.23, 27.5383),
    'plan_adam_eps1e17_4k_bs256': (84244.26, 8.0412),
    'plan_adam_eps1e17_8k_bs256': (207123.55, 20.5755),
    'plan_muon_eps1e17_16k_bs128': (142420.08, 16.4346),
    'plan_muon_eps1e17_16k_bs32': (159088.47, 18.4238),
    'plan_muon_eps1e17_16k_bs64': (213165.19, 24.4609),
    'plan_muon_eps1e17_4k_bs256': (110034.62, 13.1686),
    'plan_muon_eps1e17_8k_bs256': (101152.33, 11.9494),
    'sm_adamw_eps1e17_16k_bs256': (269255.14, 27.2257),
    'sm_muon_eps1e17_16k_bs256': (173588.09, 20.0331),
}
# Final train and held-out CE, measured 2026-08-24 by scripts/run_losses.py off each run's
# own final checkpoint (both 16k anchors' tuning models had already been cleaned up). The
# logit-scale rows are evaluated WITH their scale applied. Reproduced the 17 recorded
# heldout values to <=5e-4, so heldout is filled here only where it was missing —
# the rest keep their reuse-rule-3 provenance from the tuning winners.
RUN_LOSSES = {  # run_id: (train_loss, heldout_loss)
    'plan_adam_eps1e17_16k_bs128': (3.1162, 3.2499),
    'plan_adam_eps1e17_16k_bs16': (3.0698, 3.2498),
    'plan_adam_eps1e17_16k_bs32': (3.1031, 3.2478),
    'plan_adam_eps1e17_16k_bs512': (3.1700, 3.2754),
    'plan_adam_eps1e17_16k_bs64': (3.0715, 3.2479),
    'plan_adam_eps1e17_16k_clip1.0': (3.0919, 3.2544),
    'plan_adam_eps1e17_16k_scale0.25': (3.2238, 3.4343),
    'plan_adam_eps1e17_16k_scale0.5': (3.1896, 3.3022),
    'plan_adam_eps1e17_16k_wd0.0': (3.1078, 3.2573),
    'plan_adam_eps1e17_16k_wd0.1': (3.1077, 3.2573),
    'plan_adam_eps1e17_4k_bs256': (3.2064, 3.3154),
    'plan_adam_eps1e17_8k_bs256': (3.1309, 3.2853),
    'plan_muon_eps1e17_16k_bs128': (3.1204, 3.2502),
    'plan_muon_eps1e17_16k_bs16': (3.0518, 3.2440),
    'plan_muon_eps1e17_16k_bs32': (3.0978, 3.2439),
    'plan_muon_eps1e17_16k_bs64': (3.0648, 3.2463),
    'plan_muon_eps1e17_4k_bs256': (3.1281, 3.3119),
    'plan_muon_eps1e17_8k_bs256': (3.1560, 3.2844),
    'sm_adamw_eps1e17_16k_bs256': (3.1078, 3.2573),
    'sm_muon_eps1e17_16k_bs256': (3.1135, 3.2571),
}
for r in rows:
    if r["run_id"] in RUN_LOSSES:
        tr, ho = RUN_LOSSES[r["run_id"]]
        r["train_loss"] = tr
        if not r.get("heldout_loss"):
            r["heldout_loss"] = ho

for r in rows:
    if r["run_id"] in PARAM_DELTA:
        r["delta_l1"], r["delta_l2"] = PARAM_DELTA[r["run_id"]]

# TUNED_LR is applied here, to EVERY row, in one place. It was previously consumed
# only by the N-axis loops, which shipped 9 batch-size/ep4 rows at the 2e-4 default
# instead of their tuned 5e-5/1e-4 values - caught by bellflower-0 after 9 banks
# launched on the wrong lr. The check below makes a missed consumption fail the
# build: every completed tuning group's winner must match its experiment row.
for r in rows:
    if r["run_id"] in TUNED_LR:
        r["lr"] = TUNED_LR[r["run_id"]]

_matched = set()
for r in rows:
    if r["run_id"] in BANK_RESULTS:
        _matched.add(r["run_id"])
    r.update(BANK_RESULTS.get(r["run_id"], {}))

# BANK_RESULTS UPDATES rows, it never creates them, so a key naming a run with no
# add() call above matches nothing and its measurements are dropped in silence.
# That is not hypothetical: gpt2medium_16k_bs32 sat here with ms 0.7402 for a
# day, correct on disk and correct in this file, and never once appeared in
# experiments.csv. Recording a result and publishing a result are different
# things, and nothing connected them. Fail loudly instead.
_orphans = sorted(set(BANK_RESULTS) - _matched)
if _orphans:
    raise SystemExit(
        "BANK_RESULTS has %d key(s) with no matching row, so their results would "
        "be silently dropped:\n  %s\nAdd an add(...) row for each, or remove the "
        "entry." % (len(_orphans), "\n  ".join(_orphans)))

# D15 final ruling, scoped per Lucia: only results that RODE ON the known-invalid
# banks are struck - the MAGIC and EK-FAC cells scored against the deleted
# pre-venv banks (old s16k anchors, per-epoch banks). Metasmoothness and
# movement values are training probes with no bank dependence and STAND, as do
# loss cells and tuned lrs. Historical struck values remain in LDS_RESULTS.md.
_venv_rows = set(BANK_RESULTS)
for r in rows:
    if r["run_id"] in _venv_rows:
        continue
    had_bank_metric = any(str(r.get(k, "")) not in ("", "None")
                          for k in ("magic_lds", "ekfac_lds"))
    if had_bank_metric:
        for k in ("magic_lds", "magic_ci_lo", "magic_ci_hi", "magic_n_queries",
                  "ekfac_lds", "ekfac_ci_lo", "ekfac_ci_hi", "ekfac_n_subsets"):
            r[k] = ""
        r["reusable"] = "none"
        r["notes"] = ("MAGIC/EK-FAC cells struck: scored against a deleted pre-venv bank "
                      "(D15 ruling; historical values in LDS_RESULTS.md). ms/movement "
                      "cells stand (no bank dependence). ") + r.get("notes", "")
    has_ms = str(r.get("metasmoothness", "")) not in ("", "None")
    if str(r.get("magic_lds", "")) in ("", "None") and str(r.get("ekfac_lds", "")) in ("", "None"):
        r["status"] = "partial" if has_ms else "planned"

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning.csv")) as _f:
    _trows = list(csv.DictReader(_f))
_winners: dict[str, float] = {}
for _t in _trows:
    _g = [x for x in _trows if x["sweep_group"] == _t["sweep_group"]]
    if _g and all(x["status"] == "measured" and x["heldout_loss"] for x in _g):
        _best = min(_g, key=lambda x: float(x["heldout_loss"]))
        for _target in _best["selects_lr_for"].split(";"):
            _winners.setdefault(_target.strip(), float(_best["lr"]))
# The selected value can differ from the raw min (tie rule), so TUNED_LR is the
# authority on the value; this check enforces that a completed sweep's selection
# was RECORDED - the failure mode that shipped the wrong-lr rows was a completed
# group whose selection never reached the row.
for r in rows:
    if r["run_id"] in _winners and r["run_id"].startswith("plan_"):
        assert r["run_id"] in TUNED_LR, (
            f"{r['run_id']}: tuning group complete but its selection is not recorded "
            f"in TUNED_LR - the wrong-lr failure mode; record the selection")


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
        prev = old.get(r["run_id"])
        if prev:
            r["node_in_charge"] = prev.get("node_in_charge", "") or r.get("node_in_charge", "")
            r["node_checkin_date"] = prev.get("node_checkin_date", "") or r.get("node_checkin_date", "")



# Tail-filter ("proponent filter") results, measured 2026-08-25 via
# scripts/gen_filter.py + scripts/filter_power.py. Per query: remove the top 1%
# of docs the scorer ranks most influential, retrain once, measure that query's
# loss change against the row's own leave-k-out bank (random removals of the
# same size). Raw per-query data in data/filter_raw/ and data/filter_power.csv.
#
# z = (filter_change - random_mean) / random_sd is the comparable number: raw
# loss deltas are in nats and grow with N, so they are not comparable across
# dataset sizes. pctile saturates at 0.990 (rank 1 of 101) whenever the filter
# beats every random removal, which it does on every query measured so far, so
# it is recorded but carries no resolution at this fraction.
def _load_filter_deltas():
    """Read the measured deltas instead of restating them here.

    This used to be a hand-maintained dict, so every finished filter run needed a
    manual edit and the CSV silently fell behind -- thirteen measured rows were
    missing at once. scripts/filter_deltas.py recomputes data/filter_deltas.csv
    from every filter_summary.csv on disk, so reading that file makes the grid
    current the moment the deltas are recomputed, with nothing to remember.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "filter_deltas.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            def trio(prefix):
                try:
                    return (float(r[f"{prefix}_mean"]),
                            float(r[f"{prefix}_lo"]),
                            float(r[f"{prefix}_hi"]))
                except (KeyError, ValueError):
                    return None
            out[r["run"]] = (trio("random"), trio("ekfac"), trio("magic"))
    return out


FILTER_DELTAS = _load_filter_deltas()


def _apply_filter_power(rows):
    """Fill the tail-filter cells. Must run after every row is added."""
    for r in rows:
        hit = FILTER_DELTAS.get(r["run_id"])
        if not hit:
            continue
        for label, trio in zip(("random", "ekfac", "magic"), hit):
            # A row can have one scorer's arm measured and not the other, so a
            # missing trio is normal and must not blank the arm that exists.
            if trio is None:
                continue
            m, lo, hi = trio
            r[f"filter_{label}_delta"] = m
            r[f"filter_{label}_lo"] = lo
            r[f"filter_{label}_hi"] = hi
        r["filter_frac"] = 0.01
        r["filter_n_queries"] = 20


def _apply_ms(rows):
    """Fill the metasmoothness cells. Must run after EVERY row is added."""
    for r in rows:
        hit = MS_MEASURED.get(r["run_id"])
        if hit:
            score, movement = hit
            r["metasmoothness"] = score
            if movement is not None:
                r["ms_total_movement_l1"] = movement
            r["ms_direction_seed"] = 0
            r["ms_fd_step"] = 0.1


# Standing requirement (2026-08-24): every `done` row carries the tuned lr, the parameter
# update norm, and both losses. These are the cells that quietly emptied when the legacy
# eps-root-damping family was excluded (2026-08-22) — delta_l1/delta_l2 and train_loss were
# populated ONLY by rep-era rows, so the whole paper grid went blank without anything
# failing, and the columns survived in the schema looking like measurements that were
# skipped. Same failure shape as the wrong-lr incident, so it gets the same treatment: a
# build-time assert. KNOWN_GAPS is the shrinking allowlist of cells not yet backfilled;
# a cell that is NOT listed there fails the build, so new rows cannot ship incomplete.
REQUIRED_WHEN_DONE = ("lr", "delta_l1", "delta_l2", "train_loss", "heldout_loss")
KNOWN_GAPS = set()  # empty: every required cell is measured as of 2026-08-24


def _assert_per_run_diagnostics(rows):
    missing, excused = [], []
    for r in rows:
        if r.get("status") != "done":
            continue
        for col in REQUIRED_WHEN_DONE:
            if str(r.get(col, "")) not in ("", "None"):
                continue
            if ("*", col) in KNOWN_GAPS or (r["run_id"], col) in KNOWN_GAPS:
                excused.append((r["run_id"], col))
            else:
                missing.append((r["run_id"], col))
    assert not missing, (
        "done rows missing required per-run diagnostics: "
        + ", ".join(f"{rid}.{col}" for rid, col in missing)
        + " — measure it (scripts/param_delta.py for delta_l*, scripts/heldout_eval.py "
          "for the losses) or add it to KNOWN_GAPS with the reason")
    if excused:
        cols = {}
        for rid, col in excused:
            cols.setdefault(col, []).append(rid)
        for col, rids in sorted(cols.items()):
            print(f"  KNOWN GAP  {col:14} {len(rids)} done rows awaiting backfill")


def main():
    _apply_ms(rows)
    _apply_filter_power(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.csv")
    for r in rows:
        for c in COLUMNS:
            r.setdefault(c, "")
        assert r["dropout_effective"] in (0.0, ""), f"active dropout row admitted: {r['run_id']}"
        assert r["shuffle"] in ("per_epoch", "agnostic_1ep"), f"rep row admitted: {r['run_id']}"
        assert r["dataset"] == "smollm2", (
            f"non-smollm2 row admitted: {r['run_id']} — paper runs use the SmolLM2 "
            "pipeline only (WikiText does not scale)")
    _assert_per_run_diagnostics(rows)
    order = {"done": 0, "partial": 1, "planned": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), r["family"], r["run_id"]))
    _preserve_claims(out, rows)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    n = {s: sum(1 for r in rows if r["status"] == s) for s in ("done", "partial", "planned")}
    print(f"wrote {out}: {len(rows)} rows ({n['done']} done, {n['partial']} partial, "
          f"{n['planned']} planned)")
    for col in ("metasmoothness", "magic_lds", "ekfac_lds"):
        have = sum(1 for r in rows if str(r[col]) != "")
        print(f"  {col:16} {have:3}/{len(rows)} filled")
    ids = [r["run_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate run_id"


if __name__ == "__main__":
    main()
