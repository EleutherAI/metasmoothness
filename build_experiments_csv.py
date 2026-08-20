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
steps       n_docs * num_epochs / batch_size (global batch)
eps_root    epsilon inside the AdamW sqrt: m / (sqrt(v + eps_root) + adam_eps). Non-standard.
            For muon it reaches only the AdamW-fallback params (121,344 of 163M = 0.07%).
warmup      fraction of total steps (the trainer's convention). Planned warm-start rows that
            specify absolute steps say so in their notes and must not share a plot axis.
ckpt_avg_k  near-final checkpoints the query loss is averaged over (1 = none). Needs
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
    "metasmoothness", "ms_direction_seed", "ms_fd_step",
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
        r["steps"] = round(r["n_docs"] * r["num_epochs"] / r["batch_size"])
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
for rid, n, opt, lr, eps, bs, ep, ms, ek, lo, hi, l1, l2, bd in GRID:
    extra = {}
    if rid == "sm_adam_eps1e8_4k_rep2":
        extra = dict(dropout_cfg=0.0,
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
for rid, n, opt, lr, eps, bs, ep, ms, note in MS_ONLY:
    add(GPT2_FT, run_id=rid, status="partial", n_docs=n, optimizer=opt, lr=lr, eps_root=eps,
        batch_size=bs, num_epochs=ep, metasmoothness=ms, reusable="ms_only",
        run_dir="/mnt/ssd-2/lucia/muon_ms_steps" if opt == "muon" and n > 4000 else "",
        notes=note)

# epochs=1: shuffle-agnostic, so the rep-era measurements remain valid. The only row
# with all three metrics.
add(GPT2_FT, run_id="sm_adam_eps1e8_4k_bs32_ep1", status="done", n_docs=4000, eps_root=1e-8,
    batch_size=32, num_epochs=1, shuffle="agnostic_1ep", metasmoothness=0.837,
    ekfac_lds=0.1781, ekfac_n_subsets=50, magic_lds=0.05, magic_ci_lo=-0.054,
    magic_ci_hi=0.145, magic_n_queries=20, train_loss=3.07, delta_l1=0.0087,
    delta_l2=0.0100, run_dir="/mnt/ssd-2/lucia/muon4k/magicroll_bs32",
    reusable="bank+scores",
    notes="1 epoch = one pass over one order, so the shuffle implementation cannot matter; "
          "admitted although measured on the older code. MAGIC on the FIXED metagrad code "
          "(c0f11ba8); CI spans zero.")

# eps_root 1e-17 pair (scaling_magic, 2026-08-07): per-epoch trained, MAGIC measured.
for opt, lds, lo, hi in [("adamw", 0.9333, 0.9186, 0.9448), ("muon", 0.8470, 0.8274, 0.8685)]:
    add(GPT2_FT, run_id=f"sm_{opt}_eps1e17_16k_bs256", status="partial", n_docs=16000,
        optimizer=opt, lr=2e-4, eps_root=1e-17, batch_size=256, grad_accum_steps=16,
        num_epochs=2, magic_lds=lds, magic_ci_lo=lo, magic_ci_hi=hi, magic_n_queries=20,
        n_subsets=100, n_queries=20, run_dir=f"/mnt/ssd-2/lucia/s16k_{opt}",
        bank_dir=f"/mnt/ssd-2/lucia/s16k_{opt}/merged", code_commit="docs-4",
        reusable="bank", source_doc="examples/scaling_magic/LDS_RESULTS.md",
        notes="Paired diff adamw-muon = +0.0863 [+0.0670, +0.1052], 19/20 per-query wins; "
              "identical subset lists. metasmoothness and EK-FAC NOT measured — the 100-model "
              "bank exists, so EK-FAC is scoring-only. Base-training ckpts deleted (retrains "
              "reproduce deterministically)." +
              (" adamw scores rebuilt from per-query .pt files (padded-query bug on docs-4)."
               if opt == "adamw" else ""))

# =====================================================================================
# 2. OLMo2 from-scratch (dropout genuinely 0.0; per-epoch rows only). Different model
#    family — kept for the pre-training endpoint; filter on `family` for GPT-2-only plots.
# =====================================================================================
add(OLMO2, run_id="olmo2_muon_16k_full", status="partial", n_docs=16000,
    metasmoothness=-0.000, train_loss=2.92, delta_l1=4.56, delta_l2=4.10,
    run_dir="/mnt/ssd-2/lucia/scratch_olmo", reusable="bank",
    notes="Dead endpoint: ms ~= 0 (below the ~0.02 information floor). The rep-era full-run "
          "EK-FAC (0.0175, CI spans 0) is excluded; the rep bank exists if a per-epoch rebuild "
          "is ever wanted, but the per-epoch ms says the answer is already 'unattributable'.")
add(OLMO2, run_id="olmo2_muon_16k_tail083", status="partial", n_docs=16000,
    attr_window_frac=0.833, metasmoothness=0.984, ekfac_lds=0.161, ekfac_ci_lo=0.123,
    ekfac_ci_hi=0.198, ekfac_n_subsets=50, n_queries=50, train_loss=3.23,
    run_dir="runs/tail_bank_083_full", code_commit="5833a9b3", reusable="bank",
    notes="MAIN RESULT: attributing only the last epoch makes pre-training scoreable (9x, "
          "disjoint CIs) at the model's full loss. Tail-MAGIC was in progress, not recorded.")
for frac, ms in [(0.25, 0.025), (0.5, 0.355), (0.6, 0.669), (0.75, 0.793),
                 (0.896, 0.986), (0.95, 0.993), (0.99, 0.990)]:
    add(OLMO2, run_id=f"olmo2_muon_16k_window{frac}", status="partial", n_docs=16000,
        attr_window_frac=frac, metasmoothness=ms, reusable="ms_only",
        notes="Window sweep, ms only. frac>0.833 loses doc coverage (bank would be invalid).")
for n, st, ms, loss in [(4000, 188, 0.0095, 4.98), (8000, 375, 0.0177, 3.95),
                        (32000, 1500, 0.0051, 3.09)]:
    add(OLMO2, run_id=f"olmo2_muon_{n//1000}k_full", status="partial", n_docs=n, steps=st,
        metasmoothness=ms, train_loss=loss, reusable="ms_only",
        notes="Full-run attribution flat at ~0 across 1.9 nats of loss.")
for knob, ms, loss, kw in [
        ("opt_adamw", 0.647, 6.18, dict(optimizer="adamw", lr=8e-4)),
        ("lr3e-3", 0.019, 2.69, dict(lr=3e-3)),
        ("wd0", 0.003, 3.27, dict(weight_decay=0.0)),
        ("epsroot1e-4", 0.004, 3.34, dict(eps_root=1e-4)),
        ("bs64", 0.005, 4.31, dict(batch_size=64, num_epochs=3)),
        ("bs256", 0.006, 1.34, dict(batch_size=256, num_epochs=12))]:
    add(OLMO2, run_id=f"olmo2_muon_16k_{knob}", status="partial", n_docs=16000,
        metasmoothness=ms, train_loss=loss, reusable="ms_only",
        notes="opt_adamw's 0.647 is at loss 6.18 — unusable." if knob == "opt_adamw" else "",
        **kw)

# =====================================================================================
# 3. FILL rows — measurements missing on configs whose artifacts already exist (cheap).
# =====================================================================================
for rid, bd in [(r[0], r[13]) for r in GRID if not r[0].endswith("_rep2")]:
    add(GPT2_FT, run_id=f"fill_{rid}_magic", status="planned",
        n_docs=dict(GRID_N := {g[0]: g[1] for g in GRID})[rid],
        optimizer={g[0]: g[2] for g in GRID}[rid], lr={g[0]: g[3] for g in GRID}[rid],
        eps_root={g[0]: g[4] for g in GRID}[rid],
        batch_size={g[0]: g[5] for g in GRID}[rid],
        num_epochs={g[0]: g[6] for g in GRID}[rid],
        bank_dir=f"{PEREPOCH_RUNS}/{bd}/bank", source_doc="planned",
        notes=f"MAGIC per-query rollout for {rid}: the per-epoch bank exists, so only the "
              "metagradient run + validate --retrained_dir is needed. Budget one reverse pass "
              "per query.")
for opt in ["adamw", "muon"]:
    add(GPT2_FT, run_id=f"fill_sm_{opt}_eps1e17_16k_bs256_ms_ekfac", status="planned",
        n_docs=16000, optimizer=opt, lr=2e-4, eps_root=1e-17, batch_size=256,
        grad_accum_steps=16, source_doc="planned",
        notes=f"Measure metasmoothness + EK-FAC for sm_{opt}_eps1e17_16k_bs256; the 100-model "
              "bank exists so EK-FAC is scoring-only. Would give bs256 all three metrics.")

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
for n in [4000, 8000, 32000]:
    add(BASE17, run_id=f"plan_adam_eps1e17_{n//1000}k_bs256", n_docs=n,
        notes="N axis. 16k measured (MAGIC 0.9333).")
for n in [4000, 8000, 32000]:
    add(BASE17, run_id=f"plan_muon_eps1e17_{n//1000}k_bs256", n_docs=n, optimizer="muon",
        notes="N axis, muon. 16k measured (MAGIC 0.8470).")
for w in [100, 200, 500]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_warmup{w}", warmup=w,
        notes="ABSOLUTE warmup steps (target axis 100-500), vs the baseline's 0.25 fraction "
              "(~31 steps of 125). 500 exceeds the 125-step run — extend epochs or treat as "
              "full-warmup; decide before running, and never plot against fraction rows.")
for mdl, prm in [("gpt2-medium", 355), ("gpt2-large", 774)]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_{mdl}", model=mdl, n_params_m=prm,
        notes="Model-size axis. MAGIC is one reverse pass per query and scales with params.")
for k in [4, 8]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_ckptavg{k}", ckpt_avg_k=k,
        notes="Louis: average query loss over k near-final checkpoints. Needs "
              "cleanup_ckpts=false and the checkpoints kept.")
for mod in ["qk_norm", "preact_layernorm", "preact_batchnorm"]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_{mod}", arch_mod=mod, model="gpt2_custom",
        notes="Needs the GPT-2-like custom model. Compare ONLY against "
              "plan_adam_eps1e17_16k_arch_control (same custom model, no mod), never stock gpt2.")
add(BASE17, run_id="plan_adam_eps1e17_16k_arch_control", arch_mod="none", model="gpt2_custom",
    notes="Control for the arch_mod rows: custom model, no modification.")
for s in [0.5, 0.25]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_scale{s}", logit_scale=s,
        notes="Rep-era 4k/eps1e-8 data (excluded) moved ms 0.876->0.609 but also delta_l2 — "
              "not a clean isolate; re-measure here.")
for wd in [0.0, 0.1]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_wd{wd}", weight_decay=wd,
        notes="Rep-era data (excluded) suggested weight decay is a null on ms over 0-0.3.")
add(BASE17, run_id="plan_adam_eps1e17_16k_clip1.0", max_grad_norm=1.0,
    notes="Rep-era data (excluded) suggested clipping is a no-op (norms rarely exceed 1).")


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.csv")
    for r in rows:
        for c in COLUMNS:
            r.setdefault(c, "")
        assert r["dropout_effective"] in (0.0, ""), f"active dropout row admitted: {r['run_id']}"
        assert r["shuffle"] in ("per_epoch", "agnostic_1ep"), f"rep row admitted: {r['run_id']}"
        assert r["dataset"] == "smollm2", (
            f"non-smollm2 row admitted: {r['run_id']} — paper runs use the SmolLM2 "
            "pipeline only (WikiText does not scale)")
    order = {"done": 0, "partial": 1, "planned": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), r["family"], r["run_id"]))
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
