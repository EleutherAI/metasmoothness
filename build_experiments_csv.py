"""Build experiments.csv — the source-of-truth grid for the metasmoothness paper.

One row per experiment: a training configuration plus the attribution metrics measured on
it. Parameter columns come first, result columns last. Empty result cells are *not-yet-run*,
not failures: an agent with spare compute can select rows where a result column is empty and
fill it.

Rows are transcribed from LDS_RESULTS.md, BASELINE_LDS.md, SHAMPOO_RESULTS.md and the
scaling_magic results in bergson-damping/examples/scaling_magic/. Regenerate with
`python build_experiments_csv.py`; edit the row tables here rather than the CSV so the two
never drift.

Conventions
-----------
status      done    = every result column that this config is meant to carry is filled
            partial = at least one of {metasmoothness, magic_lds, ekfac_lds} is missing
            planned = nothing run yet; the row exists to be claimed
steps       N * num_epochs / batch_size (global batch)
shuffle     rep      = shuffle once then .repeat(epochs) — same order every epoch
            per_epoch = reshuffled each epoch (commit 1e6eea7f)
ms_shuffle  how the metasmoothness probe itself shuffled; several early rows measured
            per-epoch metasmoothness against a rep-trained bank, which is recorded, not fixed
train_mode  false => trainer calls model.eval(), so the configured dropout rate is INERT
eps_root    epsilon inside the AdamW sqrt: m / (sqrt(v + eps_root) + adam_eps). For muon it
            reaches only the AdamW-fallback params (~0.07% of GPT-2), not the 2D Newton-Schulz path
reusable    bank+scores = retrained models on disk, can re-score a new method without retraining
            bank        = retrained models only
            ms_only     = metasmoothness.json only
            none        = artifacts deleted; re-run from the config
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
    "dropout", "train_mode", "shuffle", "seed", "precision", "ckpt_avg_k",
    # --- attribution / estimator ---
    "attr_window_frac", "n_subsets", "subset_fraction", "n_queries",
    # --- results ---
    "metasmoothness", "ms_shuffle", "ms_direction_seed", "ms_fd_step",
    "magic_lds", "magic_ci_lo", "magic_ci_hi", "magic_n_queries",
    "ekfac_lds", "ekfac_ci_lo", "ekfac_ci_hi", "ekfac_n_subsets",
    "train_loss", "delta_l1", "delta_l2",
    # --- provenance ---
    "run_dir", "bank_dir", "code_commit", "reusable", "source_doc", "notes",
]

# Defaults for the GPT-2 fine-tuning family; every row overrides what differs.
GPT2_FT = dict(
    family="gpt2_ft", model="gpt2", model_init="pretrained", arch_mod="none",
    n_params_m=124, logit_scale=1.0, dataset="smollm2", chunk_length=0,
    optimizer="adamw", lr=8e-4, lr_scheduler="polynomial", warmup=0.25,
    adam_eps=1e-8, beta1=0.95, beta2=0.975, weight_decay=0.01, max_grad_norm="",
    batch_size=64, grad_accum_steps=1, num_epochs=2, dropout=0.1, train_mode=False,
    shuffle="rep", seed=42, precision="fp32", ckpt_avg_k=1,
    attr_window_frac=0.0, subset_fraction=0.01, n_subsets=50,
    ms_direction_seed=0, ms_fd_step=0.1, source_doc="LDS_RESULTS.md",
)

OLMO2 = dict(
    family="olmo2_scratch", model="olmo2_reinit", model_init="scratch", arch_mod="none",
    n_params_m=124, logit_scale=1.0, dataset="smollm2", chunk_length=0,
    optimizer="muon", lr=9e-3, lr_scheduler="polynomial", warmup=0.25,
    eps_root=1e-6, adam_eps=1e-8, beta1=0.95, beta2=0.975, weight_decay=0.1,
    max_grad_norm="", batch_size=128, grad_accum_steps=1, num_epochs=6,
    dropout=0.0, train_mode=False, shuffle="per_epoch", seed=42, precision="fp32",
    ckpt_avg_k=1, attr_window_frac=0.0, subset_fraction=0.01, n_subsets=50,
    ms_direction_seed=0, ms_fd_step=0.1, source_doc="LDS_RESULTS.md",
)

WIKITEXT = dict(
    GPT2_FT, dataset="wikitext", n_docs=4358, num_epochs=4, lr=8e-4,
)

rows = []


def add(base, **kw):
    r = dict(base)
    r.update(kw)
    if "steps" not in r and r.get("n_docs") and r.get("batch_size") and r.get("num_epochs"):
        r["steps"] = round(r["n_docs"] * r["num_epochs"] / r["batch_size"])
    rows.append(r)


# =====================================================================================
# 1. GPT-2 fine-tune on SmolLM2 — eps_root x N grid (headline). shuffle=rep banks.
#    metasmoothness on these rows was probed with a per-epoch shuffle (recorded as-is).
# =====================================================================================
add(GPT2_FT, run_id="sm_adam_eps1e6_4k", status="done", n_docs=4000, eps_root=1e-6,
    metasmoothness=0.991, ms_shuffle="rep", ekfac_lds=0.3173, ekfac_ci_lo=0.285,
    ekfac_ci_hi=0.348, ekfac_n_subsets=50, magic_lds=0.86, magic_ci_lo=0.844,
    magic_ci_hi=0.877, magic_n_queries=20, train_loss=3.18, delta_l1=0.0016,
    delta_l2=0.0021, bank_dir="EleutherAI/bergson-smollm2-lds-4k", reusable="bank+scores",
    notes="HF-hosted bank. Re-score with 1ba43f92 gives EK-FAC 0.3156.")
add(GPT2_FT, run_id="sm_adam_eps1e6_8k", status="done", n_docs=8000, eps_root=1e-6,
    metasmoothness=0.978, ms_shuffle="rep", ekfac_lds=0.3019, ekfac_ci_lo=0.267,
    ekfac_ci_hi=0.338, ekfac_n_subsets=50, magic_lds=0.98, magic_ci_lo=0.980,
    magic_ci_hi=0.987, magic_n_queries=20, train_loss=3.18, delta_l1=0.0024,
    delta_l2=0.0028, bank_dir="runs/ekfac_vs_n/N8k", reusable="bank+scores")
add(GPT2_FT, run_id="sm_adam_eps1e6_16k", status="partial", n_docs=16000, eps_root=1e-6,
    metasmoothness=0.9954, ms_shuffle="per_epoch", ekfac_lds=0.3815, ekfac_ci_lo=0.352,
    ekfac_ci_hi=0.412, ekfac_n_subsets=50, train_loss=3.17, delta_l1=0.0039,
    delta_l2=0.0042, bank_dir="runs/ekfac_vs_n/N16k", reusable="bank",
    notes="MAGIC not run.")
add(GPT2_FT, run_id="sm_adam_eps1e6_32k", status="partial", n_docs=32000, eps_root=1e-6,
    metasmoothness=0.9979, ms_shuffle="per_epoch", ekfac_lds=0.3575, ekfac_ci_lo=0.325,
    ekfac_ci_hi=0.394, ekfac_n_subsets=50, train_loss=3.19, delta_l1=0.0069,
    delta_l2=0.0072, bank_dir="runs/ekfac_vs_n/N32k", reusable="bank",
    notes="MAGIC not run.")
add(GPT2_FT, run_id="sm_adam_eps1e8_4k", status="done", n_docs=4000, eps_root=1e-8,
    metasmoothness=0.876, ms_shuffle="rep", ekfac_lds=0.3033, ekfac_ci_lo=0.274,
    ekfac_ci_hi=0.334, ekfac_n_subsets=50, magic_lds=0.17, magic_n_queries=20,
    train_loss=3.02, delta_l1=0.0079, delta_l2=0.0091,
    run_dir="/mnt/ssd-2/lucia/muon4k/magicroll_eps1e8_4k", reusable="bank+scores",
    notes="MAGIC on the FIXED metagrad code (c0f11ba8); pre-fix value was 0.37.")
add(GPT2_FT, run_id="sm_adam_eps1e10_4k", status="done", n_docs=4000, eps_root=1e-10,
    metasmoothness=0.781, ms_shuffle="rep", ekfac_lds=0.2097, ekfac_ci_lo=0.182,
    ekfac_ci_hi=0.239, ekfac_n_subsets=50, magic_lds=-0.02, magic_ci_lo=-0.065,
    magic_ci_hi=0.023, magic_n_queries=20, train_loss=2.71, delta_l1=0.0293,
    delta_l2=0.0295, run_dir="/mnt/ssd-2/lucia/muon4k/magicroll_eps1e10_4k",
    reusable="bank+scores", notes="MAGIC CI spans zero.")
add(GPT2_FT, run_id="sm_adam_eps0_4k", status="partial", n_docs=4000, eps_root=0,
    metasmoothness=0.766, ms_shuffle="rep", ekfac_lds=0.1740, ekfac_ci_lo=0.140,
    ekfac_ci_hi=0.208, ekfac_n_subsets=50, train_loss=2.61, delta_l1=0.0509,
    delta_l2=0.0528, bank_dir="/mnt/ssd-2/lucia-adam-shampoo/epsroot0_4k_bank",
    code_commit="b3790ba9", reusable="bank",
    notes="MAGIC not run at eps_root=0 on this config; read-only volume.")
add(GPT2_FT, run_id="sm_adam_eps0_8k", status="partial", n_docs=8000, eps_root=0,
    metasmoothness=0.615, ms_shuffle="rep", ekfac_lds=0.1410, ekfac_ci_lo=0.113,
    ekfac_ci_hi=0.169, ekfac_n_subsets=50, train_loss=2.61, delta_l1=0.0646,
    delta_l2=0.0668, reusable="bank", notes="MAGIC not run.")
add(GPT2_FT, run_id="sm_adam_eps0_16k", status="partial", n_docs=16000, eps_root=0,
    metasmoothness=0.437, ms_shuffle="rep", ekfac_lds=0.1097, ekfac_ci_lo=0.084,
    ekfac_ci_hi=0.136, ekfac_n_subsets=50, train_loss=2.65, delta_l1=0.0855,
    delta_l2=0.0872, reusable="bank",
    notes="Lowest-metasmoothness GPT-2 ft cell. MAGIC not run.")
add(GPT2_FT, run_id="sm_adam_eps0_32k", status="planned", n_docs=32000, eps_root=0,
    notes="Completes the eps0 N-sweep; the eps0 column stops at 16k.")

# eps_root fine sweep — metasmoothness only
add(GPT2_FT, run_id="sm_adam_eps1e7_4k", status="partial", n_docs=4000, eps_root=1e-7,
    metasmoothness=0.978, ms_shuffle="rep", reusable="ms_only", notes="ms only.")
add(GPT2_FT, run_id="sm_adam_eps1e9_4k", status="partial", n_docs=4000, eps_root=1e-9,
    metasmoothness=0.907, ms_shuffle="rep", reusable="ms_only",
    notes="ms only; non-monotone vs 1e-8 (0.876) — single direction_seed, h=0.1 noise.")

# epochs axis at fixed N
add(GPT2_FT, run_id="sm_adam_eps1e6_4k_ep4", status="partial", n_docs=4000, eps_root=1e-6,
    num_epochs=4, metasmoothness=0.9989, ms_shuffle="per_epoch", train_loss=3.18,
    delta_l1=0.0016, delta_l2=0.0021, reusable="ms_only")
add(GPT2_FT, run_id="sm_adam_eps0_4k_ep4", status="partial", n_docs=4000, eps_root=0,
    num_epochs=4, metasmoothness=0.6836, ms_shuffle="per_epoch", train_loss=2.61,
    delta_l1=0.0509, delta_l2=0.0528, reusable="ms_only",
    notes="Data fixed, steps 125->250: ms 0.766 -> 0.684.")

# dropout-disabled twin
add(GPT2_FT, run_id="sm_adam_eps1e8_4k_drop0", status="done", n_docs=4000, eps_root=1e-8,
    dropout=0.0, metasmoothness=0.8755, ms_shuffle="per_epoch", ekfac_lds=0.3203,
    ekfac_n_subsets=50, magic_lds=0.18, magic_n_queries=20, train_loss=3.01,
    delta_l1=0.008, delta_l2=0.009,
    run_dir="/mnt/ssd-2/lucia/muon4k/magicroll_eps1e8_drop0", reusable="bank+scores",
    notes="MAGIC leg VOID as a dropout test: scores bit-identical to sm_adam_eps1e8_4k "
          "because train_mode=false made dropout inert in BOTH arms.")

# batch-size axis at FIXED steps=125 (epochs = bs/32)
add(GPT2_FT, run_id="sm_adam_eps1e8_4k_bs32", status="done", n_docs=4000, eps_root=1e-8,
    batch_size=32, num_epochs=1, metasmoothness=0.837, ms_shuffle="", ekfac_lds=0.1781,
    ekfac_n_subsets=50, magic_lds=0.05, magic_ci_lo=-0.054, magic_ci_hi=0.145,
    magic_n_queries=20, train_loss=3.07, delta_l1=0.0087, delta_l2=0.0100,
    run_dir="/mnt/ssd-2/lucia/muon4k/magicroll_bs32", reusable="bank+scores",
    notes="epochs=1 so shuffle-agnostic. MAGIC CI spans zero.")
add(GPT2_FT, run_id="sm_adam_eps1e8_4k_bs128", status="done", n_docs=4000, eps_root=1e-8,
    batch_size=128, num_epochs=4, metasmoothness=0.9822, ms_shuffle="per_epoch",
    ekfac_lds=0.3369, ekfac_n_subsets=50, magic_lds=0.43, magic_n_queries=20,
    train_loss=2.98, delta_l1=0.0074, delta_l2=0.0087,
    run_dir="/mnt/ssd-2/lucia/muon4k/magicroll_bs128", reusable="bank+scores")
add(GPT2_FT, run_id="sm_adam_eps1e8_4k_bs256", status="partial", n_docs=4000, eps_root=1e-8,
    batch_size=256, num_epochs=8, metasmoothness=0.9984, ms_shuffle="per_epoch",
    reusable="ms_only",
    notes="LDS infeasible without grad_accum: fp32 lm_head logits for 256x512 tokens ~26 GB "
          "OOMs the 47.5 GiB A40s. grad_accum_steps now exists -> LDS is unblocked.")
add(GPT2_FT, run_id="sm_adam_eps1e8_4k_bs16", status="partial", n_docs=4000, eps_root=1e-8,
    batch_size=16, num_epochs=2, metasmoothness=0.500, ms_shuffle="", reusable="ms_only",
    notes="bs16 collapses adam metasmoothness (0.876 at bs64 -> 0.500); muon barely moves.")

# weight-decay axis (metasmoothness only)
for wd, ms in [(0.0, 0.867), (0.1, 0.877), (0.3, 0.862)]:
    add(GPT2_FT, run_id=f"sm_adam_eps1e8_4k_wd{wd}", status="partial", n_docs=4000,
        eps_root=1e-8, weight_decay=wd, metasmoothness=ms, ms_shuffle="",
        reusable="ms_only", notes="Weight decay is a null on ms over 0-0.3 (baseline 0.876).")

# output logit scale
add(GPT2_FT, run_id="sm_adam_eps1e8_4k_scale0.5", status="partial", n_docs=4000,
    eps_root=1e-8, logit_scale=0.5, metasmoothness=0.840, ms_shuffle="", ekfac_lds=0.220,
    ekfac_n_subsets=50, train_loss=3.10, delta_l1=0.013, delta_l2=0.016, reusable="bank",
    notes="NOT a clean ms isolate: lowers ms and LDS but RAISES delta_l2.")
add(GPT2_FT, run_id="sm_adam_eps1e8_4k_scale0.25", status="partial", n_docs=4000,
    eps_root=1e-8, logit_scale=0.25, metasmoothness=0.609, ms_shuffle="", ekfac_lds=0.169,
    ekfac_n_subsets=50, train_loss=3.36, delta_l1=0.016, delta_l2=0.019, reusable="bank")
# gradient clipping
add(GPT2_FT, run_id="sm_adam_eps1e8_4k_clip1.0", status="partial", n_docs=4000,
    eps_root=1e-8, max_grad_norm=1.0, metasmoothness=0.876, ms_shuffle="", ekfac_lds=0.307,
    ekfac_n_subsets=50, train_loss=3.03, delta_l1=0.006, delta_l2=0.007, reusable="bank",
    notes="No-op at these settings: grad norms rarely exceed 1.0.")

# =====================================================================================
# 2. GPT-2 fine-tune, muon
# =====================================================================================
MUON = dict(GPT2_FT, optimizer="muon", lr=5e-5, num_epochs=4)
add(MUON, run_id="sm_muon_eps1e6_5e5_4k", status="done", n_docs=4000, eps_root=1e-6,
    metasmoothness=0.997, ms_shuffle="rep", ekfac_lds=0.4738, ekfac_ci_lo=0.432,
    ekfac_ci_hi=0.513, ekfac_n_subsets=50, magic_lds=0.76, magic_n_queries=20,
    train_loss=3.08, delta_l1=0.0057, delta_l2=0.0061,
    bank_dir="/mnt/ssd-2/lucia/muon4k/run/N4k", reusable="bank+scores")
add(MUON, run_id="sm_muon_eps0_5e5_4k", status="partial", n_docs=4000, eps_root=0,
    metasmoothness=0.996, ms_shuffle="rep", ekfac_lds=0.4683, ekfac_ci_lo=0.427,
    ekfac_ci_hi=0.508, ekfac_n_subsets=50, train_loss=3.08, delta_l1=0.0057,
    delta_l2=0.0061, bank_dir="/mnt/ssd-2/lucia/muon4k/run_eps0_5e-5/N4k", reusable="bank",
    notes="MAGIC at eps0 is 98.4% finite on a 1-query spotcheck; no LDS.")
add(MUON, run_id="sm_muon_eps1e8_5e5_4k", status="partial", n_docs=4000, eps_root=1e-8,
    metasmoothness=0.9961, ms_shuffle="per_epoch", reusable="ms_only")
add(MUON, run_id="sm_muon_eps1e6_1e4_4k", status="partial", n_docs=4000, eps_root=1e-6,
    lr=1e-4, metasmoothness=0.9930, ms_shuffle="per_epoch", ekfac_lds=0.4514,
    ekfac_ci_lo=0.416, ekfac_ci_hi=0.486, ekfac_n_subsets=50, train_loss=2.94,
    delta_l1=0.0110, delta_l2=0.0118, bank_dir="/mnt/ssd-2/lucia/muon4k/run_1e-4/N4k",
    reusable="bank")
add(MUON, run_id="sm_muon_eps0_1e4_4k", status="partial", n_docs=4000, eps_root=0, lr=1e-4,
    metasmoothness=0.9931, ms_shuffle="per_epoch", ekfac_lds=0.4544, ekfac_ci_lo=0.419,
    ekfac_ci_hi=0.489, ekfac_n_subsets=50, train_loss=2.94, delta_l1=0.0110,
    delta_l2=0.0118, bank_dir="/mnt/ssd-2/lucia/muon4k/run_eps0_1e-4/N4k", reusable="bank")
# muon N x eps_root metasmoothness sweep (no banks built)
for n, ms0, ms6 in [(8000, 0.9956, 0.9957), (16000, 0.9951, 0.9952), (32000, "", 0.9947)]:
    if ms0 != "":
        add(MUON, run_id=f"sm_muon_eps0_5e5_{n//1000}k", status="partial", n_docs=n,
            eps_root=0, metasmoothness=ms0, ms_shuffle="per_epoch",
            run_dir="/mnt/ssd-2/lucia/muon_ms_steps/eps0", reusable="ms_only",
            notes="ms only; no leave-k-out bank.")
    add(MUON, run_id=f"sm_muon_eps1e6_5e5_{n//1000}k", status="partial", n_docs=n,
        eps_root=1e-6, metasmoothness=ms6, ms_shuffle="per_epoch",
        run_dir=f"/mnt/ssd-2/lucia/muon_ms_steps/msmuon_{n//1000}k", reusable="ms_only",
        notes="ms only; no leave-k-out bank. Muon is flat on both axes (0.9965->0.9947 over 8x steps).")
add(MUON, run_id="sm_muon_eps0_5e5_32k", status="planned", n_docs=32000, eps_root=0,
    notes="Cancelled mid-run to free GPUs. Resume: run_muon_ms_eps.sh 0 eps0 32")
add(MUON, run_id="sm_muon_eps0_5e5_4k_bs16", status="partial", n_docs=4000, eps_root=0,
    batch_size=16, metasmoothness=0.9932, ms_shuffle="per_epoch", reusable="ms_only",
    notes="Batch size does not un-saturate muon (0.996 -> 0.9932); adam collapses to 0.500.")

# =====================================================================================
# 3. Per-epoch-shuffle replication of the headline grid (shuffle is a parameter, so these
#    are separate experiments with their own EK-FAC LDS).
# =====================================================================================
PEREP = [
    ("sm_adam_eps0_16k", 16000, "adamw", 8e-4, 0, 64, 2, 0.4269, 0.1186, 0.074, 0.164, 0.0938, 0.0964),
    ("sm_adam_eps0_8k", 8000, "adamw", 8e-4, 0, 64, 2, 0.6226, 0.1555, 0.121, 0.190, 0.0714, 0.0742),
    ("sm_adam_eps0_4k", 4000, "adamw", 8e-4, 0, 64, 2, 0.7724, 0.1540, 0.106, 0.202, 0.0565, 0.0587),
    ("sm_adam_eps1e10_4k", 4000, "adamw", 8e-4, 1e-10, 64, 2, 0.7883, 0.2020, 0.164, 0.240, 0.0272, 0.0283),
    ("sm_adam_eps1e8_4k", 4000, "adamw", 8e-4, 1e-8, 64, 2, 0.8755, 0.3095, 0.269, 0.351, 0.0068, 0.0084),
    ("sm_adam_eps1e8_4k_bs128", 4000, "adamw", 8e-4, 1e-8, 128, 4, 0.9822, 0.3576, 0.317, 0.397, 0.0065, 0.0080),
    ("sm_adam_eps1e6_8k", 8000, "adamw", 8e-4, 1e-6, 64, 2, 0.9786, 0.2950, 0.253, 0.338, 0.0024, 0.0028),
    ("sm_adam_eps1e6_4k", 4000, "adamw", 8e-4, 1e-6, 64, 2, 0.9952, 0.3076, 0.268, 0.346, 0.0015, 0.0021),
    ("sm_muon_eps0_5e5_4k", 4000, "muon", 5e-5, 0, 64, 4, 0.9960, 0.4648, 0.422, 0.504, 0.0053, 0.0061),
    ("sm_muon_eps1e6_5e5_4k", 4000, "muon", 5e-5, 1e-6, 64, 4, 0.9962, 0.4630, 0.420, 0.503, 0.0053, 0.0061),
]
for rid, n, opt, lr, eps, bs, ep, ms, lds, lo, hi, l1, l2 in PEREP:
    add(GPT2_FT, run_id=f"{rid}_perepoch", status="partial", n_docs=n, optimizer=opt, lr=lr,
        eps_root=eps, batch_size=bs, num_epochs=ep, shuffle="per_epoch",
        metasmoothness=ms, ms_shuffle="per_epoch", ekfac_lds=lds, ekfac_ci_lo=lo,
        ekfac_ci_hi=hi, ekfac_n_subsets=50, delta_l1=l1, delta_l2=l2,
        run_dir="/mnt/ssd-1/lucia/perepoch", reusable="bank",
        notes="Per-epoch-shuffle replication. Both ms and EK-FAC LDS are invariant to the "
              "shuffle change (rep point inside all 11 CIs). MAGIC not re-run.")

# =====================================================================================
# 4. GPT-2 fine-tune on WikiText
# =====================================================================================
add(WIKITEXT, run_id="wt_adam_eps1e6_bs64_lotus", status="done", eps_root=1e-6,
    metasmoothness=0.998, ms_shuffle="rep", magic_lds=0.9681, magic_n_queries=50,
    ekfac_lds=0.2588, ekfac_n_subsets=400, n_subsets=400, n_queries=50, train_loss=3.06,
    delta_l1=0.0033, delta_l2=0.0040, run_dir="runs/lotus_final_q01_50",
    reusable="bank+scores",
    notes="EK-FAC variant=docspace. chunk_length is not a factor (512 vs 0 reproduces 0.9681).")
add(WIKITEXT, run_id="wt_adam_eps0_bs64", status="partial", eps_root=0,
    metasmoothness=0.609, ms_shuffle="rep", ekfac_lds=-0.0109, ekfac_n_subsets=50,
    n_queries=50, train_loss=1.86, delta_l1=0.0834, delta_l2=0.0848,
    run_dir="runs/epsroot0_bank", reusable="bank",
    notes="EK-FAC variant=allium-0. MAGIC spotcheck returned NaN (n=1).")
# eps1e-8 batch-size sweep (the MAGIC-paper regime)
WT_BS = [(64, 288, 0.8947, 0.169, 0.02, 0.41, 4, 0.0121, 0.0144),
         (128, 144, "", 0.483, 0.28, 0.69, 5, "", ""),
         (192, 96, "", 0.644, 0.54, 0.75, 5, "", ""),
         (224, 83, "", 0.407, 0.22, 0.62, 5, "", ""),
         (256, 72, "", 0.9519, 0.9435, 0.9592, 50, "", "")]
for bs, st, ms, lds, lo, hi, nq, l1, l2 in WT_BS:
    add(WIKITEXT, run_id=f"wt_adam_eps1e8_bs{bs}", status="partial", eps_root=1e-8,
        batch_size=bs, steps=st, grad_accum_steps=2 if bs >= 128 else 1,
        metasmoothness=ms, ms_shuffle="rep" if ms != "" else "",
        magic_lds=lds, magic_ci_lo=lo, magic_ci_hi=hi, magic_n_queries=nq,
        n_subsets=100 if bs == 256 else 30, delta_l1=l1, delta_l2=l2,
        code_commit="37d7b386", run_dir="experiments/batchsize_eps1e8", reusable="bank+scores",
        notes=("Definitive N=100/m=50. Committed as examples/magic/gpt2_wikitext.yaml."
               if bs == 256 else
               "metasmoothness not yet measured; EK-FAC not run." if ms == "" else
               "Measured high-ms/low-MAGIC point (ms 0.89, MAGIC 0.17)."))
# dropout actually active
for tag, lds, nsub, note in [
        ("s15", -0.2286, 15, "15 fixed 46-doc subsets, n=1 query; 95% CI approx [-0.7, +0.35]."),
        ("ragged", 0.1862, 99, "99 ragged subsets (2-47 docs), n=1 query.")]:
    add(WIKITEXT, run_id=f"wt_adam_eps1e6_dropout_{tag}", status="partial", eps_root=1e-6,
        dropout=0.1, train_mode=True, magic_lds=lds, magic_n_queries=1, n_subsets=nsub,
        train_loss=3.00, run_dir=f"runs/gpt2_wikitext_dropout_{tag}", reusable="bank+scores",
        notes="Dropout ACTIVE (train_mode=true). Collapses MAGIC vs 0.9681 without. " + note)
# BASELINE_LDS banks (100 subsets, 50 queries, eps1e-8)
BASE = [("adamw", 8e-4, 2.97, 0.5087, 0.3726, "gpt2_wikitext_bank"),
        ("muon", 8e-4, 3.61, "", 0.0643, "gpt2_muon_wikitext_bank_lr8e-4"),
        ("muon", 2e-4, 2.95, "", 0.3804, "gpt2_muon_wikitext_bank_lr2e-4"),
        ("muon", 8e-5, 2.87, 0.8575, 0.4960, "gpt2_muon_wikitext_bank")]
for opt, lr, loss, magic, ekfac, d in BASE:
    add(WIKITEXT, run_id=f"wt_{opt}_eps1e8_lr{lr:g}_bank100", status="partial",
        optimizer=opt, lr=lr, eps_root=1e-8, n_subsets=100, n_queries=50,
        magic_lds=magic, magic_n_queries=50 if magic != "" else "", ekfac_lds=ekfac,
        ekfac_n_subsets=100, train_loss=loss, run_dir=f"runs/{d}", reusable="bank+scores",
        source_doc="BASELINE_LDS.md",
        notes="metasmoothness NOT measured. NB: same nominal config as wt_adam_eps1e8_bs64 "
              "but MAGIC 0.51 vs 0.17 — different bank/estimator (100 subsets x 50 queries "
              "vs 30 x 5). Reconcile before citing either." if opt == "adamw" else
              "metasmoothness NOT measured. lr8e-4 is undertrained (loss 3.61).")

# =====================================================================================
# 5. scaling_magic — GPT-2, SmolLM2 16k, eps_root 1e-17, bs256 (2026-08-07)
# =====================================================================================
for opt, lds, lo, hi in [("adamw", 0.9333, 0.9186, 0.9448), ("muon", 0.8470, 0.8274, 0.8685)]:
    add(GPT2_FT, run_id=f"sm_{opt}_eps1e17_16k_bs256", status="partial", n_docs=16000,
        optimizer=opt, lr=2e-4, eps_root=1e-17, batch_size=256, grad_accum_steps=16,
        num_epochs=2, shuffle="per_epoch", magic_lds=lds, magic_ci_lo=lo, magic_ci_hi=hi,
        magic_n_queries=20, n_subsets=100, n_queries=20,
        run_dir=f"/mnt/ssd-2/lucia/s16k_{opt}", bank_dir=f"/mnt/ssd-2/lucia/s16k_{opt}/merged",
        code_commit="docs-4", reusable="bank",
        source_doc="examples/scaling_magic/LDS_RESULTS.md",
        notes="Paired diff adamw-muon = +0.0863 [+0.0670, +0.1052], 19/20 per-query wins; "
              "identical subset lists both optimizers. metasmoothness and EK-FAC NOT measured. "
              "Base-training checkpoints DELETED; 100 retrained models kept, so EK-FAC can be "
              "scored against this bank without retraining." +
              (" adamw scores rebuilt from per-query .pt files (padded-query bug on docs-4)."
               if opt == "adamw" else ""))

# =====================================================================================
# 6. OLMo2 from-scratch (pre-training proxy; different model, kept for the ms endpoint)
# =====================================================================================
add(OLMO2, run_id="olmo2_muon_16k_full", status="done", n_docs=16000, shuffle="rep",
    metasmoothness=0.010, ms_shuffle="rep", ekfac_lds=0.0175, ekfac_ci_lo=-0.036,
    ekfac_ci_hi=0.071, ekfac_n_subsets=50, n_queries=28, train_loss=2.92, delta_l1=4.56,
    delta_l2=4.10, run_dir="/mnt/ssd-2/lucia/scratch_olmo", reusable="bank",
    notes="Extreme low-ms endpoint. Per-epoch ms is -0.000 (dead endpoint either way).")
add(OLMO2, run_id="olmo2_muon_16k_tail083", status="partial", n_docs=16000,
    attr_window_frac=0.833, metasmoothness=0.984, ms_shuffle="per_epoch", ekfac_lds=0.161,
    ekfac_ci_lo=0.123, ekfac_ci_hi=0.198, ekfac_n_subsets=50, n_queries=50,
    train_loss=3.23, run_dir="runs/tail_bank_083_full", code_commit="5833a9b3",
    reusable="bank",
    notes="MAIN RESULT: attributing only the last epoch makes pre-training scoreable "
          "(9x gain, disjoint CIs) at the model's full loss. MAGIC over the tail was "
          "in progress and is NOT recorded here.")
for frac, ms in [(0.25, 0.025), (0.5, 0.355), (0.6, 0.669), (0.75, 0.793),
                 (0.896, 0.986), (0.95, 0.993), (0.99, 0.990)]:
    add(OLMO2, run_id=f"olmo2_muon_16k_window{frac}", status="partial", n_docs=16000,
        attr_window_frac=frac, metasmoothness=ms, ms_shuffle="per_epoch", reusable="ms_only",
        notes="Window sweep, ms only. frac>0.833 loses doc coverage (bank invalid).")
for n, st, ms, loss in [(4000, 188, 0.0095, 4.98), (8000, 375, 0.0177, 3.95),
                        (32000, 1500, 0.0051, 3.09)]:
    add(OLMO2, run_id=f"olmo2_muon_{n//1000}k_full", status="partial", n_docs=n, steps=st,
        metasmoothness=ms, ms_shuffle="per_epoch", train_loss=loss, reusable="ms_only",
        notes="Full-run attribution is flat at ~0 across 1.9 nats of loss.")
for knob, ms, loss, kw in [
        ("opt_adamw", 0.647, 6.18, dict(optimizer="adamw", lr=8e-4)),
        ("lr3e-3", 0.019, 2.69, dict(lr=3e-3)),
        ("wd0", 0.003, 3.27, dict(weight_decay=0.0)),
        ("epsroot1e-4", 0.004, 3.34, dict(eps_root=1e-4)),
        ("bs64", 0.005, 4.31, dict(batch_size=64, num_epochs=3)),
        ("bs256", 0.006, 1.34, dict(batch_size=256, num_epochs=12))]:
    add(OLMO2, run_id=f"olmo2_muon_16k_{knob}", status="partial", n_docs=16000,
        metasmoothness=ms, train_loss=loss, ms_shuffle="per_epoch", reusable="ms_only", **kw)

# =====================================================================================
# 7. PLANNED — one-factor deviations from the scaling_magic GPT-2 baseline
#    (GPT-2, SmolLM2 16k, bs256/ga16, ep2 = 125 steps, eps_root 1e-17, lr 2e-4).
#    Chosen as the anchor because it is the most recent config with BOTH optimizers
#    measured and an eps_root that does not raise the fp32 noise floor.
# =====================================================================================
BASE17 = dict(GPT2_FT, n_docs=16000, lr=2e-4, eps_root=1e-17, batch_size=256,
              grad_accum_steps=16, num_epochs=2, shuffle="per_epoch", status="planned",
              n_subsets=100, n_queries=20, source_doc="planned")

# batch size (the headline "promising" axis, 16 -> 256)
for bs in [16, 32, 64, 128]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_bs{bs}", batch_size=bs,
        grad_accum_steps=max(1, bs // 16),
        notes="Batch-size axis at eps_root 1e-17. bs256 already measured (MAGIC 0.9333).")
# tokens / steps
for n in [4000, 8000, 32000]:
    add(BASE17, run_id=f"plan_adam_eps1e17_{n//1000}k_bs256", n_docs=n,
        notes="N axis at eps_root 1e-17. 16k measured (MAGIC 0.9333).")
# optimizer x N
for n in [4000, 8000, 32000]:
    add(BASE17, run_id=f"plan_muon_eps1e17_{n//1000}k_bs256", n_docs=n, optimizer="muon",
        notes="N axis, muon. 16k measured (MAGIC 0.8470).")
# warm start (100 -> 500 steps); baseline warmup is a 0.25 FRACTION, not a step count
for w in [100, 200, 500]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_warmup{w}", warmup=w,
        notes="Warm-start axis. Baseline uses warmup=0.25 (fraction of steps); these are "
              "absolute step counts, so 125-step runs need more epochs or fewer warmup steps.")
# model size
for mdl, prm in [("gpt2-medium", 355), ("gpt2-large", 774)]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_{mdl}", model=mdl, n_params_m=prm,
        notes="Model-size axis. Needs a memory plan: MAGIC backward is one reverse pass per "
              "query and scales with params.")
# checkpoint averaging (Louis's result)
for k in [4, 8]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_ckptavg{k}", ckpt_avg_k=k,
        notes="Average the query loss over the k near-final checkpoints. Needs the base-training "
              "checkpoints kept (cleanup_ckpts: false).")
# architecture knobs — the 'not sure' list; needs a GPT-2-like custom model
for mod in ["qk_norm", "preact_layernorm", "preact_batchnorm"]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_{mod}", arch_mod=mod, model="gpt2_custom",
        notes="Requires a GPT-2-like custom model; no such run exists yet. Pair with an "
              "arch_mod=none control on the SAME custom model, not with stock gpt2.")
add(BASE17, run_id="plan_adam_eps1e17_16k_arch_control", arch_mod="none", model="gpt2_custom",
    notes="Control for the arch_mod rows: custom model, no modification.")
# logit scale / weight decay / clipping at the new baseline (only measured at 4k/eps1e-8)
for s in [0.5, 0.25]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_scale{s}", logit_scale=s,
        notes="Logit scale moved ms 0.876->0.609 at 4k/eps1e-8 but also moved delta_l2.")
for wd in [0.0, 0.1]:
    add(BASE17, run_id=f"plan_adam_eps1e17_16k_wd{wd}", weight_decay=wd,
        notes="Weight decay was a null on ms at 4k/eps1e-8.")
add(BASE17, run_id="plan_adam_eps1e17_16k_clip1.0", max_grad_norm=1.0,
    notes="Clipping was a no-op at 4k/eps1e-8.")

# Fill-in rows for measurements missing on configs that already have artifacts.
add(GPT2_FT, run_id="fill_sm_adamw_eps1e17_16k_bs256_ms", status="planned", n_docs=16000,
    lr=2e-4, eps_root=1e-17, batch_size=256, grad_accum_steps=16, shuffle="per_epoch",
    source_doc="planned",
    notes="Measure metasmoothness AND EK-FAC LDS for sm_adamw_eps1e17_16k_bs256. The 100-model "
          "bank exists, so EK-FAC needs only scoring + validate --retrained_dir.")
add(GPT2_FT, run_id="fill_sm_muon_eps1e17_16k_bs256_ms", status="planned", n_docs=16000,
    optimizer="muon", lr=2e-4, eps_root=1e-17, batch_size=256, grad_accum_steps=16,
    shuffle="per_epoch", source_doc="planned",
    notes="Measure metasmoothness AND EK-FAC LDS for sm_muon_eps1e17_16k_bs256; bank exists.")


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.csv")
    for r in rows:
        for c in COLUMNS:
            r.setdefault(c, "")
    order = {"done": 0, "partial": 1, "planned": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), r["family"], r["run_id"]))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    n_done = sum(1 for r in rows if r["status"] == "done")
    n_part = sum(1 for r in rows if r["status"] == "partial")
    n_plan = sum(1 for r in rows if r["status"] == "planned")
    print(f"wrote {out}: {len(rows)} rows ({n_done} done, {n_part} partial, {n_plan} planned)")
    for col in ("metasmoothness", "magic_lds", "ekfac_lds"):
        have = sum(1 for r in rows if str(r[col]) != "")
        print(f"  {col:16} {have:3}/{len(rows)} filled")
    ids = [r["run_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate run_id"


if __name__ == "__main__":
    main()
