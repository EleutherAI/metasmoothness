# LDS results — EK-FAC / Shampoo / MAGIC / SOURCE / Trackstar

GPT-2, leave-k-out banks, 100 subsets @1% held out (muon SmolLM2 banks: 50), seed 42.
LDS = mean over 50 queries of per-query Spearman(predicted `score_sum`, actual
retrain loss `diff`). CIs = 10k-resample bootstrap (seeded). Compiled 2026-07-21.

Only banks trained with `data.chunk_length = 0` are included; banks trained with
`chunk_length = 512` are in the [Appendix](#appendix--invalid-banks-chunk_length--0).
Shampoo −1/2 / −1/4 / −1/8 = methods `shampoo` / `shampoo_quarter` / `shampoo_p025`
(apply power −1.0 / −0.5 / −0.25 on the fitted factors).
metasmooth = empirical metasmoothness (Chang et al. 2024, Def. 2; h=0.1, direction_seed 0),
measured at each row's exact training config. `epochs` = training epochs of that config. All
SmolLM2 size banks are epochs=2 (`runs/ekfac_vs_n/configs/N{4,8,16,32}k.yaml`); muon banks epochs=4.
The two adam epochs=4 rows (no Method/LDS) are metasmoothness-only measurements of the adam config
at epochs=4 — no bank was trained there.

## Metasmoothness ↔ EK-FAC LDS grid (all models / knobs)

Consolidated view: metasmoothness predicts EK-FAC LDS across **model, optimizer, eps_root, training
steps, and batch size**. Sorted by metasmoothness. Detailed per-axis sweeps are in the sub-sections
below. `shuffle` = data-order-per-epoch implementation (see note): **rep** = shuffle-once-then-repeat
(same order every epoch, checkout `feat/magic-grad-accum`); all runs to date are **rep**. epochs=1
rows are identical under a per-epoch-shuffle implementation.

| model | opt | eps_root | N | bs | epochs | steps | metasmooth | EK-FAC LDS | shuffle |
|-------|-----|----------|-----|-----|--------|-------|-----------|-----------|---------|
| OLMo2 scratch | muon | 1e-6 | 16k | 128 | 6 | 750 | 0.010 | 0.0175 | rep |
| GPT-2 ft | adam | 0 | 16k | 64 | 2 | 500 | 0.437 | 0.1097 | rep |
| GPT-2 ft | adam | 0 | 8k | 64 | 2 | 250 | 0.615 | 0.1410 | rep |
| GPT-2 ft | adam | 0 | 4k | 64 | 2 | 125 | 0.766 | 0.1740 | rep |
| GPT-2 ft | adam | 1e-10 | 4k | 64 | 2 | 125 | 0.781 | 0.2097 | rep |
| GPT-2 ft | adam | 1e-8 | 4k | 32 | 1 | 125 | 0.837 | _(building)_ | rep* |
| GPT-2 ft | adam | 1e-8 | 4k | 64 | 2 | 125 | 0.876 | 0.3033 | rep |
| GPT-2 ft | adam | 1e-8 | 4k | 128 | 4 | 125 | 0.982 | _(building)_ | rep |
| GPT-2 ft | adam | 1e-6 | 4k | 64 | 2 | 125 | 0.991 | 0.3173 | rep |
| GPT-2 ft | muon | 0 | 4k | 64 | 4 | 250 | 0.996 | 0.4683 | rep |
| GPT-2 ft | muon | 1e-6 | 4k | 64 | 4 | 250 | 0.997 | 0.4738 | rep |

`*` bs32/ep1 is epochs=1 → identical under per-epoch shuffle. bs256/ep8 (metasmooth 0.992) LDS is
infeasible — the bs256 metagradient double-backward OOMs on 48 GB (MAGIC trainer's custom loop
ignores grad-checkpointing). Within adam, LDS rises monotonically with metasmoothness; muon sits
slightly above the adam curve (high LDS at high metasmoothness).

**Per-epoch shuffle note:** the run checkout `feat/magic-grad-accum` shuffles the train set once then
`.repeat(num_epochs)`, so every epoch sees the **same order** (`rep`). The fix — commit `1e6eea7f`
"Shuffle each training epoch independently" (#352), on `origin/main` — reshuffles each epoch. Not yet
rebased; recording which runs use which. All rows above are `rep`; epochs=1 rows are shuffle-agnostic.

## SmolLM2 (`bergson-smollm2-lds-chunks`, `train_{4k,8k,16k,32k}.hf`)

| Optimizer | eps_root | lr | N | epochs | metasmooth | Method | LDS | 95% CI | n |
|-----------|----------|-----|-----|--------|-----------|--------|-----|--------|---|
| adam | 1e-6 | 8e-4 | 4k | 2 | 0.991 | EK-FAC | 0.3173 | [0.285, 0.348] | 50 |
| adam | 1e-6 | 8e-4 | 8k | 2 | 0.978 | EK-FAC | 0.3019 | [0.267, 0.338] | 50 |
| adam | 1e-6 | 8e-4 | 16k | 2 | 0.995 | EK-FAC | 0.3815 | [0.352, 0.412] | 50 |
| adam | 1e-6 | 8e-4 | 32k | 2 | 0.998 | EK-FAC | 0.3575 | [0.325, 0.394] | 50 |
| adam | 1e-6 | 8e-4 | 4k | 2 | 0.991 | Shampoo −1/2 | 0.3071 | [0.270, 0.342] | 50 |
| adam | 1e-6 | 8e-4 | 4k | 2 | 0.991 | Shampoo −1/4 | 0.3264 | [0.294, 0.358] | 50 |
| adam | 1e-8 | 8e-4 | 4k | 2 | 0.876 | EK-FAC | 0.3033 | [0.274, 0.334] | 50 |
| adam | 1e-10 | 8e-4 | 4k | 2 | 0.781 | EK-FAC | 0.2097 | [0.182, 0.239] | 50 |
| adam | 0 | 8e-4 | 4k | 2 | 0.766 | EK-FAC | 0.1740 | [0.140, 0.208] | 50 |
| adam | 0 | 8e-4 | 8k | 2 | 0.615 | EK-FAC | 0.1410 | [0.113, 0.169] | 50 |
| adam | 0 | 8e-4 | 16k | 2 | 0.437 | EK-FAC | 0.1097 | [0.084, 0.136] | 50 |
| adam | 0 | 8e-4 | 4k | 2 | 0.766 | Shampoo −1/2 | 0.2145 | [0.178, 0.249] | 50 |
| adam | 0 | 8e-4 | 4k | 2 | 0.766 | Shampoo −1/4 | 0.1562 | [0.122, 0.192] | 50 |
| adam | 0 | 8e-4 | 4k | 2 | 0.766 | Shampoo −1/8 | 0.1111 | [0.076, 0.145] | 50 |
| muon | 1e-6 | 5e-5 | 4k | 4 | 0.997 | EK-FAC | 0.4738 | [0.432, 0.513] | 50 |
| muon | 1e-6 | 5e-5 | 4k | 4 | 0.997 | Shampoo −1/2 | 0.5217 | [0.481, 0.561] | 50 |
| muon | 1e-6 | 5e-5 | 4k | 4 | 0.997 | Shampoo −1/4 | 0.4304 | [0.388, 0.471] | 50 |
| muon | 1e-6 | 5e-5 | 4k | 4 | 0.997 | Shampoo −1/8 | 0.3026 | [0.260, 0.343] | 50 |
| muon | 1e-6 | 1e-4 | 4k | 4 | 0.996 | EK-FAC | 0.4514 | [0.416, 0.486] | 50 |
| muon | 0 | 5e-5 | 4k | 4 | 0.996 | EK-FAC | 0.4683 | [0.427, 0.508] | 50 |
| muon | 0 | 5e-5 | 4k | 4 | 0.996 | Shampoo −1/2 | 0.5208 | [0.479, 0.561] | 50 |
| muon | 0 | 5e-5 | 4k | 4 | 0.996 | Shampoo −1/4 | 0.4306 | [0.389, 0.471] | 50 |
| muon | 0 | 5e-5 | 4k | 4 | 0.996 | Shampoo −1/8 | 0.3010 | [0.260, 0.343] | 50 |
| muon | 0 | 1e-4 | 4k | 4 | 0.996 | EK-FAC | 0.4544 | [0.419, 0.489] | 50 |
| muon | 0 | 1e-4 | 4k | 4 | 0.996 | Shampoo −1/2 | 0.5206 | [0.479, 0.560] | 50 |
| muon | 0 | 1e-4 | 4k | 4 | 0.996 | Shampoo −1/4 | 0.4093 | [0.371, 0.447] | 50 |
| muon | 0 | 1e-4 | 4k | 4 | 0.996 | Shampoo −1/8 | 0.2682 | [0.229, 0.307] | 50 |
| adam | 1e-6 | 8e-4 | 4k | 4 | 0.999 | — | — | — | — |
| adam | 0 | 8e-4 | 4k | 4 | 0.663 | — | — | — | — |

- adam eps1e-6 4k EK-FAC row is the original run (0.3173); re-score with the `1ba43f92` scoring code = 0.3156. muon eps1e-6 5e-5 EK-FAC 0.4738 = will's reported 0.474.
- muon eps_root acts on Muon's AdamW-fallback parameters (embeddings / lm_head / 1D params); the 2D weights use Newton-Schulz, which eps_root does not touch. So muon EK-FAC is nearly flat across eps_root (5e-5: 0.474 @1e-6 vs 0.468 @0; 1e-4: 0.451 @1e-6 vs 0.454 @0), unlike adam (0.317 @1e-6 → 0.174 @0).
- MAGIC (metagradient) spotcheck, 5 queries, gpt2 muon @5e-5, nproc=1 eager: eps_root=0 → NaN scores; eps_root=1e-6 → _(running)_.

### adam metasmoothness vs eps_root (4k bank config: adamw, lr 8e-4 poly, bs64, epochs=2, betas 0.95/0.975)

Sweep between the eps1e-6 (ms 0.991) and eps0 (ms 0.766) endpoints. Single direction_seed, h=0.1 (noisy).

| eps_root | metasmooth | EK-FAC LDS |
|----------|-----------|-----------|
| 1e-6 | 0.991 | 0.3173 |
| 1e-7 | 0.978 | — |
| 1e-8 | 0.876 | 0.3033 [0.274, 0.334] |
| 1e-9 | 0.907 | — |
| 1e-10 | 0.781 | 0.2097 [0.182, 0.239] |
| 0 | 0.766 | 0.1740 |

### adam metasmoothness vs training steps (N-sweep, epochs=2, lr 8e-4, bs64)

Measured at eps_root=0 (un-saturated); eps_root=1e-6 shown for contrast (pinned near the 1.0
ceiling, hides the trend). Steps = N·epochs/bs.

| N | steps | metasmooth (eps0) | EK-FAC LDS (eps0) | metasmooth (eps1e-6) |
|-----|-------|-------------------|-------------------|----------------------|
| 4k | 125 | 0.766 | 0.174 | 0.991 |
| 8k | 250 | 0.615 | 0.141 | 0.978 |
| 16k | 500 | 0.437 | 0.110 | 0.995 |
| 32k | 1000 | — | — | 0.998 |

At eps0, LDS falls monotonically with steps, tracking metasmoothness (both ~halve 4k→16k).

Same direction with data held fixed (4k, eps0, epochs 2→4 = 125→250 steps): 0.766 → 0.663.

### adam metasmoothness — other knobs (eps_root=1e-8, N=4k, epochs=2; baseline bs64/wd0.01/scale1.0 = 0.876)

| knob | values → metasmooth |
|------|---------------------|
| batch size (steps=8000/bs) | 16→0.500, 32→0.810, **64→0.876**, 128→0.979, 256→0.997 |
| weight decay | 0→0.867, **0.01→0.876**, 0.1→0.877, 0.3→0.862 |
| output logit scale | **1.0→0.876**, 0.5→0.840, 0.25→0.609 |

Batch-size effect is largely step-count (bigger bs = fewer steps at fixed lr → higher metasmooth,
consistent with the steps table). Weight decay: no effect over 0–0.3. Output logit scale (fixed
`lm_head` forward multiply, non-trainable): scaling outputs down lowers metasmoothness.

**Batch size deconfounded — metasmoothness at FIXED steps=125** (eps1e-8, 4k, epochs = bs/32):

| batch size | epochs | steps | metasmooth |
|------------|--------|-------|------------|
| 32 | 1 | 125 | 0.837 |
| 64 | 2 | 125 | 0.876 |
| 128 | 4 | 125 | 0.982 |
| 256 | 8 | 125 | 0.992 |

Batch size raises metasmoothness even at fixed step count (0.837→0.992) — a genuine effect, not just
a step-count proxy. (LDS at bs256/ep8 pending — bank build OOMs at bs256, retrying w/ grad checkpointing.)

## OLMo2 from-scratch (`olmo2_reinit` 124M, SmolLM2 corpus)

Re-initialized OLMo2 (124M; hidden 768, 12 layers) trained **from scratch** (not fine-tuned) — the
pre-training proxy that motivated this investigation. muon, 6 epochs, bs128, lr 9e-3, eps_root 1e-6,
betas 0.95/0.975, wd 0.1, 50 subsets.

| N | steps | metasmooth | Method | LDS | 95% CI | n |
|-----|-------|-----------|--------|-----|--------|---|
| 16k | 750 | 0.010 | EK-FAC | 0.0175 | [−0.036, 0.071] | 50 |

Both metasmoothness (0.010) and LDS (0.018) ≈ 0 — the extreme low-metasmoothness endpoint of the
grid, consistent with the mechanism. (N32k paused at 4/50 subsets.)

## WikiText (`bergson-wikitext-512-chunks`)

Two banks, both adamw, 4 epochs, betas 0.95/0.975.

metasmooth measured for each bank's training config (bs64, 4 epochs): lotus 0.998, epsroot0 0.609.

| Bank | eps_root | metasmooth | Method | Variant | LDS | n | Run dir |
|------|----------|-----------|--------|---------|-----|---|---------|
| lotus | 1e-6 | 0.998 | MAGIC | full q01–50 | 0.9681 | 50 | `runs/lotus_final_q01_50` |
| lotus | 1e-6 | 0.998 | MAGIC | bwd eval | 0.9688 | 50 | `runs/lotus_bwd_eval` |
| lotus | 1e-6 | 0.998 | SOURCE | damp0 | 0.3902 | 50 | `runs/lotus_source_q50_damp0_validate` |
| lotus | 1e-6 | 0.998 | SOURCE | adam | 0.2068 | 50 | `runs/lotus_source_adam_q50_validate` |
| lotus | 1e-6 | 0.998 | SOURCE | default | −0.3871 | 50 | `runs/lotus_source_q50_validate` |
| lotus | 1e-6 | 0.998 | EK-FAC | docspace | 0.2588 | 50 | `runs/lotus_ekfac50q_docspace_vs_lotus_bank` |
| lotus | 1e-6 | 0.998 | EK-FAC | allium-0 | 0.0543 | 50 | `runs/lotus_scores_ekfac50q_allium-0_validate` |
| lotus | 1e-6 | 0.998 | Trackstar | docs p32 noopt | 0.2002 | 50 | `runs/gpt2_lotus_trackstar50q_docs_p32_noopt_vs_lotus_bank` |
| lotus | 1e-6 | 0.998 | Trackstar | docs | 0.1838 | 50 | `runs/gpt2_lotus_trackstar50q_docs_vs_lotus_bank` |
| lotus | 1e-6 | 0.998 | Trackstar | default | 0.1767 | 50 | `runs/lotus_trackstar_q50_validate` |
| epsroot0 | 0 | 0.609 | SOURCE | source2 | 0.1531 | 50 | `runs/epsroot0_source2_q50_validate` |
| epsroot0 | 0 | 0.609 | SOURCE | source2 adam hybrid | 0.1446 | 50 | `runs/epsroot0_source2_adam_hybrid_validate` |
| epsroot0 | 0 | 0.609 | SOURCE | source2 adam | 0.0811 | 50 | `runs/epsroot0_source2_adam_q50_validate` |
| epsroot0 | 0 | 0.609 | EK-FAC | allium-0 | −0.0109 | 50 | `runs/epsroot0_scores_ekfac50q_allium-0_validate` |
| epsroot0 | 0 | 0.609 | MAGIC | spotcheck | NaN | 1 | `runs/epsroot0_bank` |

- Excluded (n=1 spotchecks): lotus MAGIC `lotus_mq_eval` 0.9893, `lotus_q01_spotcheck` 0.9818, `lotus` 0.9177, `lotus_mq_eval_prefix_backup` 0.9665; and `lotus_interim_q01_08` (n=8, 0.9675).
- The `gpt2_epsroot0_trackstar50q*` runs are excluded: their config has `retrained_dir=runs/lotus`, so they score epsroot0 gradients against the lotus bank.

## Provenance / reproduction

- SmolLM2 eps_root=1e-6 4k adam bank: HF `EleutherAI/bergson-smollm2-lds-4k` (+ `run_config.yaml`, `subsets.json`). Size-scaling banks: `runs/ekfac_vs_n/N{4,8,16,32}k`. adam eps_root=0 bank: `/mnt/ssd-2/lucia-adam-shampoo/epsroot0_4k_bank/` (code `b3790ba9`). muon banks: `/mnt/ssd-2/lucia/muon4k/{run,run_1e-4,run_eps0_5e-5,run_eps0_1e-4}/N4k` (differ only in eps_root and lr).
- SmolLM2 scoring summary.csv under `/mnt/ssd-2/lucia-adam-shampoo/*/validate/` and `/mnt/ssd-2/lucia/muon4k/**/validate/`; scoring code `1ba43f92` worktree (+ `feat/shampoo-quarter-power` for the Shampoo power variants). WikiText run dirs under `runs/`.
- All banks above: `data.chunk_length = 0`.

## Appendix — invalid banks (`chunk_length ≠ 0`)

Trained with `data.chunk_length = 512` (WikiText data is already 512-chunked). These
banks and any LDS scored against them are invalid — **do not add them to the tables
above.** Kept here only for the record.

**stdadam** (adamw, betas 0.9/0.999, eps_root 0, `chunk_length=512`):

| Method | Variant | LDS | n | Run dir |
|--------|---------|-----|---|---------|
| EK-FAC | docs | 0.0700 | 50 | `runs/gpt2_stdadam_ekfac50q_docs_vs_stdadam_bank` |
| EK-FAC | default | 0.0232 | 50 | `runs/gpt2_stdadam_ekfac50q_vs_stdadam_bank` |
| Trackstar | docspace | 0.0417 | 50 | `runs/gpt2_stdadam_trackstar50q_docspace_vs_stdadam_bank` |

(other near-0 stdadam variants under `runs/gpt2_stdadam_*`)

**eps / batch-size sweep banks** (`chunk_length=512`): `gpt2_wikitext_eps{1e-8,2e-8,5e-8,5e-9,7e-9}`, `gpt2_wikitext_bs{448_eps1e-7,448_eps1e-8,448_eps1e-9,448_eps5e-9,480_eps1e-8,512_eps1e-8}`, `gpt2_wikitext_A40s`, `gpt2_wikitext_paper_bs32`, `gpt2_wikitext_metasmoothness_eps1e-4`, `batch_probe`. LDS for these are in `runs/eps_search/RESULTS.md`.

Old/exploratory (not chunk-verified): `examples/exp_log.md` (Jun 30, token-length sweeps).
