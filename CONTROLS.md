# Control hyperparameters for the metasmoothness paper

The fixed, non-varied values every ablation deviates from by **exactly one factor**.
Confirmed 2026-08-20 by (a) reading the trainer defaults on the current branch, (b) a held-out
lr sweep at the anchor config, and (c) dataset audits (nesting, disjointness, lengths).
Change nothing here without re-running the affected planned rows in `experiments.csv`.

## The anchor

GPT-2 (stock, 124M) fine-tuned on the SmolLM2 512-token chunk pipeline,
**N=16k docs, bs256, 2 epochs = 125 steps, lr 2e-4, eps_root 1e-17** — the config whose MAGIC
LDS is measured for both optimizers (adamw 0.9333 [0.9186, 0.9448], muon 0.8470
[0.8274, 0.8685], paired diff +0.0863, 19/20 query wins).

## Model

| control | value | why |
|---|---|---|
| model | `gpt2` (HF stock, 124M) | every measured anchor uses it; `gpt2_custom` rows get their own control |
| weight init | pretrained | fine-tuning regime; from-scratch is the separate `olmo2_scratch` family |
| logit scale | 1.0 | ablation axis |
| dropout | **configured 0.0 explicitly** | active dropout collapses MAGIC (0.97 -> ~0, WikiText); relying on `train_mode=false` to neutralise a configured 0.1 invited the cfg/effective ambiguity — new runs pin 0.0 |
| train_mode | false | trainer default; with dropout 0.0 it is also inert by construction |

## Data (token-scaling axis)

| control | value | why |
|---|---|---|
| corpus | SmolLM2 512-token chunks (`EleutherAI/bergson-smollm2-lds-chunks` lineage) | **WikiText is banned from paper runs — it does not scale**; builder asserts `dataset == smollm2` |
| doc length | uniformly 512 tokens, `chunk_length: 0`, no truncation | verified on every set |
| train sets | `train_{4,8,16,32,64,128,256}k` — one nested chain, each a superset of the last | scaling must mean "same docs plus more"; the 64k+ family was rebuilt 2026-08-20 to restore nesting (old 64k shared only ~4k docs with 32k) |
| LDS queries | `query_20.hf` (20 docs), fixed across ALL runs | verified disjoint from every train set incl. 256k; 20 queries gave CI width ~±0.013 at the anchor |
| model-selection set | `heldout_4k.hf` (4000 docs), fixed | verified disjoint from every train set (the 256k conflict is resolved by excluding heldout docs from the rebuilt 256k) — **never select lr or report generalisation on train loss** |
| epochs | 2, fixed across N | token axis = vary N at fixed epochs (OLMo3 uses 2); steps then scale 31 -> 2000 |

Caveat to carry: at bs256/2ep, N=4k is only 31 steps (8 warmup). The 4k rung stays on the axis
but short-run effects are confounded with small-N there; 8k (62 steps) is the smallest rung to
lean on.

## Optimizer

| control | value | why |
|---|---|---|
| optimizers | adamw and muon (each is the control for its own family) | both target axes |
| lr | **2e-4 for both** | held-out sweep at the anchor (below); interior optimum for BOTH optimizers, so no per-optimizer split is needed |
| schedule | polynomial, `lr_start 1e-6`, `lr_end = lr/10`, `warmup_steps 0.25` (fraction) | fraction keeps the schedule self-similar across N — required for token scaling; `warmup >= 1` would be absolute steps (code: `LRScheduleConfig`) |
| adam_beta1 / beta2 | 0.95 / 0.975 | code defaults; every measured point uses them (MAGIC-paper lineage) |
| adam_eps | 1e-8 | code default |
| eps_root | **1e-17, pinned in every yaml** | largest value that does not raise the fp32 noise floor; the current branch's code default is **1e-8** — never rely on the default |
| weight decay | 0.01 | code default; rep-era evidence says a null on ms over 0-0.3 |
| grad clipping | none (`max_grad_norm` unset) | ablation axis; rep-era evidence says a no-op at these settings |

lr selection evidence (held-out `heldout_4k` CE, anchor config, models in
`/mnt/ssd-2/lucia/s16k_lrsweep/`, one factor = lr; untrained gpt2 = 3.4981):

| lr | adamw | muon |
|---|---|---|
| 1e-4 | 3.2592 | 3.2649 |
| **2e-4** | **3.2572** | **3.2570** |
| 4e-4 | 3.2670 | 3.2660 |
| 8e-4 | 3.2990 | 3.3035 |
| 2e-3 | 3.3974 | 3.4198 |

The 2e-4-vs-1e-4 gap (~0.002) is near the seed-noise floor (~0.001), but 2e-4 is best for both
optimizers, the curve is smooth on both sides, and both measured MAGIC anchors sit at 2e-4.

## Training mechanics

| control | value | why |
|---|---|---|
| global batch | 256 | ablation axis; 256 is the anchor (the regime where MAGIC works) |
| micro-batch | **16 per device, held fixed**; `grad_accum_steps = 256 / (16 * nproc)` | ga is rank-preserving (LDS-safe, Spearman 0.9995) but rescales raw MAGIC magnitudes ~0.68x per doubling — so raw scores are comparable only at equal ga; record ga per run |
| shuffle | per-epoch (`1e6eea7f`+), training AND bank AND ms probe | admission policy; builder asserts it |
| seed | 42 (training and subset draw) | subset lists then match across optimizers, enabling paired comparisons |
| precision | fp32, `use_tf32_matmuls: false` | metasmoothness is ill-conditioned near zero; tf32 kept off as a precaution |
| checkpoints | `cleanup_ckpts: false`, `save_models: true`, keep base-training checkpoints | checkpoint-averaging axis and window analyses need them (~28 GB/run; budget it) |

## Attribution / estimator (identical in every run)

| control | value | why |
|---|---|---|
| subsets | 100 random leave-out @ **1% fraction** | fraction (not fixed count) keeps the perturbation proportional across N; 1% of 4k = 40 docs, of 256k = 2560 |
| queries | `query_20.hf`, per-query MAGIC (`query_method: none`) | one reverse pass per query (budget 20x); `mean` cannot produce per-query LDS |
| LDS | mean per-query Spearman; 10k-resample bootstrap; optimizer contrasts paired over queries | the established pipeline |
| metasmoothness | `fd_step 0.1`, `direction_seed 0`; confirm any cell that is surprising or < 0.9 at `direction_seed 1`; always record `total_movement_l1` | ms < ~0.02 carries no information (sign-statistic noise) |
| MAGIC code | record `code_commit`; all values post-`c0f11ba8` | the replay fix moved MAGIC 0.37 -> 0.17 on one config |

## Per-axis deviation table

Every ablation changes exactly one row of the tables above:

| axis | varies | everything else |
|---|---|---|
| batch size | bs 16..256 (+ ga to hold micro-batch 16) | fixed; note steps co-vary with bs at fixed epochs — the bs128 rep-era grid held steps fixed by varying epochs; decide per-plot which deconfound to show |
| tokens | N = 4k..256k (nested) | fixed, incl. epochs=2 |
| optimizer | adamw vs muon | fixed, incl. lr 2e-4 (verified optimal for both) |
| warm start | `warmup_steps` in absolute steps | fixed; label the axis in steps and keep the 0.25-fraction anchor off that plot |
| model size | gpt2-medium / gpt2-large | fixed; MAGIC cost scales with params |
| ckpt averaging | `ckpt_avg_k` 1 -> 4 -> 8 | fixed; eval-side only, same trained model |
| arch (QK-norm, pre-act norm) | one mod on `gpt2_custom` | compare only against the `gpt2_custom` no-mod control row |
| logit scale / wd / clip | one knob | fixed |

## Not controls (recorded, not fixed)

`nproc_per_node` (with micro-batch fixed it only changes wall-clock; ga absorbs it),
CephFS paths, wandb. `n_docs`-dependent `steps` is derived, never set independently.
