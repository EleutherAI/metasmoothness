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
| epochs | 2, fixed across N | matches the LLM post-training norm, which is the target setting: DeepSeek-V3 SFT = 2 epochs, Tulu 3 SFT = 2 epochs (8B and 70B), OLMo 2 SFT (Tulu 3 recipe) = 2 epochs, OLMo 3 = 2 epochs. Token axis = vary N at fixed epochs; steps then scale 31 -> 2000. (Tulu 3 also uses a warmup *ratio* — 0.3 — supporting the fraction convention.) |

Caveat to carry: at bs256/2ep, N=4k is only 32 steps (8 warmup). The 4k point stays on the axis
but short-run effects are confounded with small-N there; 8k (62 steps) is the smallest dataset size to lean on.

## Optimizer

| control | value | why |
|---|---|---|
| optimizers | adamw and muon (each is the control for its own family) | both target axes |
| lr | **2e-4 for both** | held-out sweep at the anchor (below); interior optimum for BOTH optimizers, so no per-optimizer split is needed |
| schedule | polynomial, `lr_start 1e-6`, `lr_end = lr/10`, `warmup_steps 0.25` (fraction) | fraction keeps the schedule self-similar across N — required for token scaling; `warmup >= 1` would be absolute steps (code: `LRScheduleConfig`) |
| adam_beta1 / beta2 | 0.95 / 0.975 | code defaults; every measured point uses them (MAGIC-paper lineage) |
| adam_eps | 1e-8 | code default |
| eps_root | **1e-17, pinned in every yaml** | numerically this is standard AdamW (the sqrt noise floor swamps it), which is what the paper should study. The code default 1e-8 descends from the MAGIC paper's setup (~4k docs / 4 epochs / adam), where damping made attribution work at small batch; we make **no claim** that MAGIC works in that regime — our main setting (bs256) demonstrably works at effectively-zero eps_root. Never rely on the code default. Optional cheap control: an eps_root=0 twin of the anchor to demonstrate 1e-17 vs 0 is a null (prior evidence: muon MAGIC paired diff -0.0005 [-0.0093, +0.0078]) |
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

**Scope of this tuning: N=16k only.** The optimum is not assumed to transfer along any axis that
changes the optimization problem — there is already a hint it drifts (at bs256/1ep/32k, 8e-4 beat
2e-3 and 4e-3 on held-out; lower lrs were untested there). Transfer is *verified, never assumed*
— see the tuning protocol below.

## Tuning protocol (every experiment must be a well-tuned config)

The experiments are only meaningful on fairly good training configs, so tuning is part of the
protocol, not an afterthought:

1. **Selection metric is always heldout_4k CE** (`scripts/heldout_eval.py`), never train loss —
   the train-loss optimum has been measured to generalise worse than untrained GPT-2.
2. **Before any attribution run on a config whose optimization problem differs from the anchor**
   (different N, batch size, epochs, optimizer, model size, or architecture), run a 3-point lr
   mini-sweep {0.5x, 1x, 2x around the incumbent} — train-only, no banks, so it costs minutes at
   small N and a few hours at 256k. If an endpoint wins, extend one octave and re-check. Freeze
   the winner, then build the bank at that lr.
3. **Exempt** (eval-side or provably lr-neutral): checkpoint averaging, subset/estimator
   settings, attribution-window changes on an existing run.
4. **Record `heldout_loss` in experiments.csv for every row** and require it to beat untrained
   GPT-2 (3.4981) by a clear margin; a row whose model is at or worse than untrained is flagged,
   not plotted.
5. If the mini-sweep moves lr off the anchor value, the row's lr column records the tuned value —
   the ablation is then "axis + retuned lr", which is the standard tuned-baseline convention (an
   optimizer or batch-size comparison at a fixed, mistuned lr would be the real confound).


**What may set a sweep centre (clarified 2026-08-25).** An *established trend* may be
used: longer training preferring a lower learning rate is the example that prompted
this, and the square-root batch rule is another. What may not is a single measurement
lying around — one run's number is where mistakes come from, since at n=1 nothing
separates a real effect from noise or a misconfigured run. The endpoint-extension rule
stays the mechanism for correcting a centre that turns out wrong; applying a known
trend up front is that same correction made before paying for it rather than after.

Applied to the step-scaling axis: the optimum halved as steps doubled (1000 and 2000
steps chose 5e-5, 4000 chose 2.5e-5 at its low endpoint), so the 128k grid was centred
at 2.5e-5 and 256k at 1.25e-5 instead of letting both land on an endpoint and paying
for a 10h and a 20h extension run.

## Training mechanics

| control | value | why |
|---|---|---|
| global batch | 256 | ablation axis; 256 is the anchor (the regime where MAGIC works) |
| micro-batch | a memory knob, not a control — record `grad_accum_steps` per run | micro-batch/ga does not affect results: the accumulation rescales each micro-loss by its token share, reproducing the full-batch gradient up to float associativity, and `test_magic_grad_accum_weight_grads_match` + `test_accumulate_grads_matches_full_batch` verify it (6/6 pass, checked at bergson commit 8ce0cd76). The historical "ga rescales raw MAGIC magnitudes ~0.68x" observation was a property of the retired 0.10.1 scoring code (`37d7b386`) — it applies when interpreting scores recorded on that code, not to new runs. Within one config, bank retrains and the base run keep the same ga (bit-exact reuse requires a matched config) |
| shuffle | per-epoch (`1e6eea7f`+), training AND bank AND ms probe | admission policy; builder asserts it |
| seed | 42 (training and subset draw) | subset lists then match across optimizers, enabling paired comparisons |
| precision | fp32, `use_tf32_matmuls: false` | metasmoothness is ill-conditioned near zero; tf32 kept off as a precaution |
| checkpoints | `cleanup_ckpts: false`, `save_models: true`, `save_optimizer_state: last`, keep base-training checkpoints | checkpoint averaging and window analyses need the checkpoints; TrackStar-Adam and SOURCE-Adam need the final optimizer state. ~28 GB/run; budget it |

## Attribution / estimator (identical in every run)

| control | value | why |
|---|---|---|
| subsets | 100 random leave-out @ **1% fraction** | fraction (not fixed count) keeps the perturbation proportional across N; 1% of 4k = 40 docs, of 256k = 2560 |
| queries | `query_20.hf`, per-query MAGIC (`query_method: none`) | one reverse pass per query (budget 20x); `mean` cannot produce per-query LDS. Escalation (D6): if a config's 95% CI half-width exceeds 0.025, re-score with `query_50.hf` (scoring-only, same bank) |
| LDS | mean per-query Spearman; 10k-resample bootstrap; optimizer contrasts paired over queries | the established pipeline |
| metasmoothness | `fd_step 0.1`, `direction_seed 0`; confirm any cell that is surprising or < 0.9 at `direction_seed 1`; always record `total_movement_l1` | ms < ~0.02 carries no information (sign-statistic noise) |
| MAGIC code | record `code_commit` on every row | a correctness bug in MAGIC would invalidate all results regardless of version bookkeeping, so no commit pin is required; the recorded commit exists so that any suspicious cross-row difference can be checked against a code change after the fact (one past fix moved a config 0.37 -> 0.17) |
| eval batch size | eval/validate `batch_size` is set independently of the training batch; keep it at 32 or less | per-document query-loss evaluation materialises fp32 logits of batch x 512 x vocab: ~26 GB at batch 256 (measured OOM on 47.5 GB A40s), ~52 GB at batch 512. Training batch is the science; eval batch is only a memory knob |

## Per-axis deviation table

Every ablation changes exactly one row of the tables above:

| axis | varies | everything else |
|---|---|---|
| batch size | bs 16..256, both optimizers; ga recorded per run (memory knob) | fixed; steps co-vary with bs at fixed epochs — the step-count arms below provide the deconfound |
| tokens | N = 4k..64k (nested) | fixed, incl. epochs=2 |
| step count | epochs 2 -> 4 at fixed bs; and bs 256 -> 512 at fixed epochs | fixed; the pair separates steps from batch |
| optimizer | adamw vs muon | fixed, incl. lr 2e-4 (verified optimal for both) |
| model size | gpt2-medium / gpt2-large | fixed; MAGIC cost scales with params |
| ckpt averaging | `ckpt_avg_k` 1 -> 4 (query gradient averaged over the last 4 checkpoints) | fixed; eval-side only, same trained model |
| arch (QK-norm, pre-act norm) | one mod on `gpt2_custom` | compare only against the `gpt2_custom` no-mod control row |
| logit scale / wd / clip | one knob | fixed |

## Not controls (recorded, not fixed)

`nproc_per_node` — statistically neutral (ga absorbs it), but NOT neutral for bit-exact
reuse: retraining the adam_eps1e8_4k bank base on its own commit at nproc 4 vs the bank's
nproc 8 diverges by 7.7e-3 max-abs (reduction order compounds over 125 steps). Bit-exact
recomputation (MAGIC replay, D9-style retrains) must match the original run's world size;
record nproc with every bank. Also
CephFS paths, wandb. `n_docs`-dependent `steps` is derived, never set independently.
