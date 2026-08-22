# `experiments.csv` — source of truth for the metasmoothness grid

One row per experiment. Parameter columns first, result columns last, provenance at the end.
**An empty result cell means not-yet-run, not failure.** Regenerate from the row tables in
`build_experiments_csv.py`; edit the script, never the CSV.

```
python build_tuning_csv.py        # -> tuning.csv      (stage 0: hp selection, run FIRST)
python build_experiments_csv.py   # -> experiments.csv (stage 1: the grid itself)
```

**Staging:** `tuning.csv` registers the held-out lr mini-sweeps that must be measured before a
planned experiment's lr is final — one row per (config, lr), grouped by `sweep_group`, with the
`selects_lr_for` column naming the experiments.csv row(s) that take their learning rate from the winner. The 16k anchor sweeps are
already measured (2e-4 for both optimizers); everything else is empty rows to claim. Lowest
heldout_loss in a group wins; an endpoint win adds one octave before freezing. Only the
eval-side ckptavg rows are exempt from gating (CONTROLS protocol rule 3).

## Admission policy (2026-08-20)

Only measurements made under the current implementation are admitted; the builder asserts both
filters on every row.

1. **Per-epoch shuffle everywhere** — training, the leave-k-out retrains, and the metasmoothness
   probe must all reshuffle each epoch (commit `1e6eea7f`, PR #352). Runs on the older
   shuffle-once-then-`.repeat` ("rep") code are excluded. Exception: `num_epochs == 1` runs are
   shuffle-agnostic (one pass over one order) and are admitted as `shuffle=agnostic_1ep`.
2. **No active dropout** — `dropout_effective` must be 0. Most GPT-2 rows *configure* the HF
   default 0.1 but trained with `train_mode=false`, where the trainer calls `model.eval()` and
   the rate is inert; both facts are recorded (`dropout_cfg=0.1, dropout_effective=0.0`). All
   planned rows set `dropout_cfg=0.0` explicitly so the ambiguity cannot recur in new runs.

## What was excluded, and where it still lives

Nothing was deleted — the numbers remain in the narrative docs. They are out of the CSV because
they were measured under the old shuffle (or with dropout active) and are not comparable to
current-code runs.

| excluded set | headline numbers | why | still recorded in |
|---|---|---|---|
| SmolLM2 headline grid, rep-trained banks | MAGIC 0.86 / 0.98 / 0.17 / 0.43 / 0.76 / -0.02; EK-FAC 0.11-0.47 | rep shuffle; MAGIC replays the training order so rep values do not transfer | LDS_RESULTS.md headline grid |
| All WikiText runs (lotus 0.9681, eps0 bank, eps1e-8 bs sweep incl. bs256 0.9519) | MAGIC 0.9681 / 0.9519 | rep shuffle | LDS_RESULTS.md WikiText section |
| BASELINE_LDS 100-subset banks (adamw + 3 muon lrs) | MAGIC 0.51 / 0.86; EK-FAC 0.06-0.50 + 6 baselines | rep shuffle | BASELINE_LDS.md |
| Shampoo / SOURCE / TrackStar / BM25 / embedding scorings | all | every scoring ran against a rep bank | LDS_RESULTS.md, SHAMPOO_RESULTS.md |
| adam knob tables (weight decay, logit scale, clip, bs16) | ms 0.500-0.997 | rep-era ms probe | LDS_RESULTS.md "other knobs" |
| eps_root fine sweep 1e-7 / 1e-9 | ms 0.978 / 0.907 | rep-era ms probe; 1e-9 was non-monotone (single seed noise) | LDS_RESULTS.md eps sweep |
| WikiText dropout-active runs | MAGIC -0.23 / 0.19 (n=1) | dropout active (also rep) | LDS_RESULTS.md dropout section |
| OLMo2 full-run rep bank EK-FAC | 0.0175 (CI spans 0) | rep bank | LDS_RESULTS.md OLMo2 section |

A consequence to be aware of: **the previous companion `attribution_methods.csv` is gone** —
every one of its 52 rows scored a rep-era bank. The per-epoch banks under
`/mnt/ssd-1/lucia/perepoch/runs/*/bank` (50 models each, 11 configs) make regenerating those
numbers scoring-only work; recreate the file when the first per-epoch scoring lands.

## Data caveats

Facts about the recorded data that are easy to misread:

- **WikiText eps1e-8 bs64 MAGIC 0.169 vs 0.5087** (same nominal config, different banks and
  estimator sizes): both rep-trained, so both are excluded. The discrepancy is closed for the CSV
  but unexplained on its merits — if that config is re-run per-epoch, use one estimator config
  (100 subsets x 50 queries) so it cannot recur.
- **`drop0` twin**: the rep-era "dropout 0 vs 0.1" comparison was void (dropout inert in both
  arms, bit-identical scores). Under per-epoch the two banks are kept as an honest **replicate
  pair** — `sm_adam_eps1e8_4k` (cfg 0.1, inert) and `sm_adam_eps1e8_4k_rep2` (cfg 0.0): identical
  effective training, EK-FAC 0.3095 vs 0.3048. That gap is the best available estimate of
  bank-construction noise (~0.005).
- **eps1e-9 non-monotonicity**: excluded with the rep-era sweep. If the fine sweep is re-run,
  use a second `direction_seed` before believing any non-monotone cell.
- **metasmoothness below ~0.02 carries no information** (sign-statistic noise; the same OLMo2 run
  scored 0.0101 and -0.000165 with movement agreeing to 0.12%). Applies to the OLMo2 full-run
  rows, which are recorded as ~0, not as precise values.
- **eps1e-17 adamw scores** were rebuilt from per-query `.pt` files after the `docs-4`
  padded-query bug; verified bit-identical convention against muon's normally-written scores.
- **train_loss on the per-epoch grid** was not recorded during the replication; those cells are
  empty and can be backfilled from the banks' saved base models.

## The coverage picture (honest version)

The two filters cost most of the old MAGIC data, because MAGIC was measured almost exclusively on
rep-era runs and a metagradient replay is order-dependent:

```
                     measured rows   ms   magic   ekfac
gpt2_ft                        26    24      3      12
olmo2_scratch                  18    18      0       1
rows with all three             1  (sm_adam_eps1e8_4k_bs32_ep1 — and its MAGIC CI spans zero)
```

So the metasmoothness-vs-MAGIC relationship currently rests on 3 admitted points. The `fill_*`
rows are the cheapest path back:

- `fill_<grid row>_magic` (10 rows): per-epoch banks already exist; each needs one MAGIC rollout
  (one reverse pass per query) + `validate --retrained_dir`.
- `fill_sm_{adamw,muon}_eps1e17_16k_bs256_ms_ekfac`: 100-model banks exist; EK-FAC is
  scoring-only, ms is one probe. Completing these gives bs256 all three metrics for both
  optimizers.

## Gaps against the target axis list (admitted data only)

| axis | admitted rows | note |
|---|---|---|
| batch size 16-256 | 16:1(ms) 32:1 64:20 128:1 256:3 | only bs32 has MAGIC |
| optimizer adam / muon | 15 / 11 (gpt2) | muon MAGIC only at eps1e-17 |
| tokens (axis: 4k-64k) | 4k:14 8k:4 16k:6 32k:2 64k:0 | |
| model size | 0 | |
| checkpoint averaging | 0 | |
| QK-norm / pre-act norms | 0 | needs the gpt2_custom model + its own control row |
| logit scale / wd / clip | 0 admitted | rep-era measurements excluded; planned rows re-measure at the anchor |

## Claiming work

```python
import pandas as pd
d = pd.read_csv("experiments.csv")
cheap = d[d.run_id.str.startswith("fill_")]                      # artifacts exist
new   = d[(d.status == "planned") & ~d.run_id.str.startswith("fill_")]
```

Ground rules for new runs: per-epoch shuffle code (`1e6eea7f` or later), `dropout` configured
0.0 explicitly, `train_mode` left false, fp32, `cleanup_ckpts=false` with checkpoints kept
(checkpoint-averaging and window analyses need them), one estimator config per comparison
(100 subsets @1%, 20 queries from `query_20.hf` — escalate to `query_50.hf`, scoring-only,
only where the 95% CI half-width exceeds 0.025; see D6), and record `code_commit`. **SmolLM2 pipeline only — WikiText is
banned from paper runs (it does not scale); the builder asserts `dataset == smollm2`.** The full
control set, with the evidence behind each value, is in [`CONTROLS.md`](CONTROLS.md).

**Datasets (2026-08-20):** `train_{4..256}k` are one verified nested chain (the old 64k+ family
was a different draw and was rebuilt via `scripts/rebuild_scaling_family.py`; pre-rebuild
versions kept at `*.hf.old_disjoint`). `heldout_4k` and `query_50/20` are verified disjoint from
every train set including 256k. Canonical copies on the Hub: `EleutherAI/bergson-smollm2-scaling`
(pushed by `scripts/push_scaling_datasets.py`, which re-verifies both invariants before upload). After filling a cell, edit the row in
`build_experiments_csv.py`, re-run it, and commit script + CSV together.

## Checkpoint availability

`reusable` records what survives on disk, which decides whether a blank cell is minutes or hours:
`bank+scores` / `bank` = leave-k-out models exist (new scorer = minutes); `ms_only` = any LDS
needs a fresh ~50-100-retrain bank; `none` = full rebuild. The per-epoch grid banks are under
`/mnt/ssd-1/lucia/perepoch/runs/<name>/bank`; the eps1e-17 banks under
`/mnt/ssd-2/lucia/s16k_{adamw,muon}/merged` (models kept, base-training trajectories deleted —
EK-FAC cheap, a fresh MAGIC rollout must first retrain, which reproduces deterministically).

## Reuse rules

Retrain banks and checkpoints are the expensive assets; most additions to a config are
scoring-only. Reuse is safe exactly when training config, seed, datasets, (for MAGIC
scores) code version, and — for any bit-exact recomputation — **world size** match
(nproc changes fp reduction order: measured 7.7e-3 divergence retraining a bank base at
nproc 4 vs its original nproc 8 on the same commit) — the deterministic pipeline makes matched recomputation
bit-exact, and mismatches are silent corruption, so check the match, then reuse freely:

1. **One bank per experiment row, shared by every scorer.** The 50-100 retrained models and
   the per-(query, subset) loss diffs in `validation.csv` are independent of the attribution
   method. MAGIC, EK-FAC, and any later method (TrackStar, Shampoo, tail-filter) score
   against the same bank — never rebuild a bank to add a method. Bank query losses are
   cached on disk per config (`query_loss_cache`), so re-scoring does not even reload the
   models.
2. **The tail-filter's random-removal control is already built.** A random-1% control
   retrain is distributionally identical to a bank's leave-1%-out subset retrain, and each
   bank has 100 of them: use the bank subsets' loss changes as the random-removal reference
   (a far tighter estimate than one dedicated control run). Only the targeted top-1%
   removals need new retrains.
3. **The winning tuning run is the experiment's base model.** Same config, same seed,
   deterministic trainer — so its held-out loss fills the experiment row's `heldout_loss`
   with no extra run, and if its checkpoints are kept, the bank pipeline resumes past base
   training instead of retraining it. Keep the winner's checkpoints for expensive configs
   (64k, larger models); for cheap configs, deleting and letting the bank retrain (~minutes)
   is fine.
4. **MAGIC rollouts reuse the bank's base-training checkpoints** — the backward pass walks
   that exact trajectory. Keep them until every planned rollout for the config is done;
   after that they are recoverable by deterministic retraining if ever needed again.
5. **Standing assets:** 11 per-epoch banks (`/mnt/ssd-1/lucia/perepoch/runs/*/bank`, 50
   models each) and the two 100-model banks (`/mnt/ssd-2/lucia/s16k_{adamw,muon}/merged`)
   are ready for any scorer today — that is what the `fill_*` rows exploit.

Not exploitable without code changes (recorded so nobody assumes otherwise): continuing the
double-epochs bank from the anchor bank's models. The first two epochs of a 4-epoch run are
bit-identical to the anchor (epoch shuffles are seeded `seed + epoch`), but subset retrains
save models only — no optimizer state — so continuation from them diverges from a true
4-epoch retrain.

## Optional future data (run if time permits)

- **QK-norm (and native architecture modifications generally).** Cut from the
  current grid (D16): measuring the attribution effect of an architecture
  modification natively requires PRE-TRAINING the modification in, not grafting
  fresh norm layers into pretrained weights at fine-tune time. The
  implementation is ready (`gpt2_custom/`, `EleutherAI/gpt2-custom` on the Hub,
  OLMo-style QK-norm, bit-identical none-mod); the future study pre-trains each
  variant at the from-scratch family's scale (~6-12 h per variant on 2-4 A100s)
  and then runs the standard sweep + bank pipeline on top.

- **Shampoo influence functions.** Score the paper's admitted banks with
  Shampoo-preconditioned influence. Scoring-only under reuse rule 1 (banks are
  scorer-independent), and D8 already makes every paper run save what it needs
  (`save_optimizer_state: last`). The rep-era Shampoo banks were deleted to free disk —
  their numbers live in SHAMPOO_RESULTS.md, and any paper Shampoo result comes from the
  planned per-epoch banks in this grid, never the rep-era ones.

- **Learning-rate optimum vs batch size.** The tuning sweeps center the batch-size groups
  with a square-root rule (`2e-4 * sqrt(bs/256)`, rounded to a 2x step). The completed sweeps
  will show whether the optimum actually follows that rule; extending each batch-size group by
  one or two more lr points would map the optimum-vs-batch curve properly, for both
  optimizers — the rule is an Adam heuristic and muon's scaling is unverified.
- **Matched-steps double batch.** The step-count arms are anchor (125 steps, bs256), double
  epochs (250 steps, bs256), and double batch (63 steps, bs512). A bs512 + 4-epoch run
  (125 steps at double batch) would complete the factorial and isolate batch size at matched
  step count.

## Planned pre-training experiments (not in the current grid)

**Attribution window (also called "warm start").** Do not run attribution over the first N training
steps — for example, attribute only the last epoch. The MAGIC authors identified this as a way
to overcome pre-training attribution instability, and the OLMo2 rows already measure it:
full-run attribution is dead (~0 metasmoothness) while last-epoch-only attribution reaches
0.984 metasmoothness / 0.161 EK-FAC LDS at unchanged final loss. The planned pre-training
study varies the window fraction on a from-scratch run (the `olmo2_muon_16k_window*` rows are
the measured metasmoothness-only sweep; `weight_start_frac` on branch `feat/ms-pretrain` is
the implementation). This axis is excluded from the fine-tuning grid.
