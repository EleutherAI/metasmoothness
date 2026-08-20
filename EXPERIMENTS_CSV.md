# `experiments.csv` — source of truth for the metasmoothness grid

One row per experiment. Parameter columns first, result columns last, provenance at the end.
**An empty result cell means not-yet-run, not failure.** Regenerate from the row tables in
`build_experiments_csv.py`; edit the script, never the CSV.

```
python build_experiments_csv.py   # -> experiments.csv (82 rows: 1 done, 43 partial, 38 planned)
```

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

## Resolutions of previously-flagged inconsistencies

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
- **`warmup` units**: the trainer's `warmup` is a *fraction* of total steps (baseline 0.25). The
  planned warm-start rows (target axis 100-500) are *absolute steps* and say so in notes;
  `warmup500` exceeds the anchor's 125 total steps and needs a decision (extend epochs, or treat
  as all-warmup) before running. Never mix the two units on one plot axis.
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
| batch size 16-256 | 16:1(ms) 32:1 64:19 128:1 256:3 | only bs32 has MAGIC |
| optimizer adam / muon | 15 / 11 (gpt2) | muon MAGIC only at eps1e-17 |
| tokens/steps 4k-32k | 4k:14 8k:4 16k:5 32k:2 | |
| model size | 0 | |
| warm start | 0 | |
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
(100 subsets @1%, 20+ queries), and record `code_commit`. After filling a cell, edit the row in
`build_experiments_csv.py`, re-run it, and commit script + CSV together.

## Checkpoint availability

`reusable` records what survives on disk, which decides whether a blank cell is minutes or hours:
`bank+scores` / `bank` = leave-k-out models exist (new scorer = minutes); `ms_only` = any LDS
needs a fresh ~50-100-retrain bank; `none` = full rebuild. The per-epoch grid banks are under
`/mnt/ssd-1/lucia/perepoch/runs/<name>/bank`; the eps1e-17 banks under
`/mnt/ssd-2/lucia/s16k_{adamw,muon}/merged` (models kept, base-training trajectories deleted —
EK-FAC cheap, a fresh MAGIC rollout must first retrain, which reproduces deterministically).
