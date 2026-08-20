# `experiments.csv` — source of truth for the metasmoothness grid

One row per experiment. Parameter columns first, result columns last, provenance at the end.
**An empty result cell means not-yet-run, not failure.** Regenerate both CSVs from the row tables
in `build_experiments_csv.py` / `build_methods_csv.py`; edit the scripts, never the CSVs.

```
python build_experiments_csv.py   # -> experiments.csv          (108 rows, 56 cols)
python build_methods_csv.py       # -> attribution_methods.csv  (52 rows, long format)
```

## Claiming work

```python
import pandas as pd
d = pd.read_csv("experiments.csv")
todo = d[(d.status != "planned") & d.ekfac_lds.isna() & d.bank_dir.notna()]   # cheapest wins
```

Rows where `reusable` starts with `bank` already have the leave-k-out retrained models on disk, so a
new scorer needs only `validate --retrained_dir <bank_dir> --scores <new scores>` — no retraining.
Rows with `reusable == ms_only` or `none` need a full rebuild from the parameter columns.

After filling a cell, edit the corresponding row in `build_experiments_csv.py`, re-run it, and
commit both the script and the CSV.

## Columns that are easy to get wrong

| column | meaning |
|---|---|
| `steps` | `n_docs * num_epochs / batch_size`. Several axes move this as a side effect — deconfound before attributing an effect to the knob. |
| `eps_root` | epsilon **inside** the AdamW sqrt: `m / (sqrt(v + eps_root) + adam_eps)`. Non-standard. For muon it reaches only the AdamW-fallback params (121,344 of 163M = 0.07%), which is why muon is flat across this axis. |
| `shuffle` | `rep` = shuffled once then `.repeat(epochs)`, so every epoch sees the same order. `per_epoch` = reshuffled each epoch (commit `1e6eea7f`). Both ms and EK-FAC LDS were measured invariant to this over 11 configs. |
| `ms_shuffle` | how the metasmoothness probe shuffled. Several early rows have `shuffle=rep` but `ms_shuffle=per_epoch` — a measurement mismatch, recorded rather than silently fixed. |
| `train_mode` | `False` => trainer calls `model.eval()`, so the configured `dropout` is **inert**. Most rows with `dropout=0.1` trained with no dropout at all. |
| `attr_window_frac` | 0.0 = attribute the whole run. The OLMo2 tail rows use 0.833 (last epoch). Above 0.833 some docs never appear in the window and the bank is invalid. |
| `warmup` | baseline `0.25` is a **fraction** of total steps; the planned warm-start rows use absolute step counts. Do not mix them in one plot. |
| `ckpt_avg_k` | number of near-final checkpoints the query loss is averaged over. 1 = no averaging. Requires `cleanup_ckpts: false` at train time. |
| `magic_lds` | depends on the metagradient **code version**. Everything here is post-`c0f11ba8` ("Fix metagrad replay correctness under CUDA dropout and DDP"); the same config read 0.37 pre-fix vs 0.17 after. Check `code_commit` before comparing across rows. |
| `n_subsets` / `n_queries` | the LDS estimator config, not the training config. Two rows with identical hyperparameters but different estimator settings are different measurements — see the `wt_adam_eps1e8_bs64` vs `wt_adamw_eps1e8_lr0.0008_bank100` discrepancy below. |

## Known inconsistencies

- **WikiText eps1e-8 bs64 appears twice with MAGIC 0.169 and 0.5087.** Same nominal
  hyperparameters, different banks and estimator configs (30 subsets x 5 queries vs 100 x 50).
  Both are recorded. Reconcile before citing either.
- **`sm_adam_eps1e8_4k_drop0` is not a valid dropout test.** Its MAGIC scores are bit-identical to
  the dropout-0.1 twin because `train_mode=False` made dropout inert in both arms. The only rows
  with dropout genuinely active are `wt_adam_eps1e6_dropout_*` (n=1 query each).
- **Metasmoothness below ~0.02 carries no information.** The metric is a movement-weighted average
  of sign agreements; near zero the signs are coin flips. The same OLMo2 run scored 0.0101 and
  -0.000165 with `total_movement_l1` agreeing to 0.12%. Confirm promising cells at a second
  `direction_seed` and quote `total_movement_l1`.
- **`sm_adam_eps1e9_4k`** (ms 0.907) is non-monotone against `sm_adam_eps1e8_4k` (0.876) on an axis
  that is otherwise monotone — single `direction_seed`, `fd_step` 0.1.

## Gaps against the target axis list

| axis | rows with data | note |
|---|---|---|
| batch size 16-256 | 16:2, 32:1, 64:49, 128:19, 192:1, 224:1, 256:5 | MAGIC exists at 32/64/128/192/224/256; EK-FAC only at 32/64/128 |
| optimizer (adam, muon) | 44 / 34 | muon has MAGIC on only 3 rows |
| tokens / steps (4k-32k) | 33 / 7 / 22 / 3 | 32k is nearly empty |
| model size | **0** | no run at anything but 124M |
| warm start (100-500) | **0** | every measured row uses `warmup=0.25` |
| checkpoint averaging | **0** | needs base-training checkpoints kept |
| QK-norm / pre-act norm | **0** | needs a GPT-2-like custom model + its own control |
| logit scale | 2 | only at 4k / eps1e-8 |
| weight decay | 21 | ms only; a null over 0-0.3 |
| gradient clipping | 1 | a no-op at the settings tried |

The `plan_*` rows are one-factor deviations from the `scaling_magic` anchor (GPT-2, SmolLM2 16k,
bs256/ga16, 2 epochs = 125 steps, `eps_root` 1e-17, lr 2e-4) — chosen because it is the most recent
config with both optimizers measured and an `eps_root` that does not raise the fp32 noise floor.
Change the anchor by editing `BASE17` in the builder.

## Checkpoint availability

`reusable` records what survives on disk, which decides whether a missing cell is cheap or expensive:

- `bank+scores` / `bank` — the leave-k-out retrained models exist; a new scorer is minutes of work.
- `ms_only` — only `metasmoothness.json`; any LDS needs a fresh bank (~100 retrains).
- `none` — rebuild from the parameter columns.

The two `sm_*_eps1e17_16k_bs256` rows keep their 100-model banks but **not** their base-training
trajectories, so EK-FAC against those banks is cheap while a fresh MAGIC rollout would have to
retrain first (deterministic at fixed seed, so it reproduces).
