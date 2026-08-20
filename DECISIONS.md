# Design decisions

Rulings recorded 2026-08-20. Settled controls live in [`CONTROLS.md`](CONTROLS.md). When an
open decision is resolved, move it to Resolved and edit the affected rows in
`build_experiments_csv.py` / `build_tuning_csv.py` in the same commit.

## Resolved

### D1. "Warm start" means an attribution window, not lr warmup — moved to pre-training

Warm start = not running attribution over the first N training steps (for example, attributing
only the last epoch). The MAGIC authors identified this as a way to overcome pre-training
attribution instability, and our OLMo2 result matches it (full-run attribution scores ~0;
last-epoch-only attribution scores 0.984 metasmoothness / 0.161 EK-FAC LDS).

**Ruling:** exclude from the fine-tuning experiments. The three `warmup*` experiment rows and
their tuning groups are removed. The axis is registered under "Planned pre-training
experiments" in `EXPERIMENTS_CSV.md`. The lr-warmup fraction (0.25) stays a fixed control and
is not an axis.

### D2. Step-count contribution: add a double-epochs arm and an uncontrolled double-batch arm

Number of training steps looks like a large contributor, so the grid now includes, from the
anchor (16k, bs256, 2 epochs = 125 steps):

- `plan_adam_eps1e17_16k_ep4` — double epochs (4 epochs = 250 steps, batch unchanged).
- `plan_adam_eps1e17_16k_bs512` — uncontrolled double batch (bs512, epochs unchanged,
  63 steps), for comparison.

Both get their own lr tuning groups ("well-tuned" requirement). Together with the bs16-128
axis these separate step-count effects from batch-size effects.

### D3. Token axis runs 4k to 64k

128k and 256k experiment rows and tuning groups are removed. The estimator (100 subsets, fixed
query set) is never thinned at any rung. The 128k/256k splits stay on the Hub — the nested
chain loses nothing by not using its top rungs.

### D4. Keep every rung, including 4k

No data points are removed. 4k (31 steps at bs256) stays on the axis; it is cheap. Its known
limitation — short-run effects are confounded with small-N — is recorded in CONTROLS and
handled in the sweep procedure (2 seeds, below).

### D5. Muon twins for the batch-size axis and the dataset-size axis

The dataset-size (token) axis already has muon rows. Muon rows for bs16-128 are added, with
their own tuning groups. Other axes stay adam-only.

### D8. Extra attribution methods wait; keep checkpoints and optimizer states

Shampoo / SOURCE / TrackStar / baselines wait until the MAGIC + EK-FAC set is collected.
So that they stay cheap to add later, every paper run now saves what those methods need:
**`save_optimizer_state: last`** (TrackStar-Adam and SOURCE-Adam need the final second
moments) in addition to the existing keep-checkpoints rule. CONTROLS updated.

### D9. Checkpoint averaging: last 4 checkpoints, averaged query gradient

Definition: average the **query gradient** over the **last 4 checkpoints** of the run (k=4 is
the default; the k=8 row is removed). First step is replicating Louis's effect on the existing
anchor config before adding grid rows. Note: the anchor's base-training checkpoints were
deleted, so the replication first re-trains the anchor base with checkpoints kept
(deterministic at fixed seed, ~125 steps).

### D10. Custom GPT-2: finalize, upload, use the OLMo QK-norm

The custom GPT-2 implementation gets finalized and uploaded to the EleutherAI Hugging Face
org. Where multiple QK-norm variants exist, use the OLMo implementation. Before any modified
variant runs, fine-tune the *unmodified* custom implementation once and confirm its held-out
loss matches stock GPT-2 — this separates "effect of the modification" from "effect of
reimplementing GPT-2". The arch rows stay blocked until the model is uploaded.

### D11. Model scaling: minimal, and check the plan first

Scale minimally (gpt2-medium before gpt2-large, and only if medium is informative). A concrete
cost-and-feasibility plan goes to Lucia for sign-off **before** any model-size run starts,
including the tuning sweeps. The tuning groups stay registered but are not claimable until
sign-off.

### D12. eps_root: a handful of anchor twins, eval-loss check only

Run the minimum needed to show eps_root 1e-17 does not make eval loss non-negligibly worse:
train-only anchor twins at eps_root=0 (one per optimizer, lr 2e-4), compare held-out loss.
Registered as the `tune_*_16k_eps0_control` rows in tuning.csv. No bank unless the loss check
surprises.


### D6. Query count: 20 (resolved); tail-filter estimator specified

**Ruling:** the 20-query CIs are fine — the grid stays at 20 queries everywhere.

Reference (anchor, 100 subsets, 10k bootstrap): adamw 0.9333 [0.9186, 0.9448]; muon 0.8470
[0.8274, 0.8685]; paired diff +0.0863 [+0.0670, +0.1052]. Cost note kept for the record: a
MAGIC query is a full reverse pass (~6x the cost of one bank retrain), so raising the query
count is more expensive than enlarging the bank.

**Tail-filter estimator (understanding confirmed; specification):**

- For each query: score all training documents, remove the selected slice, retrain once,
  measure that query's loss change vs the unablated baseline; average over queries.
- **Head or tail:** the bergson eval takes a user-facing switch selecting either end of the
  score ranking. In practice this paper filters the **proponents** (top-scoring documents),
  i.e. performance suppression.
- **How much to filter:** percentage, not absolute count; **1% by default**.
- **Interpretation:** values are **relative only** — a larger loss change means one
  attribution method is more efficacious than another. No absolute meaning is claimed for
  now.
- Open implementation detail carried forward: whether the comparison includes a matched
  random-1% control retrain (recommended, one per config), to anchor the relative scale.

### D13. Run-to-run variation: use existing evidence; no dedicated repeat runs

**Ruling basis:** seed randomness looks small in practice. The repo's existing data supports
this, and in this setup a stronger statement is available: with dropout off and the init
fixed (pretrained GPT-2), the training seed changes **only the data order** — and we have
direct measurements of what data-order changes do.

| evidence | perturbation | effect on the metric |
|---|---|---|
| rep -> per-epoch replication, 11 configs | entire data-order scheme changed | every EK-FAC LDS within its CI; largest shift +0.02; metasmoothness within ~0.01 |
| replicate bank pair (bs64, eps1e-8) | identical training, second bank + scoring run | EK-FAC 0.3095 vs 0.3048 (gap 0.005) |
| anchor split-half (computed from existing eval CSVs) | first 50 vs last 50 subsets | adamw 0.9294 vs 0.9336 (gap 0.004); muon 0.8451 vs 0.8429 (gap 0.002) |
| held-out loss seed repeats | seed only | sd ~0.001 nats |

**Revised proposal (adopted):** no dedicated repeat runs. Quote 0.02 as the conservative
run-to-run bound (the 11-config order-change maximum, which upper-bounds seed effects here)
and ~0.005 as the typical bank/scoring repeat gap, alongside the bootstrap CIs. Revisit only
if an axis effect the paper wants to claim falls under 0.02.

## Open

*(D6 and D13 moved to Resolved, 2026-08-20.)*

### D7. The canonical EK-FAC configuration (ruling endorsed; investigation open)

Ruling (endorsed 2026-08-20): do **not** pin whatever one run happened to use. There is a canonically correct
configuration and it should equal the library default. Open items: determine what the bergson
library default actually is (gradient space, damping selection), check whether the historical
"docspace" and "allium-0" variants differ from it and why they differed 5x on one bank, and
confirm the choice together before the ekfac_lds column fills. Until then, EK-FAC scoring
runs are on hold.


---

# The learning-rate sweep grid

This section explains the grid design in full. Terms used:

- **Held-out loss**: mean per-token cross-entropy on `heldout_4k`, the 4000-document set that
  is disjoint from every training set. This is the selection metric for all tuning.
- **Seed noise**: how much held-out loss changes when the same configuration is retrained with
  a different random seed. Measured at about **0.001 nats**. Two runs whose held-out losses
  differ by less than about 0.002 (two times seed noise) cannot be distinguished.
- **The 16k reference sweep**: the ten measured runs (five learning rates times two
  optimizers) at the main configuration, recorded in `tuning.csv` and CONTROLS.md.

## What the grid must achieve

Every experiment must run at a learning rate close enough to its optimum that no comparison in
the paper is driven by one arm being mistuned. The grid must therefore (a) detect when a
configuration's lr optimum has moved away from 2e-4, and (b) not waste runs resolving
differences smaller than seed noise.

## Why steps of 2x are the right spacing

The 16k reference sweep shows how held-out loss degrades as lr moves away from the optimum:

| distance from optimum | example (adamw) | loss penalty |
|---|---|---|
| at the optimum (2e-4) | 3.2572 | — |
| 2x away (1e-4 or 4e-4) | 3.2592 / 3.2670 | 0.002-0.010 |
| 4x away (8e-4) | 3.2990 | 0.042 |
| 10x away (2e-3) | 3.3974 | 0.140 |

Two consequences. First, a run within 2x of its optimum loses at most about 0.01 nats — small,
and detectable only at the top end of that range. Second, differences between lrs that are
both within 2x of the optimum are around seed noise (0.002), so a finer grid (for example,
steps of 1.4x) would be measuring noise. Steps of 2x are therefore the coarsest spacing that
still catches meaningful mistuning, and the finest spacing the metric can support.

## The procedure

1. For each tuning group, train three runs at {0.5x, 1x, 2x} of that group's center (centers
   below). No banks, no attribution — training plus one held-out evaluation each.
2. If the lowest held-out loss lands on an endpoint of the three, add one more run one step of
   2x further in that direction, and re-check. Stop after two extensions: an optimum more than
   4x from the center means something unexpected changed, and that needs investigation, not
   more sweep.
3. **Tie rule:** if the best and second-best runs differ by less than 0.002 (two times seed
   noise), select the anchor value 2e-4 (or the group's center) rather than the numerical
   winner. This keeps lr constant along an axis unless the data clearly demands a change, so
   most comparisons stay one-factor.
4. **Seeds:** one run per point, except at the 4k rung (31 training steps), where run-to-run
   noise is plausibly larger than the differences being measured: use two seeds per point
   there and select on the mean. If any group's three points are not lowest-in-the-middle or
   monotone, rerun the odd point with a second seed before acting on it.
5. Record every run in `tuning.csv` (train_loss, heldout_loss, run_dir); when the group is
   complete, write the winning lr into the experiment rows named in its `gates` column.

## Sweep centers

Centers use prior knowledge about how lr optima move, so that most groups finish in three runs
instead of needing extensions:

| axis | center | reason |
|---|---|---|
| dataset size (4k-64k), both optimizers | 2e-4 | no strong reason to expect the optimum to move with dataset size at fixed batch and epochs |
| batch size 16-32 | 5e-5 | optima tend to scale with the square root of batch size; sqrt(16/256) = 1/4 of 2e-4, rounded to a 2x step |
| batch size 64-128 | 1e-4 | same rule; sqrt(64/256) = 1/2 |
| double batch (bs512) | 2e-4 | same rule rounds back to the anchor value |
| double epochs (ep4) | 2e-4 | longer runs sometimes prefer slightly lower lr; the grid's 1e-4 point covers that |
| model size (medium/large) | 1e-4 | larger models under standard parameterisation usually prefer lower lr — pending D11 sign-off |
| logit scale, weight decay, clipping | 2e-4 | these knobs rarely move the lr optimum; their groups exist to verify that, not to hunt |
| gpt2_custom (all variants) | 2e-4 | same model family; the D10 equivalence check validates the transfer |

If the square-root batch rule is wrong, the endpoint-extension rule corrects it at the cost of
one extra run per group — the prior only has to be roughly right to pay for itself.

## Cost

Priority-1 groups (dataset size x 2 optimizers x 5 rungs, batch size x 2 optimizers x 4
values, ep4, bs512): 28 groups, about 90 runs, almost all under 10 minutes; the 64k rungs cost
about an hour each. The whole grid costs less than one retrain bank, and it protects roughly
40 of them.
