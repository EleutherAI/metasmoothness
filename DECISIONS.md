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
query set) is never thinned at any dataset size. The 128k/256k splits stay on the Hub — the nested
chain loses nothing by not using the 128k and 256k sizes.

### D4. Keep every dataset size, including 4k

No data points are removed. 4k (32 steps at bs256) stays on the axis; it is cheap. Its known
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
the default; the k=8 row is removed). **Both scorers get the averaged gradient**: MAGIC
seeds its reverse pass with it, and EK-FAC preconditions it. First step is
replicating Louis's effect on the existing anchor config before adding grid rows. Note: the
anchor's base-training checkpoints were deleted; a fresh deterministic re-train exists
(with checkpoints) at /mnt/ssd-2/lucia/paper_runs/d9_magic_base, but per D15 it does NOT
bit-reproduce the stored bank base (~0.7% weight gap, cause unidentified), so whether the
existing anchor bank remains valid for the ckptavg comparison is exactly the D15 open
question — resolved empirically by the snapshot-gradient transplant test and, if needed,
a fresh bank.

### D10. Custom GPT-2: finalize, upload, use the OLMo QK-norm

The custom GPT-2 implementation gets finalized and uploaded to the EleutherAI Hugging Face
org. Where multiple QK-norm variants exist, use the OLMo implementation. Before any modified
variant runs, fine-tune the *unmodified* custom implementation once and confirm its held-out
loss matches stock GPT-2 — this separates "effect of the modification" from "effect of
reimplementing GPT-2". The arch rows stay blocked until the model is uploaded.

### D11. Model scaling: gpt2-medium is the registered target

**Ruling: `gpt2-medium` (355M, stock HF — same architecture, tokenizer, and
pre-training corpus as gpt2-124M) is the registered scaling target.** `gpt2-large` is
deferred and runs only if medium proves informative. A concrete cost-and-feasibility plan
still goes to Lucia for sign-off **before** any model-size run starts, including the tuning
sweeps; the tuning groups stay registered but are not claimable until sign-off.

### D12. eps_root: a handful of anchor twins, eval-loss check only

Run the minimum needed to show eps_root 1e-17 does not make eval loss non-negligibly worse:
train-only anchor twins at eps_root=0 (one per optimizer, lr 2e-4), compare held-out loss.
Registered as the `tune_*_16k_eps0_control` rows in tuning.csv.

**Measured — null confirmed:** the eps_root=0 twins reproduce the eps1e-17 anchors' held-out
loss to all four reported decimals (adamw 3.2572 vs 3.2572; muon 3.2570 vs 3.2570). No bank
needed; the check closed with no surprise.


### D6. Query count: 20 (resolved); tail-filter estimator specified

**Ruling:** the 20-query CIs are fine — the grid stays at 20 queries everywhere.
**Escalation rule:** 20 queries (`query_20.hf`) unless a config's 95% bootstrap
CI half-width exceeds **0.025**; then re-score with `query_50.hf` (scoring-only against the
same bank). 0.025 is just above the widest anchor half-width (muon ~0.021) and comparable to
the D13 run-to-run bar, so escalation triggers only when the CI would blur a reportable
effect.

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
- **Control (resolved):** every config includes a matched **random-1% removal control** by
  default. Report the targeted-removal loss change next to the random-removal loss change;
  the gap between them is the estimator's signal. No new runs needed: the config's bank
  subsets are 100 random-1% removals already — use their loss changes as the control
  reference (see Reuse rules in EXPERIMENTS_CSV.md).

### D13. Run-to-run variation: quantified from existing data; no dedicated repeat runs

**The question this answers:** when the paper reports that two configs differ by some amount
of LDS, how much of that could be luck — the same experiment redone giving a different
number? Bootstrap CIs do not answer this: they capture estimator noise (resampling the
subsets/queries we have), not what happens if the training run, the bank, or the scoring is
redone.

**Why no new runs are needed.** In this setup the training seed has exactly one job: it sets
the per-epoch data order. Dropout is off and the initialization is the fixed pretrained
GPT-2, so two seeds differ only in the order examples are visited. We already measured
something strictly stronger than a seed change — switching the entire data-order scheme
("rep": one fixed order repeated every epoch, the old implementation) to independent
per-epoch reshuffling — across 11 configurations (LDS_RESULTS.md, "Per-epoch shuffle"
section). Every measured repeat source:

| evidence | what was redone | observed change | source |
|---|---|---|---|
| order-scheme switch, 11 configs | the full data order | EK-FAC LDS max shift 0.02, all within CI; metasmoothness within ~0.01 | LDS_RESULTS.md per-epoch table |
| replicate bank pair | second bank build + scoring, identical training | EK-FAC 0.3095 vs 0.3048 (0.005) | rows sm_adam_eps1e8_4k / _rep2 |
| anchor split-half | LDS from subsets 0-49 vs 50-99 | adamw 0.9294 vs 0.9336 (0.004); muon 0.8451 vs 0.8429 (0.002) | /mnt/ssd-2/lucia/s16k_{opt}/eval_q20/validation.csv |
| held-out loss, repeated seeds | training seed | sd ~0.001 nats | measured 2026-08-06 |

**Ruling (adopted):** run no dedicated repeat experiments. Treat **0.02 LDS** as the
conservative run-to-run error bar (the worst case observed under a perturbation larger
than any seed change) and **~0.005** as the typical repeat gap. Any effect the paper
claims must clear 0.02; an effect below that is reported as "within run-to-run
variation". Revisit only if a key axis effect lands under 0.02. The bar is an internal
heuristic for deciding when more data is needed — the paper does not discuss or justify
it, and no caveat about its provenance is required there.

### D7. Canonical EK-FAC configuration: the bergson default, `damped_inverse`

**Resolved:** the canonical EK-FAC configuration for every `ekfac_lds` cell is
the bergson library default: `inversion="damped_inverse"` (uniform Tikhonov,
`1/(lambda + c*mean(lambda))`) with `damping_factor=0.1` (relative to the mean eigenvalue).
`factored_tikhonov` (the Martens-Grosse pi-split) exists in the library but is not used.
**EK-FAC scoring is unblocked.** Historical note: the rep-era "docspace" and "allium-0"
variants have no counterpart in the bergson codebase (checked at commit 8ce0cd76); their 5x disagreement
on one bank stays unexplained but cannot recur — every new score uses the one canonical
config above, recorded via `code_commit`.

### D14. preact_batchnorm dropped from the architecture axis

**Resolved:** the `preact_batchnorm` tuning group and experiment row are
removed. Two unfixable conflicts with the control set:

1. The trainer runs with `train_mode=false` (`model.eval()`), where BatchNorm uses its
   never-updated running statistics — the row would either train a broken model or require
   `train_mode=true`, contradicting a fixed control.
2. BatchNorm couples per-document gradients to the other documents in the micro-batch, so
   MAGIC's per-doc metagradient and EK-FAC's per-sample gradients lose their meaning and
   micro-batch size becomes a hidden factor. Any attribution change on the row would be
   confounded between "batchnorm hurts attribution" and "per-doc gradients are ill-defined
   under batch coupling".

The architecture axis keeps `qk_norm` and `preact_layernorm` (both per-sample operations).

### D15. Reproducibility tuple includes the environment (measured); D9 consequence open

**Finding (gate series, lotus-0):** bergson training is bit-deterministic within an
environment — two fresh runs of the s16k anchor bank config (its own commit `410aee93`,
ga 8, nproc 4, seed 42, magic step) agree on 160/160 tensors — but NO available
combination of commit/config/step reproduces the stored bank bases (best attempt: 12/160
tensors, ~7e-3 divergence). The first round of elimination gates was VOID — python's cwd-first sys.path silently
loaded the live checkout instead of the PYTHONPATH-pinned commit in every run launched
from inside a bergson repo (protocol fix: run from /tmp with `python -P`; see NODES.md).
The verified rerun (imports asserted inside the run): the s16k bank at its true build
commit (410aee93), config, seed, and world size (nproc 4) STILL fails to reproduce the
stored base (12/160 tensors equal, same signature). Also eliminated: the datasets (raw
files unmodified since 2026-07-16), a local torch upgrade (installed 2026-07-08, before
the banks), and the transformers version (5.13 vs 5.1 train bit-identically at equal
nproc). **The cause is an unrecorded environment component.** The pip history bounds the
candidates — since the 08-03 build: nvidia-nccl 2.26.2->2.28.9, datasets 4.5->5.0, numpy
2.2.6->2.4.6, triton 3.3.1->3.6.0 — and NCCL cannot be tested by downgrade (today's torch
binary hard-requires 2.28 symbols, which also proves pip records did not describe the
08-03 runtime). Operationally: reproducing a stored run bit-exactly requires more than
(code commit, config, seed, world size), and lotus-0 today does not have the missing
factor. Snapshot-gradient scoring is measured immune to the resulting ~0.7% model gap
(transplant test, LDS_RESULTS); MAGIC-path mixing remains untested.

**Consequences:** the "retrains reproduce deterministically" annotations on historical
banks hold only in their original environments. MAGIC fill rollouts and the D9 anchor
retrain cannot be bit-faithful to the stored trajectories/bases from today's environment.

**Open (needs Lucia):** D9's replication path. Options: (a) rebuild the anchor bank fresh
in the current environment (base + 100 retrains, ~10 4-GPU-hours, ~55 GB — also refreshes
the anchor MAGIC number on one consistent environment; the fresh base + trajectory already
exist at `/mnt/ssd-2/lucia/paper_runs/d9_magic_base`); or (b) accept a mixed-provenance
comparison (fresh trajectory's ckptavg gradients scored against the old bank), which
breaks the base==bank-base invariant and is not recommended. The 10 `fill_*_magic`
rollouts inherit the same choice.

## Open

*(D15's D9 consequence, above, is the only open item.)*


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
   below). No banks, no attribution — training plus one held-out evaluation each. The sweep
   scales the whole schedule, not just the peak: `lr_end` is defined as `lr / 10`, so it moves
   with `lr`. This keeps each sweep a one-parameter family.
2. If the lowest held-out loss lands on an endpoint of the three, add one more run one step of
   2x further in that direction, and re-check. Stop after two extensions: an optimum more than
   4x from the center means something unexpected changed, and that needs investigation, not
   more sweep.
3. **Tie rule:** if the best and second-best runs differ by less than 0.002 (two times seed
   noise), select the anchor value 2e-4 (or the group's center) rather than the numerical
   winner. This keeps lr constant along an axis unless the data clearly demands a change, so
   most comparisons stay one-factor.
4. **Seeds:** one run per point everywhere. An earlier draft required two seeds for
   short-run groups (63 steps or fewer), on the guess that seed noise grows when a run is
   only a few dozen batches; a direct measurement showed it does not: seeds 42 and 43 of
   `tune_adamw_8k_lr0.0002` (63 steps) give heldout 3.2851 vs 3.2839 — a 0.0012-nat gap,
   at the ~0.001 seed-noise floor measured at 125 steps. The remaining safeguards cover
   the residual risk at zero cost: the tie rule (step 3) absorbs sub-0.002 differences,
   and if a group's three points are not lowest-in-the-middle or monotone, rerun the odd
   point with a second seed before acting on it.

5. Record every run in `tuning.csv` (train_loss, heldout_loss, run_dir); when the group is
   complete, write the winning lr into the experiment rows named in its `selects_lr_for` column.

## Sweep centers

The rule for grid ranges: the only value taken from our own measurements is the shared
reference center (2e-4, the reference-configuration optimum), applied uniformly to every arm
— the standard tune-at-a-reference-then-transfer practice. Per-arm differences from that
center come only from published or conventional heuristics, never from our own data: the
batch-size arms shift by sqrt(bs/256), and the model-size arms shift one 2x step down
(larger models conventionally prefer lower lr). Extensions beyond any grid are data-driven
(endpoint rule) and recorded in tuning.csv like every other run:

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

Priority-1 groups still to run (dataset size: 2 optimizers x 4 sizes; batch size: 2
optimizers x 4 values; ep4; bs512): 18 groups, 54 runs, almost all under 10 minutes; the 64k
runs cost about an hour each. The whole grid costs less than one retrain bank, and it
protects roughly 40 of them.
