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
reimplementing GPT-2".

**Status:** implemented (`gpt2_custom/` in this repo — paper-specific code, kept out of
the bergson library per Lucia) and uploaded as
`EleutherAI/gpt2-custom` (private, remote code). The implementation makes the equivalence
requirement structural: modifications attach as forward hooks on stock modules, so
`arch_mod="none"` has the stock state dict and produces bit-identical logits under real
gpt2-124M weights (tested). Variants select at load time:
`from_pretrained(..., arch_mod="qk_norm"|"preact_layernorm", trust_remote_code=True)`.
No bergson changes are needed to run the variants: the existing `model_kwargs`
plumbing coerces types, so a row runs with `model: EleutherAI/gpt2-custom` and
`model_kwargs: "arch_mod=qk_norm,trust_remote_code=True,resid_pdrop=0.0,attn_pdrop=0.0,embd_pdrop=0.0"`
(verified: Hub load with override works; parser produces a real bool).

Remaining gate before the arch rows unblock: the dynamic check — one fine-tune of
`arch_mod="none"` in the pinned env confirming held-out loss matches stock. Note the
check may be satisfiable by construction: `arch_mod="none"` has the stock state dict
and bit-identical logits at 124M, so its fine-tune IS the stock fine-tune function;
Lucia may waive the literal run on that argument or keep it as a pipeline test. Then
the `tune_adamw_16k_{arch_control,qk_norm,preact_layernorm}` groups open.

### D11. Model scaling: gpt2-medium is the registered target

**Ruling: `gpt2-medium` (355M, stock HF — same architecture, tokenizer, and
pre-training corpus as gpt2-124M) is the registered scaling target.** `gpt2-large` is
deferred and runs only if medium proves informative. A concrete cost-and-feasibility plan
still goes to Lucia for sign-off **before** any model-size run starts, including the tuning
sweeps; the tuning groups stay registered but are not claimable until sign-off.

**Cost plan signed off (Lucia, 2026-08-23).** gpt2-medium is claimable. Plan as costed,
on three borrowed A100-80GB pods (marisa-0, maria-1, shivam2-0) available ~24 h:

| stage | work | estimate |
|---|---|---|
| lr mini-sweep | 3 train-only runs {5e-5, 1e-4, 2e-4}, no banks, selected on heldout_4k | ~1.5 h |
| MAGIC scoring | 20 queries, one reverse pass each, scales with params | ~10 h |
| retrain bank | 100 retrains x 125 steps, sharded across the three pods | ~4.5 h |

~190 GB disk. Scoring is the least certain estimate and **cannot be sharded** (slices resume
from `per_query/`, so scoring must complete first); if it overruns, the bank is what gets
squeezed. If the sweep's winner lands on an endpoint, extend one octave and re-check per the
CONTROLS tuning protocol.

**Measured against the plan (2026-08-23, shivam2-0, 4x A100-80GB, nproc 4).** The
stage the plan called least certain is the one that broke it.

| stage | estimate | measured | note |
|---|---|---|---|
| lr mini-sweep | ~1.5 h | ~0.4 h | three points ran concurrently on three pods |
| MAGIC scoring | ~10 h | **~26 h** | ~78 min per query x 20, one reverse pass each |
| retrain bank | ~4.5 h sharded | ~7 h sharded (est) | base train is 125 steps in ~13 min |

MAGIC is **2.6x the estimate**, and the plan already recorded that it cannot be
sharded -- `ValidationConfig` exposes `subset_start`/`subset_stop` but no query
range, so the 20 queries are strictly serial no matter how many pods are free.
Total is ~33 h against the ~19 h left on the borrowed pods, so **gpt2-medium
will not finish there**, and no amount of extra hardware changes that while
scoring is the binding stage.

Plan of record: keep it on the A100s for the remaining window (fastest hardware
available, and `per_query/*.pt` is the resume unit, so completed queries
survive), then migrate the run directory from ssd-4 to ssd-2 before the pods are
returned and finish on the A40 fleet at a slower rate.

**Migration done 2026-08-23, ahead of the deadline rather than at it.** The run
directory now lives on ssd-2 (fleet-wide) instead of ssd-4 (A100 pods only), so
an early reclaim of the borrowed pods can no longer strand it. It was moved
while the footprint was still ~40 GB -- during MAGIC the run only accumulates
small `per_query/*.pt` files, so moving early cost the same copy as moving late
and bought the whole window of insurance. Copy was verified file-for-file (51
entries, 39.2 GiB) before the source was deleted, and the run resumed on
shivam2-0 with query 1 intact rather than restarting from scratch.

The open question this raises for Lucia is whether a 355M scaling point is worth
~33 GPU-days of a shared cluster, or whether the model-size axis is better
served by a cheaper design (fewer queries, fewer subsets, or a smaller step
count) that keeps the comparison honest against the 124M anchor. D6 governs
query count, so cutting it is a controls change, not a scheduling one.

Batch size stays at the anchor's 256. The control for a model-size comparison is the *same
batch size in both arms*, not bs256 specifically -- the 124M batch-size sweep means any
measured batch would serve -- but at fixed epochs bs256 is 125 steps against bs32's 1000, so
it is 8x cheaper per retrain and the only setting where a 100-model bank fits the window.
Dependency to watch: bs256 is the one batch size whose 124M counterpart is not yet
re-measured in the pinned venv (`sm_adamw_eps1e17_16k_bs256` is scoring). If that anchor
fails, bs128 is the hedge -- 2x cost, and its 124M arm is already scoring.

`gpt2-large` remains deferred pending medium's result.

### D12. eps_root: a handful of anchor twins, eval-loss check only

Run the minimum needed to show eps_root 1e-17 does not make eval loss non-negligibly worse:
train-only anchor twins at eps_root=0 (one per optimizer, lr 2e-4), compare held-out loss.
Registered as the `tune_*_16k_eps0_control` rows in tuning.csv.

**Measured — null confirmed:** the eps_root=0 twins reproduce the eps1e-17 anchors' held-out
loss to all four reported decimals (adamw 3.2572 vs 3.2572; muon 3.2570 vs 3.2570). No bank
needed; the check closed with no surprise.


### D6. Query count: 20 (resolved); tail-filter estimator specified

**Ruling:** the 20-query CIs are fine — the grid stays at 20 queries everywhere.
**Escalation rule (revised 2026-08-22 by Lucia):** 20 queries (`query_20.hf`) unless a
config's 95% bootstrap CI half-width exceeds **0.06**; then re-score with `query_50.hf`
(scoring-only against the same bank).

The threshold was originally **0.025**, set just above the widest anchor half-width
(muon ~0.021) and comparable to the D13 run-to-run bar. In practice the clean-env grid
produces wider single-row intervals than the anchors did: of the first three recorded
MAGIC rows, two exceeded 0.025 (`plan_muon_eps1e17_4k_bs256` at 0.0475,
`plan_adam_eps1e17_16k_bs64` at 0.0512), and both are rows whose per-query Spearman is
widely spread (bs64 ranges 0.53 to 0.97). At 0.025 escalation would have been the norm
rather than the exception, at ~2.5x scoring cost per row.

**Why 0.06 does not endanger the headline claims:** this bar governs *single-row*
intervals. The optimizer contrasts are paired over queries and have their own, much
tighter intervals — the anchor's +0.0863 carries [+0.0670, +0.1052], a half-width of
0.019. A single row at ±0.06 still supports a paired difference far inside that.

Rows recorded before this revision keep their measured intervals; nothing is re-scored
retroactively. Raising the query count to 50 across the grid is registered as future
work (EXPERIMENTS_CSV.md, optional future data).

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
| GPU architecture, same python | A40 vs A100, everything else held fixed | per-cell query-loss-diff mean 9.6e-4, max 3.3e-3, against a signal spread of 1.2e-3 | `data/gpu_noise_floor.csv`, D17 control |

**Ruling (adopted):** run no dedicated repeat experiments. Treat **0.02 LDS** as the
conservative run-to-run error bar (the worst case observed under a perturbation larger
than any seed change) and **~0.005** as the typical repeat gap. Any effect the paper
claims must clear 0.02; an effect below that is reported as "within run-to-run
variation". Revisit only if a key axis effect lands under 0.02. The bar is an internal
heuristic for deciding when more data is needed — the paper does not discuss or justify
it, and no caveat about its provenance is required there.

**Amendment 2026-08-24 (D17).** The 0.02 bar was derived entirely from
same-hardware evidence and does not cover GPU architecture. Measured: retrains
on A40 vs A100 with python, code, venv, nproc, seed and bank held fixed disagree
by 9.6e-4 per cell against a 1.2e-3 signal spread (`data/gpu_noise_floor.csv`).
Scoring one bank whose subsets were split across both moved its LDS from 0.8379
to 0.7828, i.e. **0.055 -- nearly 3x the 0.02 bar**.

Two distinct quantities, and only the first is measured:
- *mixed bank* (subsets from both GPU types in one bank): 0.055. D17 now
  forbids this, so it should not recur.
- *two clean banks, one all-A40 and one all-A100, same config*: **NOT YET
  MEASURED.** This is the number that would apply to cross-hardware ROW
  comparisons such as bs128 (A100) against bs32 (A40). Until it exists, treat
  cross-hardware row comparisons as carrying an unquantified error at least as
  large as the same-hardware 0.02, and prefer within-hardware contrasts.

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

### D17. GPU type is part of run identity — CAUSE CONFIRMED; scope still needs Lucia

**CONTROL RESULT (2026-08-24, lotus-0).** lotus-0 is an A100 running Python
3.11.15 -- the same Python as the A40 fleet -- so retraining subsets of an
A40-built bank there varies the GPU and nothing else. Subsets 0-2 of
`plan_adam_eps1e17_16k_wd0.0`, 60 cells:

| comparison | Python | mean disagreement |
|---|---|---|
| A40 vs A40 | same | 2.5e-07 |
| **A40 vs A100 (this control)** | **same (3.11.15)** | **9.6e-04** |
| A40 vs A100 pods (original) | differed (3.11.15 vs .16) | 6.9e-04 |
| *diff signal being ranked* | | *1.2e-03* |

Holding Python constant does not shrink the gap -- it is slightly larger. **The
Python patch version contributed nothing; GPU architecture is the cause.** The
rule now rests on a measurement rather than an argument about cuBLAS reduction
orders.

**Measured 2026-08-23/24.** sm_muon was accidentally sharded across A40 and A100
nodes at the *same* nproc (2, verified in both slice configs, so this is not the
world-size effect constraint 2 already covers). Retrains agree to **2.5e-7**
A40-vs-A40 — matching the 8k shard-boundary check, so retraining is
deterministic on fixed hardware — and differ by **6.9e-4 mean / 2.1e-3 max**
across GPU types.

The within-query spread of `diff`, the quantity LDS ranks, has median std
**1.1e-3**. The cross-hardware disagreement is therefore **43% of the signal**,
and it moves the answer: the same bank scores **0.7828** from the mixed set and
**0.8379** from the homogeneous A40 set. That 0.055 gap is larger than most
optimizer effects in the grid.

**CONFOUND IN THIS MEASUREMENT (found 2026-08-24, after Lucia asked whether the
nodes differ in torch/nccl).** Every numerically relevant package is identical
fleet-wide -- torch 2.13.0+cu126, CUDA 12.6, NCCL 2.29.3, triton 3.7.1,
transformers 5.15.1, datasets 5.0.1, numpy 2.4.6, `tf32_matmul=False` -- and all
runs are inside a pinned venv. But **Python patch version tracks the GPU split
exactly** for the nodes compared: A40 nodes are 3.11.15 and the three borrowed
A100 pods are 3.11.16. The slices compared were A40/3.11.15 against
A100-pod/3.11.16, so GPU architecture and Python version are **not separated by
this experiment**, and the original wording here attributed it to hardware
without checking.

Hardware remains the likely cause -- the arithmetic happens in identical
compiled CUDA kernels, and sm_86 vs sm_80 select different cuBLAS reduction
orders, whereas a CPython patch release should not alter kernel numerics -- but
that is an argument, not a measurement.

**The control that settles it:** lotus-0 is an A100 (sm_80) running Python
3.11.15, via `/mnt/ssd-2/lucia/envs/paper` (same package versions, different
prefix). Retraining a handful of subsets of an existing A40 bank there costs a
few retrains rather than a new bank: agreement to ~1e-7 means Python was the
cause, a ~7e-4 gap means it is the GPU.

Note also `tf32_cudnn=True` fleet-wide while `tf32_matmul=False`. CONTROLS asks
for tf32 off; the matmul path is the one that matters for a transformer and it
is off, and GPT-2 has no convolutions, so this is not believed to affect any
result -- but it does not match the stated intent and should be set explicitly.

**Settled regardless of which cause it is, and already applied:** a bank's
retrains must all run on one node type. Both anchors were rebuilt from homogeneous A40 data; the A100 slices are
quarantined as `validation_*.csv.a100`.

**Open question, which is a controls decision rather than a scheduling one.**
Three rows are being produced entirely on A100 while the rest of the grid is
A40. Each is internally homogeneous, so each is a valid measurement — but the
comparison each exists to support is confounded with hardware:

| row | node | the comparison it feeds | confound |
|---|---|---|---|
| `plan_adam_eps1e17_16k_bs128` | marisa-0 (A100) | pairs against muon bs128, measured on A40 | optimizer contrast vs GPU type |
| `plan_adam_eps1e17_16k_bs512` | maria-1 (A100) | batch axis against bs32/64/256, all A40 | batch size vs GPU type |
| `plan_adam_eps1e17_16k_gpt2-medium` | shivam2-0 (A100) | model size against the 124M anchor, A40 | the entire D11 axis vs GPU type |

Note the confound is only demonstrated for the *retrain* half of the pipeline.
The attribution scores could not be compared here: slices resume MAGIC from
`per_query/`, so both slice sets reused scores computed once by the original A40
run, which is why `score_sum` matched to exactly 0.0. A fresh A100 run computes
both halves on A100, so its exposure is at least as large, not smaller.

**SCOPE RULING (Lucia, 2026-08-24): accept and label. No re-runs.** The
cross-hardware comparisons stay as measured and are reported as cross-hardware,
carrying the unquantified error recorded in D13. Nothing is re-measured to
unify hardware, and the A100 anchor control is NOT run.

What this does and does not touch:

| comparison | hardware | status under this ruling |
|---|---|---|
| four optimizer pairs (anchor, 8k, bs32, bs64) | all A40 | **unaffected** -- the EK-FAC vs MAGIC result is within-hardware |
| bs16 optimizer pair | A100 both arms | **unaffected** -- internally consistent |
| batch-size axis as a curve | bs16/bs128/bs512 A100, bs32/bs64/bs256 A40 | label cross-hardware |
| bs128 optimizer pair | adam A100 vs muon A40 | label cross-hardware; do not quote as an optimizer effect |
| model size (D11) | gpt2-medium A100 vs 124M anchor A40 | label cross-hardware |

The rule that a single bank must not mix node types still stands and is
enforced; this ruling is only about not repairing comparisons ACROSS rows.

**Options considered, not taken:** (a) re-run these three arms on A40 so every
comparison is within-hardware — costly, and gpt2-medium is ~50 h on A100 and
worse on A40; (b) keep them and re-measure their A40 partners on A100 instead;
(c) accept them with the confound recorded, and treat cross-hardware contrasts
as indicative only. Rows are being left running meanwhile: the data is valid for
any within-hardware comparison either way.

### D16. QK-norm experiments cut

**Ruling:** the QK-norm rows are cut from the current grid and registered as
FUTURE WORK (see EXPERIMENTS_CSV.md, optional future data). The fine-tune-graft
design measures "attribution after splicing a norm into a pretrained model";
the native question requires pre-training the modification in, which is out of
scope for this campaign. **APPLIED 2026-08-24:** the ruling covers the whole architecture axis, so
`preact_layernorm` (the same fine-tune-graft design) and `arch_control` (which
exists only to control the arch_mod rows, and with those cut controls nothing)
are cut from `build_experiments_csv.py` as well. `preact_batchnorm` was already
dropped under D14. The architecture axis is therefore empty and there is no
outstanding design question here -- the earlier wording ("registered but
blocked") described their state, not an open decision.

### D15. Reproducibility tuple includes the environment (measured); RESOLVED by ruling

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
08-03 runtime). World size is separately verified as part of run
identity by a clean same-environment test: nproc 2 vs 4 with everything else identical
diverge (max 1.15e-5 after 125 steps). Operationally: reproducing a stored run
bit-exactly requires (code commit, config, seed, world size) AND an environment
component the stored runs did not record; lotus-0 today does not have the missing
factor. Snapshot-gradient scoring is measured immune to the resulting ~0.7% model gap
(transplant test, LDS_RESULTS); MAGIC-path mixing remains untested.

**Consequences:** the "retrains reproduce deterministically" annotations on historical
banks hold only in their original environments. MAGIC fill rollouts and the D9 anchor
retrain cannot be bit-faithful to the stored trajectories/bases from today's environment.

**Ruling (final):** no further environment forensics. The pinned venv
(ENVIRONMENT.md) is the sole valid environment; every bank measured outside it
is INVALID - struck from experiments.csv (historical values remain in the
narrative docs) and the artifacts deleted. Configs the paper needs are re-run
in the venv. This supersedes the salvage options below, which are kept for the
record only.

**Superseded options (historical):** D9's replication path. Options: (a) rebuild the anchor bank fresh
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
| model size (medium/large) | 1e-4 | larger models under standard parameterisation usually prefer lower lr (D11 signed off 2026-08-23) |
| logit scale, weight decay, clipping | 2e-4 | these knobs rarely move the lr optimum; their groups exist to verify that, not to hunt |
| gpt2_custom (all variants) | 2e-4 | same model family; the D10 equivalence check validates the transfer |

If the square-root batch rule is wrong, the endpoint-extension rule corrects it at the cost of
one extra run per group — the prior only has to be roughly right to pay for itself.

## Cost

Priority-1 groups still to run (dataset size: 2 optimizers x 4 sizes; batch size: 2
optimizers x 4 values; ep4; bs512): 18 groups, 54 runs, almost all under 10 minutes; the 64k
runs cost about an hour each. The whole grid costs less than one retrain bank, and it
protects roughly 40 of them.
