# Open design decisions

The register of decisions that must be made before the corresponding runs start. Settled
controls live in [`CONTROLS.md`](CONTROLS.md) and are not repeated here. Each entry: the
question, the evidence in hand, a recommendation, and what it blocks. When a decision is made,
move it to the "Resolved" section at the bottom with the date and rationale — the planned rows
in `experiments.csv` / `tuning.csv` get edited in the same commit.

## D1. Warm-start axis: what does it mean, and where can it run?

**Question.** The target axis "warm start (100 to 500)" is currently implemented as *absolute
warmup steps*. Two problems: (a) it may instead have meant warm-starting from a
partially-trained checkpoint — the two readings are entirely different experiments; (b) the
anchor run is 125 steps total, so warmup 100 is 80% of training and warmup 500 does not fit.

**Evidence.** `LRScheduleConfig`: `warmup_steps >= 1` is absolute, `< 1` is a fraction. The
anchor uses 0.25 (~31 steps). Tulu 3 uses a warmup ratio of 0.3.

**Options.** (i) warmup-steps reading, run on a longer rung (64k = 500 steps or 128k = 1000
steps) so 100/200/500 are all interior; (ii) warmup-steps reading, as fractions of the 125-step
anchor {0.1, 0.25, 0.5, 0.8}; (iii) checkpoint-warm-start reading — a new axis definition.

**Recommendation.** Confirm the intended reading first. If warmup-steps: option (i) at 64k,
because fractions (ii) change the question from "how many high-lr steps" to "what schedule
shape", and 80%-warmup runs are not realistic post-training configs.

**Blocks.** `plan_adam_eps1e17_16k_warmup*` (3 rows) + their tuning groups (one already
marked blocked).

## D2. Batch-size axis: which deconfound?

**Question.** At fixed epochs, bs co-varies with step count (bs16 = 2000 steps, bs256 = 125).
Attribute an effect to batch size or to steps?

**Evidence.** The rep-era grid (excluded) held steps fixed at 125 by scaling epochs with bs and
still saw metasmoothness rise with bs (0.837 to 0.998) — evidence the bs effect is real and not
purely step count, but it needs re-establishing under per-epoch shuffle.

**Options.** (i) fixed epochs only (realistic; steps confounded); (ii) fixed steps only (clean
bs isolate; epochs 1-16 unrealistic); (iii) both arms.

**Recommendation.** (iii) both, but stage it: run fixed-epochs first (the realistic arm the
paper narrates), add the fixed-steps arm only for the bs values where the fixed-epochs effect
is large. Tuning sweeps transfer between arms at equal bs only if steps are close — treat the
arms as separate tuning groups.

**Blocks.** `plan_adam_eps1e17_16k_bs*` (4 rows); adds up to 4 more rows + tuning groups if
the fixed-steps arm is adopted.

## D3. Token-axis extent and estimator scaling at large N

**Question.** The axis now registers 4k-256k. A 256k bank is ~16x the 16k bank (~2000-step
retrains x 100) and a 256k MAGIC rollout is ~16x per query — roughly 1-2 GPU-weeks for the
rung. Run the full axis? Thin the estimator at the top?

**Evidence.** Bank cost scales linearly with steps; MAGIC is one reverse pass per query
(measured 25 min/query at 125 steps / 4 GPUs — scales to ~7 h/query at 2000 steps). Retrain
banks parallelise well (subset_start/stop slicing); MAGIC rollouts parallelise per query.

**Options.** (i) full axis, full estimator (100 subsets x 20 queries everywhere); (ii) full
axis, reduced estimator at 128k/256k (e.g. 50 subsets, 10 queries) with CIs honestly wider;
(iii) cap the *banked* axis at 64k and run 128k/256k as metasmoothness-only rows (ms needs no
bank).

**Recommendation.** (iii) as the default plan — banked LDS through 64k, ms-only above — with
(i) for 128k as a stretch goal if the 64k trend is interesting. Do not thin the estimator:
mixed estimator configs created the WikiText 0.17-vs-0.51 mess.

**Blocks.** The 64k/128k/256k planned rows (6) and their tuning groups (which stay: lr sweeps
are cheap even at 256k, and ms-only rows still need a tuned lr).

## D4. The 4k rung (31 steps)

Keep the rung (it is cheap and the curve should span it) but pre-register that it is
short-run-confounded and lean on 8k+ in the narrative — this is already the CONTROLS caveat.
Sub-decision: its tuning sweep at 31 steps is noisy; use 2 seeds there (see grid discussion).
**Recommendation:** keep, 2-seed tuning. **Blocks:** nothing; affects
`tune_{opt}_4k` procedure.

## D5. Muon coverage beyond the token axis

**Question.** bs / model-size / warmup / scale / wd / clip axes are adam-only. Which get muon
twins (roughly doubling those axes' cost)?

**Evidence.** The one measured optimizer contrast (anchor) is large and consistent (+0.086,
19/20 queries), so optimizer differences are real and axis-dependent muon behavior is
plausible; muon's ms was flat where adam's collapsed (bs16: 0.9932 vs 0.500, rep-era).

**Recommendation.** Muon twins for the batch-size axis only (where the optimizers demonstrably
diverge), adam-only elsewhere; revisit if the muon bs curve surprises.

**Blocks.** Whether to add `plan_muon_eps1e17_16k_bs*` rows + tuning groups.

## D6. LDS estimator: queries, and the alternative metrics

**Question.** (a) 20 vs 50 queries — 20 gave CI ~±0.013 at the anchor and one reverse pass per
query prices MAGIC; 50 tightens CIs ~1.6x for 2.5x rollout cost. (b) Her notes list two
alternatives: tail-based subset sampling (LDS otherwise unchanged) and metagradient-descent
steps. In the paper, or a separate one?

**Recommendation.** (a) Stay at 20 for the grid; the paired design already resolves +0.086
with room to spare, and query count is fixed grid-wide by the one-estimator rule. Run 50 only
on the final headline configs if reviewers will want tighter CIs. (b) Prototype tail-based
sampling on the two existing eps1e-17 banks (scoring-side change, no retraining) before
deciding; metagradient-descent-steps is a different measurement and defaults to the separate
paper unless the prototype shows they track each other.

**Blocks.** Nothing in the current grid; adds scoring-only work items.

## D7. Which EK-FAC is THE EK-FAC

**Question.** On the same lotus bank, EK-FAC "docspace" scored 0.2588 and "allium-0" scored
0.0543 — a 5x variant effect (rep-era, but the sensitivity is the point). The paper's EK-FAC
column must be one pinned variant + damping policy.

**Recommendation.** Pin the variant used by the per-epoch grid's `ekfac_scores` runs (the 11
banks under `/mnt/ssd-1/lucia/perepoch/runs`), document its damping, and state the variant in
CONTROLS. Verify by re-scoring one bank with the pinned config and matching the recorded LDS.

**Blocks.** Every future ekfac_lds cell; the two `fill_*_ms_ekfac` rows.

## D8. Extra methods on per-epoch banks

**Question.** Shampoo powers / SOURCE / TrackStar / BM25+embedding baselines were all excluded
with the rep banks. Which return for the paper? All are scoring-only against existing banks.

**Recommendation.** Regenerate TrackStar and one Shampoo power (-1/4) on the per-epoch grid as
the paper's non-MAGIC gradient baselines; leave SOURCE (variant-unstable: +0.39 to -0.39 on
one bank) and the lexical/embedding baselines to an appendix, regenerated only if cited.
Recreate `attribution_methods.csv` when the first scoring lands.

**Blocks.** Appendix scope; no grid rows.

## D9. Checkpoint averaging: definition and k

**Question.** Louis's result: averaging the *query loss* over multiple near-final checkpoints
improves MAGIC. Exactly what is averaged (query loss in the LDS diff? the MAGIC backward's
loss target?), over which checkpoints (the save_mode="sqrt" set is not evenly spaced), and k in
{4, 8} was my placeholder.

**Recommendation.** Get Louis's exact recipe before registering more rows; implement as an
eval-side variant on the existing anchor banks first (no retraining) to replicate his effect,
then decide k values.

**Blocks.** `plan_adam_eps1e17_16k_ckptavg{4,8}` definitions.

## D10. gpt2_custom: implementation and equivalence gate

**Question.** Which QK-norm (per-head RMSNorm as in OLMo 2? LayerNorm?), what
"pre-activation batch norm" means exactly in a GPT-2 block, and who implements it.

**Recommendation.** Whatever is implemented, the no-mod `gpt2_custom` control must first
reproduce stock gpt2 within noise on heldout after the same fine-tune (equivalence gate)
before any mod row runs — otherwise mod effects are confounded with implementation drift.
Register the gate as a tuning-CSV row when the model exists.

**Blocks.** All 4 arch rows + their tuning groups (already status=blocked).

## D11. Model-size axis feasibility

**Question.** gpt2-medium (355M) and gpt2-large (774M) MAGIC rollouts: the reverse pass scales
with parameters; micro-batch 16 may not fit; fp32 checkpoint storage grows ~3-6x.

**Recommendation.** Before registering banks, run the tuning sweeps (cheap) plus one MAGIC
q0-only feasibility probe at medium; decide large based on measured cost. Center the lr sweep
one octave lower (see grid discussion).

**Blocks.** `plan_adam_eps1e17_16k_gpt2-{medium,large}`.

## D12. eps_root=0 anchor twin

Cheap null-check (one bank + rollout) demonstrating 1e-17 vs 0 is indistinguishable, so the
paper can say "standard AdamW" without an asterisk. **Recommendation:** run it at adam only,
after the p1 axes. **Blocks:** nothing; one added row when adopted.

## D13. Replicate budget (noise floors)

The only bank-noise replicate pair is at bs64/eps1e-8 (EK-FAC gap 0.005). The paper's claims
live at the bs256 anchor. **Recommendation:** one replicate of the adam anchor bank (seed 43
subsets, same training) to put a bank-noise number on the headline config; likewise one
2-seed training replicate to quote trajectory noise on MAGIC LDS. **Blocks:** nothing; 2
added rows when adopted.

---

# The sweep grid: discussion and recommendation

**What the grid must deliver.** Not the exact lr optimum — a guarantee that (a) every arm is
within noise of its optimum, and (b) no comparison in the paper is confounded by one arm being
mistuned. Under-tuning is the real risk (a 2x-mistuned baseline flips optimizer and batch-size
conclusions); over-tuning wastes runs resolving differences the heldout metric cannot see.

**Resolution: octave spacing is the right quantum.** The measured anchor curve gives the
scale: at the optimum the curve is flat — 2e-4 vs 1e-4 differ by ~0.002 (twice the ~0.001
seed-noise floor), while one octave out (4e-4) costs ~0.010 and two octaves (8e-4) ~0.042.
So: differences *within* an octave of the optimum are at or below noise — a finer grid
(sqrt(2) spacing) measures nothing; and a 2x mistuning costs ~0.01, which is detectable but
small, while 4x costs real loss. A grid that lands within one octave of the true optimum is
therefore "well-tuned" in every sense the paper can measure, and octave steps are exactly the
resolution the noise floor supports.

**Shape: 3 points, centered on a prior, extend on endpoint win.** `{0.5x, 1x, 2x}` around the
incumbent, one octave extension when an endpoint wins, at most two extensions before stopping
to investigate (a >4x drift means the config changed something unmodeled). Expected cost ~3.3
runs per group. This is a bracketing search: with a smooth unimodal curve (which the anchor
sweep shows), 3 octave-spaced points either bracket the optimum or point at the extension
direction.

**Centering: spend priors, save rounds.** Centering everything at 2e-4 wastes extensions where
the optimum predictably moves:

| axis | center | prior |
|---|---|---|
| token (4k-256k), both optimizers | 2e-4 | no strong N-dependence expected at fixed bs/epochs; the 32k/1ep 8e-4 hint is confounded by epochs |
| batch size | `2e-4 * sqrt(bs/256)`, octave-rounded: bs64/128 -> 1e-4, bs16/32 -> 5e-5 | square-root batch-size scaling; if the rule is wrong the endpoint rule catches it in one round |
| model size (medium/large) | 1e-4 | standard-parameterisation optima fall with width |
| warmup / logit scale / wd / clip | 2e-4 | schedule-shape and regularisation knobs rarely move the optimum an octave |
| gpt2_custom (all mods) | 2e-4 | same model family; the equivalence gate (D10) validates this |

**Tie-breaking: prefer the incumbent.** When the best two points differ by less than 0.002
(the 2-sigma noise band), select the *anchor* lr rather than the numerical winner. This keeps
lr constant along an axis unless the data demands otherwise — so most axes remain literally
one-factor, and lr changes only where a real optimum shift is detected. This is strictly
better than both alternatives: always-retuning injects selection noise into every arm;
never-retuning risks the 4x-mistuned baseline.

**Seeds.** One seed per point everywhere except the 4k rung (31 steps), where run-to-run
noise plausibly exceeds the octave signal: use 2 seeds there and select on the mean. If any
group's curve is non-unimodal, treat it like the eps1e-9 ms anomaly — rerun the offending
point with a second seed before believing it.

**What is deliberately not swept.** Warmup fraction, schedule shape, betas, eps_root: fixed
controls. wd/clip groups exist not to tune lr but to *verify* lr-neutrality (expected result:
flat within noise, keep 2e-4); they are priority 3 for that reason.

**Cost.** p1 groups (token x 2 optimizers, bs x 1) = 16 groups, ~53 runs, dominated by the four
64k-256k rungs (0.5-3 h each at bs256); everything else is minutes. Total is well under the
cost of a single retrain bank — the grid is cheap insurance on ~40 banks.
