# Prose: learning-rate selection for the Qwen 2.5 1.5B data

Scope: the Qwen chain at batch size 256, two epochs, AdamW. Rungs $N=4$k--64k are
the plotted figure-1 replication (TIERS Tier 1); $N=128$k and $256$k were swept
and trained but their EK-FAC pipelines are Tier 3 backlog, so they are not
plotted. There is no Muon arm for Qwen.

Numbers live in `qwen_tuning.csv`, extracted from the held-out evaluation logs
under `paper_runs/tuning/_logs/`. Regenerate and gate with
`python scripts/lr_sweep_table.py`.

---

## Draft (LaTeX)

For Qwen 2.5 1.5B we select the peak learning rate by the same criterion as for
GPT-2 --- mean per-token cross-entropy on a held-out set of 4000 documents
disjoint from every training set --- but on a different grid. Rather than a
three-point window around a centre with data-driven extensions, we sweep a fixed
six-point grid at every dataset size, $\{1, 2, 5\}\times10^{-5}$ and
$\{1, 2, 5\}\times10^{-4}$. Successive points differ by a factor of two or two and
a half, so the grid has approximately the resolution of the factor-of-two grid we
use for GPT-2 while spanning a decade and a half in a single pass; sweeping the
full range once is cheaper at this model scale than paying for a sequence of
extension runs. At $N=4$k and $N=8$k we omit the $5\times10^{-4}$ point, since
$2\times10^{-4}$ already costs more than 0.18 nats against the optimum at those
sizes. At $N=32$k, $5\times10^{-4}$ diverged.

The optimum is an interior point of the grid at every dataset size, so no rung
required an extension. It is also nearly flat at the bottom: at $N=4$k the entire
range from $1\times10^{-5}$ to $5\times10^{-5}$ spans 0.0073 nats of held-out
loss, and the per-rung optimum moves only once across the whole chain, from
$2\times10^{-5}$ at $N=4$k and $8$k to $5\times10^{-5}$ at every larger size.
Larger learning rates are clearly worse everywhere: $1\times10^{-4}$ costs between
0.017 and 0.054 nats against the optimum depending on the rung, $2\times10^{-4}$
costs between 0.086 and 0.193, and $5\times10^{-4}$ costs between 0.29 and 0.53
where it converges at all.

We therefore use a single peak learning rate of $5\times10^{-5}$ for every rung of
the Qwen chain. It is the per-rung optimum at every $N\ge16$k. At $N=4$k and
$N=8$k the per-rung optimum is instead $2\times10^{-5}$, better by 0.0021 and
0.0013 nats respectively --- differences at the resolution limit of the selection
metric. Holding the learning rate fixed keeps the dataset-size axis a one-factor
comparison, which a per-rung retune would break; this matters more here than for
GPT-2, since the Qwen rungs are nested by construction (the query set is chunks
$[0, 20)$ and each training set is chunks $[20, 20+N)$ of one pool) and are
intended to differ only in how much of that pool the model has seen.

Other than the learning rate, every rung uses identical hyperparameters: batch
size 256, two epochs, AdamW with $\beta = (0.95, 0.975)$ and weight decay 0.01, a
polynomial schedule decaying to one tenth of the peak with warmup over the first
25\% of steps, bfloat16, and seed 42. The sweep scales the whole schedule rather
than the peak alone, since the final learning rate is defined as one tenth of the
peak, which keeps each sweep a one-parameter family.

## Sweep results

Held-out cross-entropy; the selected value in bold, the per-rung optimum
underlined where it differs. Rungs 4k--64k are plotted.

| $N$ | 1e-5 | 2e-5 | 5e-5 | 1e-4 | 2e-4 | 5e-4 |
|---|---|---|---|---|---|---|
| 4k   | 2.6537 | _2.6464_ | **2.6485** | 2.7005 | 2.8395 | -- |
| 8k   | 2.6504 | _2.6436_ | **2.6449** | 2.6803 | 2.8277 | -- |
| 16k  | 2.6466 | 2.6397 | **2.6389** | 2.6667 | 2.7860 | 3.1709 |
| 32k  | 2.6428 | 2.6368 | **2.6354** | 2.6590 | 2.7590 | diverged |
| 64k  | 2.6392 | 2.6342 | **2.6324** | 2.6528 | 2.7386 | 2.9970 |
| 128k | 2.6368 | 2.6314 | **2.6288** | 2.6474 | 2.7232 | 2.9525 |
| 256k | 2.6345 | 2.6284 | **2.6245** | 2.6415 | 2.7101 | 2.9125 |

## Open items

1. **The $N=4$k margin is 0.0021 nats, marginally outside the tie rule.** The
   GPT-2 protocol keeps the grid centre when the best and second-best differ by
   less than 0.002 nats. At $N=8$k the Qwen margin is 0.0013 and comfortably
   inside that; at $N=4$k it is 0.0021 and just outside. Report the fixed-lr
   choice explicitly, as the draft above does, rather than describing Qwen as
   per-configuration selection of the lowest held-out loss --- at $N=4$k that
   would not be true.

2. **The 0.002 threshold is imported from GPT-2 and has not been measured for
   Qwen.** It is twice the GPT-2 seed-noise figure of about 0.001 nats. No
   repeated-seed measurement exists for Qwen, so we do not actually know the
   resolution limit of the selection metric on this model. One retrain of the 4k
   rung at a second seed would settle both this and item 1, and 4k is the
   cheapest rung in the chain.

3. **No untrained-Qwen held-out baseline is on record.** The GPT-2 protocol
   requires every row to beat the untrained model (held-out 3.4981) by a clear
   margin, and flags rows that do not. The equivalent number for Qwen 2.5 1.5B
   has never been evaluated, so the corresponding check has not been performed
   for any Qwen row. This is one forward pass of the base model over
   `heldout_4k_qwen_v2`.

4. **The logs do not record which held-out set each sweep used.**
   `scripts/heldout_eval.py` does not print its `--heldout` argument, and two
   Qwen held-out sets existed before the sweeps ran: `heldout_4k_qwen`, which
   `extend_qwen_chain.py` records as sitting at chunks $[128020, 132020)$, and
   `heldout_4k_qwen_v2` at $[256020, 260020)$. Both begin at or beyond chunk
   128020, so **every plotted rung ($N\le64$k, training ending at chunk 64020) is
   disjoint from the held-out set either way** and the plotted selections are
   safe. Only the $N=256$k sweep is at risk, since its training range
   $[20, 256020)$ would swallow the earlier held-out position --- exactly the
   failure `extend_qwen_chain.py` was written to fix --- and 256k is not plotted.
   An exhaustive scan found no chunk of either held-out set appearing anywhere in
   `train_256k_qwen`, so there is no evidence of actual contamination, but chunk
   boundaries differ between the two builders and an exact-match scan cannot rule
   out text-level overlap. Before the 128k or 256k rungs are promoted out of
   Tier 3, re-run their sweeps against `heldout_4k_qwen_v2` and make
   `heldout_eval.py` log the set it evaluated against.
