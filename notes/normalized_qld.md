# Normalized QLD: what share of the learning does the filter take back?

Written 2026-09-05. Adds a `<figure>_normalized.pdf` beside every
`<figure>_absolute.pdf`. Regenerate with `python scripts/make_figures.py`.

    normalized QLD = (filtered - random) / (base - unfiltered)
                   = mean_q(QLD) / mean_q(base_q - unfiltered_q)

Numerator is the QLD the main figures already plot. Denominator is what training
on the corpus bought that query: the untrained base model's loss minus the
trained unfiltered model's. 0.6 reads "removing the proponents takes back 60% of
what the corpus taught the model about these queries".

Base losses are per query, cached in `data/base_query_losses.csv` by
`scripts/base_query_losses.py` (20 docs x 512 tokens per combination, seconds on
one GPU). They match bergson's `per_doc_query_losses`: mean CE over the
document's label tokens, float32 logits. The gpt2/in-distribution mean comes out
at 3.449890, reproducing the `PRETRAINED_GPT2_LOSS` constant
`plot_filter_absolute_losses.py` already drew as its reference line.

The base is resolved per row from `experiments.csv` (`model`, `logit_scale`), not
assumed to be pretrained GPT-2 -- `scripts/normalized_qld.py:base_key`.

## Ratio of means, and why not mean of ratios

Per query the denominator crosses zero: on the GPT-2 AdamW row it spans -0.07 to
+0.36 nats at 4k and is negative on 10 of 20 queries there (2 at 16k, 1 at 32k,
none from 64k up), because on a small corpus training leaves some queries worse
than the base model. A per-query ratio divides by ~0 and explodes. Every point is a ratio of
means, and the CI is a **paired** bootstrap -- resample the 20 query indices
once, recompute both means from that resample. Resampling the two independently
would throw away the pairing that makes the QLD interval ~12x tighter than the
absolute one.

`MIN_DENOM_SNR = 3` drops any point whose mean denominator is within 3 SE of
zero, and every plot script prints the reason per rung rather than letting a
point vanish. On the GPT-2 rows that removes 4k and 8k.

## Headline numbers, EK-FAC top 1%, in-distribution queries

    tokens     16M    33M    66M   131M   262M   524M
    share    43.1%  39.1%  58.4%  79.3%  83.1%  91.0%

By 512k documents the top 1% of documents account for essentially all of what
training taught the model about its own training queries. Held-out queries reach
only 33% at 256k (`filter_heldout_normalized.pdf`) -- the proponents carry the
memorised part, not the generalising part.

The normalized view also flattens a difference the raw QLD shows: GPT-2 medium's
QLD is lower than the baseline's (0.0479 vs 0.0527) but its share is the same
(44.5% vs 43.1%), because it learned less from the corpus to begin with.

## Two rows where the denominator is not the quantity you want

**Qwen2.5-1.5B: no normalized figure at all.** `(base - unfiltered)` is
0.005-0.033 nats and within 3 SE of zero at every rung -- pretrained Qwen already
models these queries, and 64k documents of training add almost nothing. There is
no learning for a share to be of, so `qwen_scaling_plot.py` deliberately writes
no figure and raises if a point ever becomes drawable. Note the QLD there is
*not* small: 0.161 nats at top 10%/64k, about 5x the total training gain, i.e.
the filter leaves the model worse than the base model it started from.

**Logit-scale variants: marked (\*) and greyed.** Their base is pretrained GPT-2
evaluated at the run's own logit scale (5.13 at 0.5, 7.58 at 0.25), because the
filter's `baseline_loss` carries the scale too. So their denominators are 1.74
and 4.08 nats, almost all of which is training undoing the deliberate
miscalibration rather than learning the corpus, and the share reads ~1%. The
arithmetic is right for the definition; the quantity is not comparable to the
other rows.
