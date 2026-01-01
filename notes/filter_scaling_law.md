# Power-law vs log-linear for QLD vs N

Written 2026-09-05. Asks whether the scaling-law literature's standard recipe --
the power-law parameterization of Kaplan et al. (2020) and Goyal et al. (2024),
fitted the robust log-space way of Hoffmann et al. (2022) -- beats the
log-linear fit this repo has been using for the filter effect. Regenerate with

    python scripts/filter_scaling_law.py                  # per-query, default
    python scripts/filter_scaling_law.py --means          # rung means
    python scripts/filter_scaling_law.py --compare-qsets ekfac 'top1%' --boot 2000
    python scripts/scaling_law_plot.py                    # figures/filter_scaling_law.pdf

Supersedes the `scale-line` / `scale-power` predictors in
`scripts/filter_extrapolation.py`, whose `scale-power` is the naive
OLS-on-log-of-means version of the Kaplan fit below.

Data: `data/qld_per_query.csv` (GPT-2 AdamW bs256 token ladder, QLD =
`filter_change - random_mean`, 20 queries per rung). No GPU, no run dirs read.

## Verdict

**Yes, but the gain is in description, not extrapolation, and it comes from the
parameterization -- not from the robust fitting.**

On the clean series (EK-FAC, top 1%, held-out queries, 4k-256k), fitting the
whole row and scoring against the rung means:

    model                            rmse (nats)   rmse (log)
    log-linear   d = a + b*log2 N        0.0094       0.624   [d <= 0 below N=3728]
    Kaplan       d = A*N^alpha           0.0052       0.189
      + Huber/L-BFGS as Hoffmann         0.0057       0.193
      + smearing back-transform          0.0052       0.181
    offset       d = d0 + A*N^alpha      0.0057       0.193   d0 -> 1.6e-9
    saturating   d = dinf - A*N^-alpha   0.0132       0.194   alpha pinned at bound

Error in nats halves; error in log space falls by 3.5x. Two structural wins
beyond the number:

1. **The log-linear fit predicts QLD <= 0 below N = 3.7k, and below 5.4k-6.1k on
   every in-distribution series** -- at or inside the smallest rung we measure
   (4k). It is saying that removing the top 1% most influential documents *helps*
   the query. A power law cannot say that.
2. **The residuals stop being patterned.** Held-out row, pred - obs by rung:

       rung           4k       8k      16k      32k      64k     128k     256k
       log-linear -0.0066  -0.0035  +0.0013  +0.0055  +0.0183  -0.0015  -0.0136
       power law  +0.0023  -0.0042  -0.0070  -0.0072  +0.0057  -0.0076  -0.0038

   The log-linear residual sweeps low-high-low: that is curvature it cannot
   represent. The power-law residual is flat noise.

## The exponent is the reportable quantity

This is the Goyal point -- a filter's damage has no scale-free number attached
to it, so what goes in a table is an exponent, not a delta at one N. Per-query
fit, 95% CI from 1000 bootstrap resamples of the 20 queries:

    series                        alpha              QLD per doubling of N
    ekfac top1%  held-out       0.553 [0.514, 0.591]      x1.47
    ekfac top1%  in-dist        0.686 [0.648, 0.723]      x1.61
    ekfac top40  in-dist        0.630 [0.584, 0.673]      x1.55
    ekfac top40  held-out      -0.074 [-0.167, 0.032]      --
    bm25  top1%  in-dist        0.743 [0.690, 0.788]      x1.67
    bm25  top40  in-dist        0.734 [0.672, 0.789]      x1.66

The held-out top-40 row is the useful negative: alpha's CI covers zero, so the
fit reports "this series does not scale" rather than drawing a slope through
noise. `filter_scaling_law.py` prints that as an explicit warning.

## Contamination inflates the exponent by 0.19

`--compare-qsets ekfac 'top1%'`, same trained runs and same rungs, the query set
being the only difference:

    in-dist   alpha = 0.746
    held-out  alpha = 0.553
    difference = 0.193  [0.132, 0.253]   (2000 boots over queries)

(0.746, not the 0.686 above, because this restricts in-dist to 4k-256k to match
held-out.) The interval excludes zero. `notes/qld_query_variance.md` established
that the query text enters the pool at 64k and that in-distribution QLDs are
inflated from there; this puts the effect on the *slope*, which is the thing a
scaling claim is about. Any exponent quoted from an in-distribution row above
64k is high by roughly this much.

## What did NOT help

* **The extra terms are unsupported.** The additive floor collapses (d0 -> 1e-9,
  six orders below the smallest rung mean) and the offset model becomes
  literally the pure power law. The saturating model pins alpha at its lower
  bound with dinf 7-8x the largest observed QLD -- there is no identifiable
  ceiling in 4k-256k, and the apparent 128k->256k plateau in the in-distribution
  row is the contamination kink, not saturation. Two parameters, not three.
* **Hoffmann's robust fitting is nearly a no-op here.** Huber vs OLS-on-logs
  moves alpha by 0.02-0.03 -- inside the bootstrap CI. With delta_h = 1e-3 and
  log residuals of order 0.3, the Huber is in its linear regime everywhere, so
  it is least-absolute-deviations in log space. It earns its keep only on the
  contaminated in-distribution rows, where it cuts nats RMSE 0.0438 -> 0.0361 by
  refusing to chase the duplicate-bearing queries. The parameterization did the
  work; the robustification is insurance.
* **Extrapolation is not improved, and honestly reported it is worse from few
  points.** Fit the first k rungs, predict the rest (clean row, nats; power law
  = per-query Huber fit with smearing, the recommended variant):

        k=3   log-linear 0.0175   power law 0.232
        k=4   log-linear 0.0207   power law 0.097
        k=5   log-linear 0.0325   power law 0.0033
        k=6   log-linear 0.0254   power law 0.0074

  From 3-4 rungs the power law overshoots badly: 4k-16k is the steepest part of
  the row, and a two-parameter law honours that slope forever. The log-linear
  line wins there for a bad reason -- it is structurally flat, so it under-shoots
  growth and lands closer by accident. From 5 rungs the power law is an order of
  magnitude better. **Do not fit an exponent to three points.** On the
  in-distribution row nothing smooth extrapolates through the 64k step; that row
  should not be used for a scaling claim at all.

## Fitting-target gotcha, if this gets reused

A log-space fit to pooled per-query points targets the GEOMETRIC mean over
queries. Per-query QLD is strongly right-skewed, so the raw back-transform
under-predicts the arithmetic mean that every figure in this repo plots -- by 9%
on the clean row and by 25-46% on the contaminated ones. Three ways out, and the
script does all three: fit the rung means instead (`--means`, the direct
Hoffmann analogue since one rung = one trained pair), or keep the per-query fit
and apply Duan's smearing factor (`kaplan-smear`), or read the log-space metric
and not the nats one. Smearing is right on the clean row (factor 1.09, best fit
of all) and over-corrects on the contaminated rows, where the skew itself grows
with N and a single global factor cannot be right.

## Recommendation

Report `QLD ~ A * N^alpha` with alpha and a bootstrap CI over queries, fitted
per-query in log space with the Huber loss, back-transformed with smearing when
the target is a mean over queries. Quote alpha from held-out query sets only.
Keep the log-linear line out of the paper: it is not merely less accurate, it
puts a zero crossing inside the measured range.
