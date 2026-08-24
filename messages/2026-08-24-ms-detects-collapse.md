# 2026-08-24 — UPDATE: metasmoothness does move when attributability collapses

This supersedes the read in `2026-08-24-ms-vs-attributability.md`, which was
written before the logit-scale rows had ms values.

## The logit-scale axis, which is where ms finally varies

| logit scale | ms | MAGIC | EK-FAC |
|---|---|---|---|
| 1.0 (anchor) | 0.9930 | 0.9411 | 0.4253 |
| 0.5 | 0.9878 | 0.9448 | 0.1760 |
| **0.25** | **0.9150** | **0.0456** | 0.1733 |

ms drops sharply at exactly the configuration where MAGIC collapses.

## The rank correlation is the wrong statistic and should not be quoted

`spearman(ms, MAGIC) = +0.060 (p=0.85)` over 13 rows -- essentially unchanged by
adding scale0.25, and it is **not** evidence of no relationship. Eleven of the
thirteen rows sit inside a 0.007-wide ms band, so eleven of thirteen ranks are
noise and Spearman reports ~0 whatever happens at the extremes.

Sorting by ms shows the structure the correlation destroys:

    ms < 0.95   (n=2)    MAGIC  0.0456 - 0.3020
    ms >= 0.95  (n=11)   MAGIC  0.7712 - 0.9448

**Clean separation, no overlap** -- the best low-ms row (0.3020) is far below the
worst high-ms row (0.7712). Both rows where ms leaves the band are the two
lowest-attributability rows in the grid.

## What is and is not established

Established: when a training configuration destroys attributability, ms detects
it. Two for two.

NOT established: that ms ranks configurations *within* the healthy regime. It
does not -- inside the band, ms ordering and MAGIC ordering are unrelated, and
in three of four optimizer pairs muon has the higher ms and the lower MAGIC
(gaps of 0.001-0.004, i.e. inside the noise).

So ms looks like a **detector of pathological configurations**, not a
fine-grained predictor of attributability. That is a narrower claim than "ms
predicts LDS" and it is the one the data supports.

## The out-of-sample test, running now

`bs16` has MAGIC 0.1796 -- squarely in the collapsed regime -- and its ms is
still training. It was not used to draw the boundary above, so it is a genuine
test:

- ms < 0.95 => three for three, and the detector claim holds on a row that did
  not inform it.
- ms >= 0.95 => the separation breaks on its first out-of-sample case, and the
  two low-ms rows were coincidence.

`muon bs16` and `gpt2-medium` ms are also running. n=2 in the low group is
small; nobody should lean on this until bs16 lands.
