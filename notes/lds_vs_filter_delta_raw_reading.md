# LDS vs filter delta: read the column, do not bin it

Lucia, 2026-08-27: "I think you can often aggregate data to produce apparent
relationships that don't exist, especially using thresholding. Let's be
conservative and focus on high quality unaggregated data."

She is right and this note records what the raw rows actually show, plus a
statistic of mine that should NOT be used.

## WITHDRAWN: the step-count split

scripts/corr_by_steps.py splits rows at 125 steps and reports

    ekfac  <=125   n=13  rho +0.692  [+0.224, +0.899]
    ekfac  >125    n=13  rho -0.374  [-0.900, +0.316]

Do not cite this. The threshold was chosen after looking at the data, n=13 per
side, and the >125 interval spans nearly the whole range. Worse, on the >125
rows BOTH variables are close to constant, so the correlation is dominated by a
few atypical rows. The script stays in the repo because re-running it as new
rows land is cheap, but the number is not evidence.

## What the raw EK-FAC column actually shows

At steps >= 125, filter_ekfac_delta for 20 rows:

    18 rows      0.0474 - 0.0552
    scale0.25    0.0432
    scale0.5     0.0253

So the delta is close to FLAT across steps 125 -> 2000, across both optimisers,
and across N from 16k to 32k. The two exceptions are both logit_scale variants,
and they are also the two lowest EK-FAC LDS values in the table (0.173, 0.176).

I first wrote this band as "0.0473-0.0553 across every row from 125 steps up",
which was the range AFTER silently dropping the two scale rows. That is exactly
the quiet exclusion Lucia was warning about. The full range is 0.0253-0.0552.

Because the EK-FAC delta barely varies, there is little for its LDS to correlate
with above 125 steps, and any rho computed there is a statement about the scale
rows rather than about the method.

## What the MAGIC column shows

MAGIC LDS and MAGIC filter delta move together, visibly, without binning:

    0.30 -> 0.0134     0.77 -> 0.0329
    0.84 -> 0.0659     0.94 -> 0.0909

That is monotone across the measured range and does not depend on a threshold.

The random control is 0.00002-0.00121 on every row, two orders of magnitude
below the filter deltas, so these are not control artifacts.

## The prediction to check by eye, not by statistic

If the EK-FAC delta is genuinely flat, the pending high-step rows land in
0.047-0.055 too:

    adam/muon 64k_bs32     4000 steps    filters running
    adam 128k_bs32         8000 steps    scoring
    muon 128k_bs32         8000 steps    base training
    adam 256k_bs32        16000 steps    scoring

That is 32-128x the step count at which the band was measured. Landing inside
it, or outside it, is something you can see in a column of numbers.

## UPDATE after the 64k adamw point landed (2026-08-27)

Two things I wrote above are now wrong and the conclusion is sharper.

**The "flat band" was an artefact of the N range we had.** I said the EK-FAC
delta sits in 0.047-0.055 from 125 steps up. The 64k row measures **0.12307
[0.11110, 0.13481]**, 20/20 queries beating all 100 random subsets. The band was
flat only because every row in it was 16k-32k docs.

**The delta tracks N, not LDS and not steps.** Sorted by N, every row falls in
line:

    4k     0.012 - 0.018
    8k     0.023 - 0.029
    16k    0.040 - 0.055
    32k    0.054
    64k    0.123

**EK-FAC LDS carries almost no information about the delta.** Sorted by LDS, 20
of 27 rows sit in 0.404-0.473 - a 0.07 spread - while their deltas span 0.040 to
0.123. Four rows make the point without any statistic:

    ekfacLDS   ekfacD    steps
    0.4276     0.04874    2000    plan_muon_eps1e17_16k_bs16
    0.4281     0.04959    2000    plan_muon_eps1e17_32k_bs32
    0.4284     0.04860     500    plan_muon_eps1e17_16k_bs64
    0.4336     0.12307     500    plan_adam_eps1e17_64k_bs256

Four LDS values within 0.006 of each other; the last has 2.5x the delta of the
other three. The two logit_scale rows are the only genuinely low LDS values
(0.173, 0.176) and their deltas (0.043, 0.025) sit inside the ordinary 16k range,
so the low end does not rescue the relationship either.

MAGIC behaves differently and it is visible in the same way: its LDS actually
varies over 0.04-0.95 and co-moves with the delta (0.30->0.0134, 0.77->0.0329,
0.84->0.0659, 0.94->0.0909). EK-FAC's LDS is pinned near 0.43 almost regardless
of configuration, so it cannot predict a delta that varies threefold.

**What is still confounded:** N and step count move together in most of these
rows. The 4000-step filters now running at N=64k are the first points that
separate them - same N as the 0.123 row, 8x the steps. If they land near 0.123,
the delta is an N effect. If they land near 0.05, it is not.

