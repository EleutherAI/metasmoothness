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
