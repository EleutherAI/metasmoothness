# The zero endpoint: where MAGIC has no LDS, it has no filter effect either

gpt2-medium finished its MAGIC filter run, and it is the most informative single
point the correlation has. Its MAGIC LDS is -0.0407, an interval containing zero
-- attribution on that model carries no signal at all. The question was whether
its filter delta would agree.

    run                              MAGIC LDS   filter delta   random control
    sm_adamw_eps1e17_16k_bs256          0.9411        0.09090          0.00023
    plan_adam_eps1e17_16k_bs16          0.1796        0.01443          0.00040
    plan_adam_eps1e17_16k_scale0.25     0.0456        0.00135          0.00010
    plan_adam_eps1e17_16k_gpt2-medium  -0.0407       -0.00018          0.00017

It agrees, and tightly. On gpt2-medium, removing the documents MAGIC ranks most
influential changes the query loss by -0.00018, which is the same magnitude as
the random control (0.00017) and the wrong sign. Removing "influential" data is
indistinguishable from removing arbitrary data.

Contrast the anchor: delta 0.09090 against a random control of 0.00023, a factor
of about 400.

And the four rows are monotone in LDS across nearly the whole range: 0.9411 ->
0.09090, 0.1796 -> 0.01443, 0.0456 -> 0.00135, -0.0407 -> -0.00018.

## Effect on the correlation

    MAGIC  +0.561 [+0.212, +0.788]  16 rows  ->  +0.632 [+0.297, +0.827]  17 rows

The point estimate rose and the lower bound moved from +0.212 to +0.297. Adding a
row at the zero end STRENGTHENED the relationship, which is the outcome that
distinguishes a real effect from one carried by the middle of the range. Had the
delta been large despite a zero LDS, the correlation would have been in trouble.

## What it does not say

EK-FAC is unchanged at +0.186 [-0.131, +0.497] and still contains zero. The
asymmetry between the two scorers is the open question, and the likeliest reason
remains that EK-FAC LDS barely varies -- roughly 0.40-0.47 across every row
measured, against MAGIC's 0.87-0.95 on identical banks. A scorer whose LDS does
not move has little for a delta to track.

Also note gpt2-medium is a single model-size point. That MAGIC collapses there is
established; whether it is the parameter count, the ms of 0.8580, or something
else about that run is not.
