# ckptavg needs a bergson feature that does not exist yet

`plan_adam_eps1e17_16k_ckptavg4` is the last unrun row that is neither cut nor
waiting on a tuning sweep, so it is the obvious candidate for idle capacity. It
cannot run today, and the reason is not the one recorded in D9.

## The recorded blocker no longer applies

D9 says the comparison is blocked because "the anchor's base-training
checkpoints were deleted", which made the existing anchor bank's validity the
D15 open question. That was about the old anchor. The current 16k anchor,
`sm_adamw_eps1e17_16k_bs256`, is a venv-valid bank and **still has its
checkpoints**:

    step_0  step_62  step_93  step_109  step_117  step_121  step_123  step_124

The last four (117, 121, 123, 124) are exactly what `ckpt_avg_k = 4` needs, and
CONTROLS classes the axis as "eval-side only, same trained model" — a rescore,
not a rebuild. So the expensive path D9 weighed (rebuild the bank, ~10 4-GPU-hours,
~55 GB) is not required.

## The actual blocker

bergson has no checkpoint-averaging support at all:

    grep -rn 'ckpt_avg_k\|checkpoint_avg\|avg_k' bergson/   # no matches
    (checked against bergson-main-paper-429)

D9 defines the semantics — average the **query gradient** over the last k
checkpoints, and **both** scorers use the averaged gradient: MAGIC seeds its
reverse pass with it, EK-FAC preconditions it. Nothing implements that. The row
needs a library feature and a PR, not GPU time.

## What that means for the grid

`ckpt_avg_k` is a column in experiments.csv and a fixed control in CONTROLS.md,
which reads as though the knob exists. Until the feature lands, every row is
implicitly k=1 by absence rather than by setting.

Worth deciding explicitly: implement it, or cut the axis the way D16 cut the
architecture axis. Right now it is neither, and it is the only thing standing in
the "## Open" section of DECISIONS.
