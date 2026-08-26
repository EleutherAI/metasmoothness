# gpt2-medium ms peaks at 0.907; lr alone cannot reach 0.98

Lucia asked for a gpt2-medium configuration at ms >= 0.98 with the same step count
and comparable training loss. Every gpt2-medium run in the grid sat at 0.858 or
below:

    plan_adam_eps1e17_16k_gpt2-medium   bs256  125 steps  lr 1e-4     0.8580
    gpt2medium_16k_bs32                 bs32  1000 steps  lr 5e-5     0.7402
    gpt2medium_64k_bs32                 bs32  4000 steps  lr 2.5e-5   0.8580

After the london result showed lr to be the dominant ms lever -- halving it moved
ms by +0.23 on adamw and +0.43 on muon at 64k -- lr was the obvious sweep.

## The curve

16k, bs256, 125 steps, world size 4, everything else fixed:

    lr 1e-4 (tuned)   0.8580
    lr 5e-5           0.8938
    lr 2.5e-5         0.9074     <- peak
    lr 1.25e-5        0.8652     <- falls again

An INTERIOR optimum, not a monotone climb. ms tops out near 0.907 and going lower
makes it worse.

So: lr alone cannot get gpt2-medium to 0.98. That is settled by measurement
rather than extrapolated -- I had read the first three points as a rising curve
and predicted more headroom, and the fourth point contradicts it.

Worth noting the tuned lr is not the ms-optimal one. 1e-4 was chosen by held-out
loss and gives 0.858; 2.5e-5 gives 0.907. Same pattern as london, where the
loss-tuned lr cost 0.26-0.42 of ms. Tuning on loss systematically picks lrs that
are bad for the property being studied.

## The batch lever

Running now, both at 125 steps:

    gpt2medium_32k_bs512_lr2.5e-5
    gpt2medium_32k_bs512_lr5e-5

32k docs at bs512 over 2 epochs is exactly 125 steps, matching the bs256/16k
anchor -- Lucia's constraint, with the extra data compensating for the doubled
batch as she suggested. Two lr points because the optimum moves with batch.

On gpt2 the batch lever is worth a lot: bs16 0.9133, bs32 0.9800, bs256 0.9930,
bs512 0.9950. If gpt2-medium follows that shape from 0.907 at bs256, bs512 is
where 0.98 becomes plausible. If it does not, the honest answer may be that
gpt2-medium does not reach 0.98 at 125 steps at any (lr, batch), which is itself
a result about the model-size axis.

## Caveat

These are A40 measurements; the original gpt2-medium values are A100. The
lr-to-lr comparison is internally consistent, but the absolute offset against the
0.8580 anchor carries the ~0.005 hardware effect D17 documents.


## Where this is heading

Three levers tried at fixed 125 steps: learning rate (interior optimum, 0.907),
batch size (+0.006 for a doubling), and their combination (0.9136).

The gap to 0.98 is 0.066 and nothing so far moves it by more than 0.036. On this
evidence the honest answer is likely to be that **gpt2-medium does not reach
ms 0.98 at 125 steps under any (lr, batch) in reach**, and that is a result about
the model-size axis rather than a failure to search hard enough: the same probe
puts gpt2 at 0.9930 in the configuration gpt2-medium reaches 0.858 in.

If the two running points confirm it, the remaining levers are ones that change
what is being asked -- more steps, or a different epoch count -- and those break
the "same number of training steps" constraint Lucia set.


## bs512 lr curve: the optimum moved DOWN with batch

    bs256 / 16k    lr 5e-5 0.8938   2.5e-5 0.9074   1.25e-5 0.8652
    bs512 / 32k    lr 5e-5 0.8204   2.5e-5 0.9136   1.25e-5 0.9186

The bs512 curve has not turned over. At bs256 the optimum was 2.5e-5 and halving
again cost 0.042; at bs512 the same halving GAINS 0.005. So the ms-optimal lr
falls as batch rises, which is backwards from the usual scaling rule and
consistent with everything else measured: ms prefers smaller steps than loss does.

Best configuration so far: bs512, 32k docs, lr 1.25e-5, 125 steps, ms 0.9186 --
0.061 short of 0.98, with the last halving worth 0.005.

Running 6.25e-6 and 3.1e-6 at bs512 to find the turn.

For the tuning discussion: the ms-optimal lr for gpt2-medium is at least eight
times smaller than the loss-tuned 1e-4, and that gap WIDENS with batch. A grid
that picks lr by held-out loss and then reports ms is reporting a configuration
selected against the property it is measuring.
