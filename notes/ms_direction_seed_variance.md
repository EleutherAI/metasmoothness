# ms direction variance: smollm2 is stable, london+muon is not

Follow-up to notes/ms_direction_seed_variance.md, with the control arm in.

## Measured

    smollm2 muon 16k bs256    seed 0  0.99636   seed 1  0.99660   seed 2  0.99672
    smollm2 adamw 16k bs256   seed 0  0.99304   seed 1  0.99425
    london  adamw 16k bs256   seed 0  0.98670   seed 1  0.98274
    london  muon  16k bs256   seed 0  0.85470   seed 1  0.98580   seed 2  0.98900

Spreads: 0.0004, 0.0012, 0.0040, **0.1343**.

## What this settles, and what it un-settles

ms is NOT generally direction-noisy. smollm2 reproduces to four decimal places
across three independent directions, which is better than I expected and means
the ms column in experiments.csv can be read at the second decimal after all.
That was the outcome I most needed to rule out, and it is ruled out.

So the earlier retraction was too broad. I withdrew the london result entirely on
the grounds that 0.8547 was "a direction artifact". It is an artifact in the
sense that the mean is ~0.987 and quoting 0.8547 as THE ms of that run was wrong.
It is not noise in the sense of "ms is unreliable" -- the same probe on smollm2
is stable to 0.0004, and london adamw to 0.0040.

The variance itself is the finding. london+muon has a direction along which
training smoothness collapses to 0.855 while other directions read 0.986+.
Nothing else measured behaves that way. That is a sharper claim than either the
original headline (a lower mean) or the flat retraction (nothing to see).

## Caveat that keeps this honest

0.8547 is one draw in three. It could be a rare tail rather than a second mode,
and three points cannot distinguish those. Seeds 3, 4 and 5 are running on london
muon, plus seed 3 on smollm2 muon as a continued control. Six directions will say
whether roughly one in three collapses, or whether seed 0 was a one-off.

If it recurs, the interesting question is what that direction is -- a perturbation
that muon on this corpus responds to discontinuously is worth identifying, and
total_movement_l1 is already recorded per run as a first place to look.
