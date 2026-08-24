# 2026-08-25 — metasmoothness collapses with STEPS, and lr tuning hides it

Raw data: `data/ms_diagnostics.csv` (11 probes), configs in `configs/diagnostics/`.

## The ladder at a fixed learning rate

At 16k docs and 2 epochs, `steps = 32000 / batch_size`, so sweeping batch size at
a **fixed** lr of 2e-4 sweeps the step count with nothing else moving:

| batch | steps | ms @ lr 2e-4 | ms at that row's own tuned lr |
|---:|---:|---:|---:|
| 256 | 125 | 0.9930 | 0.9930 (lr 2e-4) |
| 128 | 250 | 0.9813 | 0.9935 (lr 1e-4) |
| 64 | 500 | 0.9613 | 0.9853 (lr 1e-4) |
| 32 | 1000 | **0.8063** | 0.9800 (lr 5e-5) |
| 16 | 2000 | **0.5127** | 0.9133 (lr 5e-5) |

Monotone, and it crosses the 0.95 collapse boundary somewhere between 500 and
1000 steps. **This is the collapse we were climbing the ladder to find, and it
was already inside the existing grid** -- hidden because every row is measured at
its own tuned lr.

## Why the tuned-lr ladder may never collapse

Look at the last column: tuning the lr down largely rescues ms. At 2000 steps it
is the difference between 0.5127 and 0.9133. So the step-scaling runs now in
flight (bs32 at 32k/64k/128k/256k, each at its own swept lr) are measuring the
*rescued* curve, and the rescue gets stronger as the optimum drifts down --
which is exactly the drift the sweeps keep finding (1000-2000 steps -> 5e-5,
4000 -> 2.5e-5).

That does not make the ladder useless, but it changes what it measures:

- **fixed lr** answers "does more optimisation break smoothness?" -> yes, sharply
- **tuned lr** answers "does more optimisation break smoothness *for a
  well-tuned run*?" -> unresolved, and the rescue may hold indefinitely

The second is the honest paper question, since CONTROLS requires every row at its
own best lr. The first is the mechanism.

## Corroboration from the token axis

At bs256, quadrupling the corpus 16k -> 64k leaves ms at 0.988-0.995 (adam 64k
0.9876, muon 64k 0.9947). So **more data alone does not move ms** -- only more
steps do. That rules out "longer training sees more tokens" as the explanation
and points at the optimisation trajectory itself.

## What is worth running next

The cheap decisive experiment is not another rung. It is the same fixed-lr ladder
at a second lr, to check the boundary moves with lr as the rescue story predicts:
if 1000 steps at 5e-5 lands near 0.98 (it does -- that is the bs32 row) and 1000
steps at 2e-4 lands at 0.8063, then ms depends on lr x steps together rather than
steps alone. Three trainings per point, no bank.
