# The MAGIC arm cannot reach 2000 steps for about a day, and that is a hard floor

Lucia has asked repeatedly for the filter-delta / LDS correlation at higher step
counts. This records exactly what gates it, because the answer is not capacity
and no amount of GPU sweeping changes it.

## Where the two arms stand

    MAGIC   +0.674 [+0.389, +0.835]   19 rows   every row <= 250 steps
    EK-FAC  +0.142 [-0.147, +0.437]   19 rows   includes one 2000-step row

The EK-FAC arm already has its first point above 250 steps
(plan_adam_eps1e17_32k_bs32, ekfac_lds 0.4146) and gains a second the moment the
muon bank finishes -- it is at 99/100.

## What the MAGIC arm is waiting on

MAGIC scoring for the two 2000-step rows, and it is serial by construction.
Lucia's ruling: "MAGIC Scoring can't be sharded because it's a serial process, if
you use the bergson multi query support it will still be serial internally."

Measured from the logs:

    plan_adam_eps1e17_32k_bs32   query 9/20   started Aug 25 08:25
    plan_muon_eps1e17_32k_bs32   query 7/20

That is roughly 2.3 h per query on the adam row, so about 25 h left there and
nearer 30 h on muon. Both are progressing normally; there is nothing to fix.

## Why this cannot be worked around

A MAGIC filter delta needs MAGIC scores to choose the proponents, and a MAGIC LDS
needs the same scores. So a 2000-step MAGIC point needs that serial pass to
finish, twice over. Neither more GPUs nor more nodes shortens it.

The cheap route that worked for EK-FAC -- build the bank with validate(method:
lds) against scores that already exist, skipping MAGIC scoring entirely -- is
exactly what cannot be applied here, because MAGIC scores are the missing input
rather than a by-product.

## What this means for the question

Until then, any statement about MAGIC's filter-delta/LDS relationship is a
statement about rows at 250 steps or fewer. The +0.674 is real but its
step-count range is narrow, and the EK-FAC arm already shows the relationship
weakening when a 2000-step row enters (+0.210 over 18 rows -> +0.142 over 19).
Whether MAGIC does the same is the open question and it is a day out.
