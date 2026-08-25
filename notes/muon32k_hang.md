# muon + london at N>16k deadlocks in a collective; adamw does not

Parked with a solid diagnosis. Recording so nobody re-derives it.

## What it is

Both ranks get ~18 seconds into setup, then **stop**. Measured directly:

    ps -eo pid,stat,time,wchan  ->  Sl  00:00:18  wait_woken
    ... 20 seconds later       ->  Sl  00:00:18  wait_woken

CPU time frozen, `wchan = wait_woken` (blocked on a socket wait queue), GPU at
3 MiB (context created, nothing allocated), log 0 bytes even under
`PYTHONUNBUFFERED=1` and `python -u`.

Frozen CPU rules out "slow". Blocked on a socket after setup, with both ranks in
the same state, is a **distributed collective deadlock**.

## The boundary

    muon   + london_16k    5/5 lr points completed
    muon   + london_32k    stalls
    muon   + london_64k    stalls
    adamw  + london_16k/32k/64k    all run
    muon   + smollm2 train_32k     runs (plan_muon_eps1e17_32k_bs256 has a bank)

So it needs muon AND london AND N>16k. No two of those are sufficient.

## Ruled out

* the node -- reproduced on iris-0, shared-ord-0, marisa-0, maria-1
* ports, stale processes, leftover run dirs -- all cleared between attempts
* the launcher -- an `sh -c` wrapper was suspected, but a direct
  `setsid nohup env python` launch reproduces it identically
* world size -- nproc 4 deadlocks the same as nproc 2
* **dataset lock contention** -- this was the leading hypothesis, because
  HF-datasets writes cache and lock files INSIDE the dataset directory and three
  adamw jobs were holding `london_32k.hf`. It is wrong: muon deadlocks on
  `london_64k.hf` too, a fresh directory with no other job touching it.

## Best remaining hypothesis

Rank divergence during dataset preprocessing. The ranks preprocess independently,
and a larger corpus makes them diverge further in wall-clock; if muon's setup
performs a collective that adamw's does not, the ranks can arrive at it far apart
and deadlock. london_16k is small enough that they stay in step. This is
consistent with everything above but is NOT yet tested.

The cheap test: `py-spy dump` on a stalled rank names the exact call. Second
cheapest: run muon london_32k at nproc 1, which removes collectives entirely.

## Cost of leaving it

The muon arm of the london N-scaling question. The adamw arm runs at 16k, 32k and
64k and answers the same question one optimizer at a time. Six GPUs were released
rather than left at 3 MiB.
