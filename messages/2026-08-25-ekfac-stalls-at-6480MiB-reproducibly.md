# EK-FAC query-gradient write stalls at exactly 6480 MiB, reproducibly

Three attempts, two different rows, three different nodes, same outcome. This
is not a node problem and not a one-off, which is what the earlier note in this
directory claimed before the test that refuted it.

## The signature

`ekfac_scores/query.part/gradients.bin` grows to **6,794,772,480 bytes — exactly
6480 MiB** — and stops. Then:

    child A  wchan=pipe_read                  launcher, idle by design
    child B  wchan=ceph_mdsc_wait_request     holds gradients.bin, never returns
    child C  wchan=0, utime climbing          spinning in the collective wait

Rank B is blocked in a CephFS metadata request. Rank C keeps burning CPU waiting
for a peer that has stopped, which is why the GPUs read as 100% busy. If left
alone the NCCL watchdog eventually aborts a rank, which is the `code -6` SIGABRT
seen the first time.

The byte count being *identical* across rows is the useful clue: it is the same
dataset size (32k), same model, same 20 queries, so the same expected gradient
volume. The write finishes; what follows it hangs.

## Where it has been seen

    plan_adam_eps1e17_32k_bs256   shared-ord-0   6480 MiB, stalled
    plan_adam_eps1e17_32k_bs256   allium-0       6480 MiB, stalled
    plan_adam_eps1e17_32k_bs32    secret-ord-0   6480 MiB, stalled

## Ruled out

* stale output -- clearing `ekfac_scores` and starting clean reproduces it
* world size -- three EK-FAC runs succeeded at the same nproc 2 on 4k and 16k rows
* two processes racing -- fleet-wide scan found one
* the dataset path -- copying the dataset to ssd-2 and repointing changed nothing
* the output directory -- other nodes write 100 MB into it at 468-566 MB/s
* one node's client -- reproduced on three

## What is left

Size. Every EK-FAC run that has succeeded is a 4k or 16k row, whose query
gradients are far smaller. Every 32k attempt has failed at the same byte count.
The next test that would separate "large file" from "this dataset" is a 32k row
scored with fewer queries: if 10 queries write ~3240 MiB and complete, the
problem is the size of the write, and the workaround is to score in query
batches and concatenate.

## Why it matters

This blocks the proponent-filter curve past 16k. The step-ladder rows have no
bank, so EK-FAC is the only scorer available to them -- MAGIC needs a training
trajectory these runs do not keep. No scores means nothing for the filter to
rank, so 32k and every rung above it is blocked on this, not on GPU time.
# CORRECTION: EK-FAC scoring is broken fleet-wide, and it is not about size

The previous note in this directory concluded the stall was about the size of
the query-gradient write, because every failure was a 32k row and every success
was 4k or 16k. That reasoning was wrong, and Lucia pointed out why in one line:
**the query gradient does not scale with the training set.** It is (number of
queries x tracked parameters), so a 4k row and a 32k row write the same amount.

The test that follows immediately, and that I should have run before writing the
size hypothesis down: re-run EK-FAC on a 16k row that has already succeeded once.

    plan_adam_eps1e17_16k_bs64, fresh output dir, nproc 2, lucia-ord-0
    gradients.bin -> 6,794,772,480 bytes, static for 90s
    scores/ never created
    child wchan = ceph_mdsc_wait_request

Byte-identical to the 32k failures. So the file size is the same everywhere --
6480 MiB for 20 queries, dataset-independent, exactly as predicted -- and the
16k row now fails the same way it once passed.

## What that means

EK-FAC scoring is currently broken for EVERY row, not for large ones. The rows
that succeeded did so before a copy operation got stuck in uninterruptible sleep
and started holding CephFS metadata. Every EK-FAC attempt since has stalled at
the point where the finished gradient file is finalised.

Consequences:

* it is not a bergson bug and there is nothing to fix in the config
* query batching would not help; the writes are the same size that used to work
* it blocks the proponent-filter curve everywhere it needs new EK-FAC scores,
  which for the bank-free ladder rows is all of them
* it does NOT block the filter runs already going, which write per-query files
  far below this size, nor training, nor MAGIC on rows that keep a trajectory

## What clears it

The stuck operation needs to go, which means restarting the node holding it
(marisa-0, ~18 hours in uninterruptible sleep at the time of writing). That is
an operator action; `kill -9` cannot touch a process in that state because the
signal is only delivered when it returns to user space.

Until then: do not queue EK-FAC scoring. It will consume GPUs, write 6.8 GB, and
stop. Three attempts on three nodes did exactly that before this was understood.

## Route around it: MAGIC can score the ladder, with one allocator flag

EK-FAC being down fleet-wide does not have to stop the proponent-filter curve.
MAGIC scores a bank-free row fine, provided two things:

1. the base run keeps a trajectory. The ladder bases were trained with
   save_mode interval (final state only), which MAGIC cannot replay. Re-running
   with save_mode sqrt costs one training run -- 17 min at 32k, 34 at 64k.

2. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Without it the replay dies
   with CUDA OOM trying to allocate 1.53 GiB on a 47.5 GiB card, which is
   fragmentation rather than a real limit. With it the same run proceeds at
   6.5 s/it.

Score-only config: point run_path at the trajectory, resume: true,
skip_validation: true, num_subsets: 0. Backward replay is ~3.5h at 32k/2000
steps, which is the whole scoring cost -- not per query.

    plan_adam_eps1e17_32k_bs32   scoring now, Backward 23/2000, ETA ~3.5h
