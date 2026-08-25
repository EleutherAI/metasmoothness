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
