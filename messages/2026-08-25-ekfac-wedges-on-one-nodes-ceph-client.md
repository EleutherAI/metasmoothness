# EK-FAC scoring wedges on one node's ceph client, not on the code

`plan_adam_eps1e17_32k_bs256` EK-FAC scoring failed four times on shared-ord-0.
Two shapes: once `RuntimeError: build child exited with code -6` (SIGABRT), then
silent hangs. Both after the same point in the run.

## What it actually does

Rank 1 writes `ekfac_scores/query.part/gradients.bin` up to **6.79 GB**, then
parks in `ceph_mdsc_wait_request` -- a CephFS metadata request -- and never
returns. Rank 0 keeps spinning in the collective wait, which is why the GPUs
read as busy at 100%. The SIGABRT is downstream: the NCCL watchdog eventually
kills a rank that has been waiting too long.

    child 2556248 wchan=pipe_read                 (launcher)
    child 2556249 wchan=ceph_mdsc_wait_request    (blocked, holds gradients.bin)
    child 2556250 wchan=0                         (spinning)

`gradients.bin` static at 6,794,772,480 bytes over 90 s confirms it is stuck
rather than slow.

## What it is not

Each of these was tested and ruled out:

* **stale output** -- deleting `ekfac_scores` and starting clean reproduces it
* **world size** -- three EK-FAC runs succeeded at the same nproc 2
  (adam bs32, muon bs32, the adamw anchor)
* **two instances racing** -- a fleet-wide scan found exactly one process
* **the dataset path** -- the run reads `train_32k.hf`, and the wedged ssd-1
  directory is only a problem when something ENUMERATES the parent
  `datasets/`. Copying the dataset to `/mnt/ssd-2/lucia/datasets_local` and
  repointing the config changed nothing.
* **the output directory** -- bellflower-0 and allium-0 both write 100 MB into
  that exact directory at 468 and 566 MB/s

## What it is

shared-ord-0's ceph client is wedged on that inode. The directory is healthy
from every other node; ordinary metadata operations on ssd-2 from shared-ord-0
itself still work (touch, mkdir, write into query.part all succeed). It is one
client stuck on one file.

## What to do

Run EK-FAC scoring for this row on a different node. Nothing in the config or
the code needs changing.

More generally: when a bergson step hangs, check `wchan` on the CHILDREN, not
the launcher. The launcher always sits in `do_wait`, which looks stalled and
tells you nothing. A child in `ceph_mdsc_wait_request` means the filesystem, a
child spinning at 100% with a static log means it is waiting on a peer that has
already stopped.

This is the second ceph wedge in a day; the other is the `mv` on marisa-0, now
16+ hours in uninterruptible D state, which strands that node's six A100s.


## CORRECTION: it is not the node

The conclusion above was wrong. Running the same row on allium-0 -- a node that
writes 100 MB into that very directory at 566 MB/s -- wedged identically, same
`ceph_mdsc_wait_request`, same point in the run:

    child 238498 wchan=pipe_read
    child 238499 wchan=ceph_mdsc_wait_request
    child 238500 wchan=0

So "shared-ord-0's client is stuck on that inode" does not survive the test that
should have been run before writing it down.

What still distinguishes this row from the EK-FAC runs that succeeded is SIZE.
The successful ones are 4k and 16k rows; this is 32k, and it writes a 6.79 GB
`gradients.bin` before stalling. The next thing to test is whether a smaller
32k-family row wedges too, which separates "large query-gradient file" from
"this particular row".

Until that is known, EK-FAC scoring at 32k should be treated as blocked rather
than retried -- four attempts on two nodes have produced nothing but a 6.8 GB
partial each time.
