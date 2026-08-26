# Error analysis log

Per D18 and D19: a crashed job releases its claim and files an entry here; a hung
job is killed, its state captured, and filed here. An entry is a **task**, not a
record -- it stays open until someone explains the failure or closes it as
understood.

Append newest at the bottom. Format:

    ## YYYY-MM-DD <run_id> :: <kind> on <host>
    state    crashed | hung | killed
    capture  wchan / thread histogram / fds / py-spy, or "none (pre-D19)"
    status   OPEN | CLOSED (<one line>)
    <what is known>

---

## 2026-08-26 london tuning 32k + 64k :: tune :: 13 runs across 6 nodes

    state    hung
    capture  partial -- wchan wait_woken, 130 threads (39 futex_wait), fd 45 open
             on the dataset arrow file, zero-byte log. No py-spy (not installed).
    status   OPEN

All 13 london tuning runs at 32k and 64k hang identically: nothing written to the
run directory but `config.yaml`, a zero-byte log, ~9.5 hours with no progress.
**Both optimizers**, so `notes/muon32k_hang.md` is wrong to blame muon.

London 16k and london 128k both complete. Only the 32k/64k band hangs, which is
the part that makes no sense -- if it were corpus preparation, 128k would fail
too. Datasets are intact and scale correctly (65.9 MB / 131.8 MB data arrow).

Ruled out: missing/truncated dataset, muon specifically, node, ports, launcher,
world size, dataset lock contention (no `.lock` files, no lock fds).

Not yet ruled out: whether the hang is inside dataset load or after it. The
zero-byte log cannot distinguish, because stdout is block-buffered to a file and
nothing had flushed. **Next step: relaunch ONE run with `python -u` and the log
in the run directory, alone on a node**, and see how far it gets. Until then we
do not know whether this is a startup hang or a training hang.

Seven of these were still live holding 14 GPUs. Lucia interrupted the kill, so
they were left running deliberately.

Contributing failure: the tuning launcher writes to `/tmp/tune_<name>.log`, not
the run directory, so every health sweep reported "no log" and scored them
healthy. That is why this ran for ten hours unnoticed. D20 exists because of it.

## 2026-08-26 plan_muon_eps1e17_64k_bs32 :: ekfac score :: iris-0 + lucia-ord-0

    state    killed (duplicate writer)
    capture  none needed -- cause known
    status   CLOSED (self-inflicted; see below)

Two processes were running the same config into the same `ekfac_scores`
directory and the same log, on two nodes. The shared log is why the run looked
healthy: interleaved output from both.

Cause, and it was mine, not the reservation system's. `check_runs.py` reported
the row DEAD, I deleted its claim and relaunched **without verifying the original
process was actually dead**. The claim system has no enforcement: a claim is an
`mkdir`, releasing it is an `rm -rf`, and nothing checks liveness on either side.
It records intent; it cannot prevent anything.

The gap D18 does not close: releasing on exit fixes *stale* claims, but nothing
stops a second launch against a row whose first process is alive. A launcher
should refuse when a live process already holds the run's config path -- that
check does not exist yet and is the obvious follow-up.

## 2026-08-26 64k EK-FAC watchdog abort :: both optimizers

    state    crashed
    capture  log tail -- "Watchdog caught collective operation timeout",
             ProcessGroupNCCL.cpp:733, always after "Collecting gradients: 100%"
    status   CLOSED (root cause found; fix upstream as EleutherAI/bergson#444)

Not a hang and not a network fault. After the gradient loop, `teardown()` runs
`process_autocorrelation_matrices` and `processor.save()` **on rank 0 only**,
while every other rank is already blocked in `dist.all_reduce(total_processed)`
(`collector/collector.py:916`). Rank 0 does arrive -- but the rank-0 work grows
with the dataset and the write lands on shared storage, so at 64k it exceeds the
collective timeout and the watchdog kills a healthy run.

The gradient loop was never the problem: `data.py` pads the batch count to a
multiple of the world size and asserts each rank gets an equal number, so ranks
leave the loop together.

The pinned -429 checkout had `timeout=timedelta(minutes=30)` in `build.py`, so
rank 0 is exceeding **half an hour** at 64k -- worth knowing before assuming a
bigger timeout is sufficient rather than merely necessary.

Correction to an earlier claim of mine: `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` was
the wrong knob (it governs the monitor thread). `TORCH_NCCL_ASYNC_ERROR_HANDLING=0`
does stop the abort, but by disabling the watchdog, which is a mitigation and not
a fix. The collective timeout has no env override at all.

Fixed by raising every call site to one `DIST_TIMEOUT` (1 h upstream, 2 h in the
pinned checkout), overridable via `BERGSON_DIST_TIMEOUT_MIN`. This raises the
ceiling; it does not remove the asymmetry. Doing the all_reduce before teardown,
or sharding the Hessian save, would shrink the window itself.

## 2026-08-26 tune_muon_128k_bs32 lr2.5e-05 + lr5e-05 :: tune :: secret-ord-0, shared-ord-0

    state    hung -> killed
    capture  wchan pipe_write; 130 threads (128 futex_wait_queue, 1 pipe_write,
             1 do_sys_poll); fd 1 and 2 -> train.log (deleted); PPid 1
    status   CLOSED (mechanism understood), but see the follow-up below

Found by `scripts/hung_check.py` within minutes of the script existing, which is
the entire argument for D20 -- neither had a claim, so no claim-based check could
ever have seen them. Quiet for 644 and 323 minutes, holding two GPUs each.

Mechanism, fully determined by the capture:

  * PPid 1 -- the launching shell is gone, these were orphaned
  * fd 1/2 point at a DELETED train.log, so their output had nowhere to land
  * the main thread sits in `pipe_write`

The child writes stdout through a pipe to its launcher rather than straight to
the file. When the launcher died the read end stayed open but nothing drained it,
so once the 64 KB pipe buffer filled the child blocked in `pipe_write` forever. A
fully closed read end would have delivered EPIPE and killed the process; a
half-open, undrained one hangs it instead. That is why these held GPUs for ten
hours rather than dying with their parent.

Two consequences worth carrying forward:

  * a run whose launcher dies does not die, it hangs. Anything relying on "the
    process exits when its parent goes away" is unsound here.
  * deleting a log out from under a live job (disk cleanup did this) destroys the
    only evidence and does not free the space, since the fd is still open.

NOT the same signature as the london 32k/64k hangs, which show `wait_woken`, 39
threads in `futex_wait`, and intact non-deleted logs. Two distinct failures; do
not merge them.

Follow-up, still open: launch children with output going directly to a file
rather than through the launcher, so an orphaned rank cannot block on a pipe.
