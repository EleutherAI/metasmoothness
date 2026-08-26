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

## 2026-08-26 plan_adam_eps1e17_64k_bs32 :: ekfac score :: iris-0 -- LIVE CONFIRMATION of #444

    state    running (not stale, despite what check_runs.py says)
    capture  log parked at "Collecting gradients: 100%" for 59 min;
             GPU 5 at 100% utilisation, GPU 4 at 0%; parent in do_wait
    status   CLOSED as a diagnosis; the run itself is still going

This is the rank-0 asymmetry from EleutherAI/bergson#444 caught in the act rather
than inferred from a crash. The split across the two GPUs is the whole argument:

    GPU 5   100%   rank 0: process_autocorrelation_matrices + processor.save()
    GPU 4     0%   rank 1: blocked in dist.all_reduce(total_processed)

One rank saturated, the other idle, at exactly the point where teardown does
rank-0-only work. Nothing else produces that pattern.

It has now been in that section for **59 minutes**, twice the 30-minute timeout
the unpatched build.py allowed. So the 30-minute value was not merely tight, it
was well under what a 64k row needs, and every previous abort at this point was
a healthy run being killed. That settles the question the earlier entry left
open: a bigger timeout is necessary AND, on this evidence, 1-2 hours is the right
order of magnitude rather than a guess.

Two detector consequences:

  * check_runs.py called this STALE at 59 minutes because it infers health from
    log age, and the rank-0 Hessian section emits nothing by design. Silence here
    is expected, not a symptom. A long-running EK-FAC row will always trip a
    log-age threshold near the end.
  * hung_check.py would make the same mistake for the same reason. Neither should
    be trusted on an EK-FAC row sitting at "Collecting gradients: 100%" without
    checking per-GPU utilisation first -- one rank at 100% and the rest at 0% is
    the signature of healthy rank-0 work, not a hang.

The same check applies to the 512k ms probes launched today: an empty log with
the launcher in state D on folio_wait_bit_common is dataset loading, and GPU
utilisation is what distinguishes it from a hang. Log age alone cannot.

## 2026-08-26 london 32k hang :: DIAGNOSTIC RUN (updates the OPEN entry above)

Ran the single unbuffered reproduction the earlier entry called for. It
reproduces cleanly and rules out the explanation that entry was leaning on.

    python -u -X faulthandler, PYTHONUNBUFFERED=1, log OUTSIDE the run dir,
    alone on secret-ord-0 GPUs 6,7, nproc 2

Result after 160 s: **zero bytes of output**, GPUs at 0% utilisation, 130 threads
(128 in futex_wait_queue, 1 wait_woken, 1 do_sys_poll), CPU frozen at ~18 s, 0
MiB read.

What this settles:

  * It is NOT a buffering artefact. The earlier entry said the zero-byte log
    could not distinguish a startup hang from a training hang because stdout was
    block-buffered. With -u and PYTHONUNBUFFERED the log is still empty, so the
    process hangs BEFORE bergson emits its first line.
  * It is not I/O. read_bytes stays at 0 and CPU stops climbing at 18 s, so it is
    blocked, not slowly working. That is different from the 512k ms probes, which
    sit in state D on folio_wait_bit_common with read_bytes rising -- those are
    genuinely loading and do start.
  * No GPU work ever begins.

Also found while setting this up, and it explains an old mystery: **bergson wipes
the run directory at startup**. A log redirected into run_path is deleted out
from under the process, which is exactly the `train.log (deleted)` fd seen on the
hung muon 128k jobs, and why the tuning launcher writes to /tmp instead. Any
diagnostic log must live outside the run directory.

Still not root-caused. Both stack-dump routes are blocked in this container:

    py-spy dump   Permission Denied as lucia AND as root -- the container has no
                  CAP_SYS_PTRACE, so no ptrace-based tool will work here
    SIGABRT with PYTHONFAULTHANDLER=1   produced no dump and did not kill the
                  process, so the signal is not reaching a thread that can serve it

Next thing to try, in order: (1) faulthandler.dump_traceback_later() compiled into
a wrapper so the dump is scheduled from inside the process rather than delivered
by signal; (2) strace if the container permits it; (3) bisect the config -- london
16k bs256 works and london 32k bs256 does not, so halve the dataset until it
starts, which at least localises the trigger to size rather than corpus.

A note on process hygiene: my first three attempts left three competing copies on
the same GPU pair, which made the first capture meaningless. The kill loop that
cleaned them up also killed my own shell, because the command line contained the
config path the pattern matched. Third time today. Capture the PID at launch and
signal by PID; never name the target in the same command as the kill.

## 2026-08-26 plan_adam_eps1e17_64k_bs256 :: bank build :: marisa-0 -- ssd-1 PATH LOOKUP STALLS

    state    hung -> fixed
    capture  wchan=walk_component; read_bytes 0 -> 0 over 50 s; CPU frozen at
             1792 jiffies; 129 threads; GPUs 0%; log 0 bytes
    status   CLOSED (cause found and fixed)

A third distinct hang signature, and the most useful one because it is not
bergson at all.

`wchan=walk_component` is the kernel resolving a path component. Nothing was
reading, computing, or touching a GPU. The config pointed `data` and `query` at

    /mnt/ssd-1/lucia/bergson-damping/runs/ekfac_vs_n/datasets/{train_64k,query_20}.hf

and a plain `ls -d` on that path **times out with no output**. So it is the
filesystem, not the job. ssd-1 is the volume at 100% use with ~87 GB free against
970 GB of our own data.

Rewriting both paths to the ssd-2 mirror and relaunching: training at 7% within
three minutes, GPUs at 99/100%.

Three hang signatures now, and they are genuinely different mechanisms:

    wait_woken   + futex_wait x39, log 0 B, CPU frozen   london 32k/64k, OPEN
    pipe_write   + PPid 1, fd -> deleted log             orphaned launcher, CLOSED
    walk_component + read_bytes flat, CPU frozen         ssd-1 path stall, CLOSED

The discriminator that separates a hang from a slow start is read_bytes plus CPU,
not log age: a 512k ms probe sits in state D for minutes with an empty log and
read_bytes CLIMBING, and it starts fine. Frozen CPU with flat read_bytes is a
hang every time so far.

`scripts/gen_bank.py` now rewrites dataset paths to the mirror, which
gen_filter.py and gen_ms.py already did. Any config still naming a
bergson-damping path under ssd-1 is a latent instance of this.

## 2026-08-26 london 32k hang :: ROOT CAUSED -- uniform token lengths at >=32k docs

    status   was OPEN, now localised to allocate_batches

Bisected it. Four tests, each changing one variable against the same config:

    nproc 2, stale run dir     hangs        (130 thr, wait_woken, CPU frozen)
    nproc 1, stale run dir     FAILS FAST   PermissionError in shutil rmtree
    nproc 1, clean run dir     hangs        same signature
    nproc 1, london_16k        TRAINS       125 steps, GPU 100%
    nproc 1, 16k SLICE of london_32k   TRAINS

The last two are the answer. A 16000-row slice **of london_32k itself** trains
fine, so the file is not corrupt and the corpus is not the problem. **It is the
size.** 32000 rows hangs; 16000 rows from the same file does not.

Why london and not smollm2 at the same 32k: every london row is exactly 512
tokens. distinct_lengths=1 for london_16k, _32k and _64k alike. bergson packs
batches by bin-packing on length, and `data.py` carries this assertion text:

    "Could not construct a number of batches divisible by the world size. If
     variability of item lengths in your dataset is low consider using a
     different dataset size or token batch size."

Zero variability is the degenerate case that warning describes, and it only
bites once there are enough items. smollm2 rows have varied lengths, which is
why 32k works there and not here.

Note the size threshold is between 16k and 32k and has not been pinned exactly;
a bisect at 20k/24k/28k would do it, using .select() on the existing file.

Two real bugs found on the way, neither of which was the hang:

  * bergson rmtrees run_path at startup, and with the uid split a run dir created
    by uid 1001 cannot be wiped by uid 1000 -- PermissionError on config.yaml.
    At nproc 1 this surfaces as a clean crash; at nproc 2 it does not propagate
    and the ranks deadlock instead, which is a second way to get the same
    zero-byte-log symptom. All 13 stale london run dirs were removed.
  * the run dirs are created with no group write (umask 022), which is what makes
    the cross-uid wipe fail. umask 002 for run dirs would prevent it.

Workaround to try before touching bergson: the assertion text suggests a
different token batch size. That is one config change and would unblock the
london ablation at 32k/64k without waiting on an upstream fix.

## 2026-08-26 london 32k/64k hang :: SOLVED -- damaged datasets .map() cache

    status   CLOSED. Six london tuning runs now training, muon included.

Got a stack by scheduling faulthandler from inside the process
(`faulthandler.dump_traceback_later(60, repeat=True)`), which needs no ptrace and
works where py-spy is refused:

    bergson/magic/cli.py:288  attach_doc_ids_if_missing
      datasets/arrow_dataset.py:3580  map
        :3469  load_processed_shard_from_cache
          datasets/table.py:120  _memory_mapped_arrow_table_from_file
            pyarrow/ipc.py:52  __init__          <- blocked here, forever

`.map()` finds a `cache-*.arrow` beside the dataset and blocks memory-mapping it.
Moving the three cache files out of `london_32k.hf` and rerunning the identical
config: training, 16/250 steps, GPU 100%. Same for 64k.

Note the step count -- 250, not the 125 the 16k runs show -- confirming it is
using all 32000 documents rather than silently falling back.

**Correcting my earlier entry**, which said the trigger was dataset SIZE and
pointed at bin-packing. Both parts were wrong:

  * the bin-packer is fine. Called directly on 32000 uniform-512 docs it returns
    125 batches in 0.01 s. Uniform lengths are not the problem.
  * size was a proxy. london_16k also has cache files and works, so it is not
    "caches are bad" -- these particular files are damaged.

Why they were damaged is the ugly part: every hung london run was eventually
killed, and a killed `.map()` leaves a partially written cache. The next run
memory-maps it, blocks, gets killed, and the state persists. It was
self-perpetuating, which is why it survived ten hours and several relaunches.

Also refuted along the way, each with a test rather than an argument: buffering
(empty log with -u), world size (hangs at nproc 1), the run directory (hangs
clean), thread pools (hangs with TOKENIZERS_PARALLELISM=false, OMP/RAYON/MKL=1),
and the GPU or node (full 32k hangs on the same GPU where a 16k slice trained).

Caches moved to `/mnt/ssd-2/lucia/datasets_local/_stale_caches_aug26/` rather than
deleted, so this is reversible if a cache turns out to matter.

Lesson worth keeping: when a job hangs with no output and cannot be attached to,
`faulthandler.dump_traceback_later` from inside beats every external tool. Three
hours of hypothesis-testing produced nothing; the stack took one run.

## 2026-08-26 plan_adam_eps1e17_64k_bs32 :: ekfac score :: iris-0 -- WATCHDOG FIRED

    state    crashed -> relaunched with BERGSON_DIST_TIMEOUT_MIN=360
    capture  log tail "Watchdog"; GPU 5 at 100%, GPU 4 at 0% at the time
    status   CLOSED, but it revises the timeout guidance in #444

The abort this row had been threatening actually fired. It launched before the
DIST_TIMEOUT backport landed, so it still carried build.py's 30-minute value.

The number that matters comes from its muon twin, which was still ALIVE in the
same rank-0 section at **114 minutes**, GPU 3 at 100% and GPU 2 at 0%. So the
rank-0 Hessian processing and save on a 64k row takes over 110 minutes.

That means the 1-hour default I proposed in EleutherAI/bergson#444 is **too
small** for this workload -- it would abort exactly these runs. The env override
is doing the real work here; relaunched with BERGSON_DIST_TIMEOUT_MIN=360 and it
is running, both GPUs at ~100%.

Worth stating plainly: a timeout is not a fix, it is a ceiling, and I picked the
ceiling before I had a measurement of what it needs to clear. The measurement now
exists and says 2 h minimum for 64k, more for anything larger. The asymmetry
itself -- rank 0 working while every other rank blocks in all_reduce -- is what
should be removed, by doing the all_reduce before teardown or sharding the
Hessian save.

Reminder for the health scripts: the muon twin at 114 minutes reads STALE to
check_runs.py and would read hung to hung_check.py. One rank at 100% with the
rest at 0% is healthy rank-0 work. Check per-GPU utilisation before acting.
