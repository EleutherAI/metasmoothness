# Working rules for this repo

Short, hard-won rules. Each one exists because it was violated and cost real GPU
time. Prefer the tool that enforces a rule over remembering it.

## Filters: 3 controls per ROW, not per shard

A proponent filter is `queries` retrains plus 3 control retrains for the whole row.
The controls are 3 trained models; scoring them against all 20 queries is forward
passes and costs nothing. bergson retrains them per process unless `retrained_dir`
is set, so sharding S ways turns 3 controls into 3S full trainings -- 30 instead of
3 on the 128k row at S=10.

**There is no per-shard baseline retrain.** An earlier version of this file said
there was. That was wrong: the baseline is a forward pass on the already-trained
model (`per_doc_query_losses` inside `fwd_state.activate`). The claim came from a
buffered log where the filter loop bar appeared after a completed training, making
that training look like a pre-loop baseline. It was query one. An old-shape filter
is 5 deep and 50 runs, not 6 and 60.

Build the bank once and let every shard read it:

    python scripts/gen_bank.py <run_id> --num-subsets 3 --subset-fraction 0.01
    python scripts/shard_filter.py <run_id> --controls shared      # the default

`shard_filter.py` refuses `--controls per-shard` when the plan exceeds
`queries + controls`. **If a script only prints a number you should have reacted
to, that is a bug in the script.**

## Verify that a launch actually started

Three launches have silently no-op'd here: an abort filtered out because the caller
grepped for the success line, a kill loop that parsed an empty pid from
right-aligned `ps` output, and config mutations written after the file. Every one
reported success and nothing re-checked.

`scripts/launch_one.sh` registers each launch in
`paper_runs/_logs/launch_registry.tsv` and exits non-zero with `LAUNCH-FAILED`, so
a filtered pipe cannot swallow the failure. Ten minutes later run

    scripts/check_launches.sh

which marks each entry GOOD (alive and holding GPU memory), DONE (exited with real
log output) or **DEAD** (gone, empty log -- it never started). DEAD is the silent
failure and the only state that needs a human. Never conclude a run is progressing
from its own stdout.

## Sharding a filter needs BOTH halves sliced

Slicing only the query dataset leaves `scores` pointing at the full 20-column file.
Whenever `validate_scores` runs it asserts one score column per query and dies --
**after** the shard has trained, so every failure costs a full retrain. Use
`scripts/shard_scores.py` for the score slices; `shard_filter.py` now wires each
shard to its own slice and refuses if one is missing.

## Which bergson checkout

* filter / validate steps: `/mnt/ssd-2/lucia/bergson-filter`
* training and EK-FAC scoring: `/mnt/ssd-1/lucia/bergson-main-paper-429`

A filter config carries `method`, which only exists on `Validate` in
bergson-filter. Against the training checkout it either dies immediately or --
worse, on a tolerant checkout -- silently drops `method: lds` and computes
something other than what the config says. Select it from the config, do not
remember it:

    grep -qa '^ *- *validate:' "$CFG" && BERG=/mnt/ssd-2/lucia/bergson-filter

## Never trust a partial bank

A partial bank is not a noisy version of a finished one; it can be **precisely
wrong** with a tight interval. Dropping 8 subsets from a finished 100-subset file
reproduces a bogus 0.1085 against a true 0.4146. `merge_bank.py` prints INCOMPLETE
-- if you see it, there is no number to read. The muon 64k bs256 bank is 57/100:
`magic_lds.py` will return a clean-looking 0.8957 for it. Discard that.

## The uid split

lucia is uid **1001** on iris-0 and secret-ord-0, uid **1000** everywhere else.
Consequences that have all actually happened:

* `model.safetensors` written mode 0600 by uid 1000 reads as **FileNotFoundError**
  from a uid-1001 node. Not a permissions error -- a missing-file error.
* A job writing into a dir owned by the other uid fails with PermissionError, so
  merges and recoveries must run on the side that owns the output directory.

After producing any model or bank, `chmod -R g+rwX,o+rX` it.

## Reading fleet state

* **A single `nvidia-smi` utilisation sample is worthless.** It has twice read
  3/8 GPUs busy and 42% on nodes that were at 99% across a dozen samples.
* **Allocated is not progressing.** Ten shards once held GPUs for 40 minutes while
  failing in a loop. Cross-check compute contexts against live launcher processes.
* **A log tail is not liveness.** stderr is block-buffered at 8KB, and a relaunch
  that truncates a log leaves `tail` showing the dead run's output.
* **A kill that reports success is not a kill.** `ps` right-aligns PIDs, so
  `${line%% *}` yields an empty string for a 6-digit PID and every kill silently
  no-ops. Parse with `read -r pid rest`, then re-check `nvidia-smi`.
* Killing a launcher **leaves its torch workers holding GPU memory**, reparented to
  init. Sweep for `ppid==1` processes whose argv contains `multiprocessing`.

## Launching

Use `setsid`, and write logs **outside** `run_path` -- bergson clears `run_path` at
startup and unlinks any log inside it, after which the run writes to a dead inode
and looks silent while training normally. A run path that already exists is a hard
`FileExistsError`, so a preempted job leaves a dir that must be cleared before
relaunch.

## Standing decisions

* **D17** GPU type is part of run identity. A row's base and its retrains must match.
* **D22** No new 100-retrain banks and no new LDS. A 3-subset *control* bank is not
  one of these -- it is strictly fewer retrains than the alternative.
* **D23** Never write to /mnt/ssd-1. Reading it is fine.
* Learning rates in `experiments.csv` are final; use the row's lr as-is.
* Allowlist: lotus-0, lucia-ord-0, secret-ord-0, allium-0, shared-ord-0,
  bellflower-0, iris-0. marisa-0 and shivam2-0 are permanently off. On iris-0,
  GPUs 0-2 belong to another user.
