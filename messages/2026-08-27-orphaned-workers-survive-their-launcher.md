# Killing a bergson launcher does NOT stop the run

This is the systematic failure behind several "silent deaths" I have been
mis-diagnosing, and it caused a second-writer incident today.

## What actually happens

bergson launches distributed workers via multiprocessing. The workers' argv is:

    /home/lucia/envs/paper/bin/python -s -P -c from multiprocessing...

There is no "bergson" in that string. Every liveness census I have run is some
form of

    ps -eo args= | grep -c "[b]ergson"

which counts LAUNCHERS ONLY. Kill the launcher and the workers are reparented to
init (ppid=1) and keep going: still on the GPUs, still training, still writing
to the run directory and the log. The census then reports the job as dead.

## What it cost today

1. I "killed" plan_muon_eps1e17_64k_bs256/bank_shard_0_10 - the misconfigured
   shard with no subsets.json, which would have removed different documents
   than its peers. `pgrep` found it, but the kill was
   `kill -9 $P` under zsh where $P had a trailing space -> "illegal pid", and I
   had written `2>/dev/null`, so the error was invisible. It ran for another
   two hours as a SECOND WRITER into the same bank directory that bank_build
   was building, and survived my `rm -rf` of that directory.
2. I then read a log tail showing steps 240/500 at 36 minutes elapsed from a
   bank_build I had relaunched 60 seconds earlier, and briefly believed the
   node was contended by a foreign tenant. It was my own orphaned workers.

No data was lost - the bank had 0 retrained models at every point - but I wiped
and restarted the muon bank rather than trust a directory two processes had
been writing to.

## Rules

1. NEVER suppress stderr on a kill. `2>/dev/null` on a kill hid the failure that
   started this.
2. zsh does not word-split unquoted variables. `kill -9 $P` with a multi-PID or
   trailing-space $P fails. Use `pgrep -f ... | xargs -r kill -9`.
3. Do not put the target string in the command line that kills it - `pgrep`
   matches your own shell and you kill yourself. I did this again today via an
   `echo` that merely MENTIONED the run name. The `[b]racket` trick only helps
   if the literal name appears nowhere else in the command.
4. Verify a kill by absence, not by the kill's exit code:

       ps -eo pid,ppid,etime,rss,args= | grep multiprocessing | awk '$2==1'

   Orphaned workers have ppid=1. That is the check that finds them; grepping
   for "bergson" never will.
5. After killing, confirm the GPUs actually released (util AND memory to 0).
   A launcher-only kill leaves them pinned.

## marisa-0: unkillable, separate problem

marisa-0 has a bank_build for plan_adam_eps1e17_64k_bs256, 21h46m old, in state
**DN** - uninterruptible sleep, i.e. wedged in I/O and immune to kill -9. All 8
GPUs sit at a uniform 54432 MiB.

I previously attributed marisa-0 to a foreign tenant based on PID invisibility.
That argument was weak: nvidia-smi reports HOST pids while `ps` in the container
shows namespaced ones, so the pids never match, for my own jobs too. I checked
for orphaned workers there and found none, so the GPUs are not mine to reclaim
either way. The operational conclusion is unchanged - do not launch on marisa-0 -
but the reason is "a wedged D-state process and GPUs we cannot account for",
not a confirmed second tenant.
