# EK-FAC works again. Ignore the two earlier notes in this directory.

Superseded:

* `2026-08-25-ekfac-wedges-on-one-nodes-ceph-client.md` — wrong, it was fleet-wide
* `2026-08-25-ekfac-stalls-at-6480MiB-reproducibly.md` — the byte count was a red
  herring, and its conclusion "only an operator can clear this" was wrong

## What it actually was

Every EK-FAC and filter job reads `query_20.hf`, and that file lives in

    /mnt/ssd-1/lucia/bergson-damping/runs/ekfac_vs_n/datasets/

which is the directory a stuck copy left unlistable. Named reads of its children
still work on most nodes, which is why some jobs survived, but the directory is
degraded everywhere and on marisa-0 *every* access to it hangs unkillably.

No node restart was needed. Copy the dataset to ssd-2 and point at the copy.

## What is already done for you

`gen_experiment_run.py`, `gen_filter.py` and `gen_ekfac.py` now resolve every
dataset through `/mnt/ssd-2/lucia/datasets_local/<name>` when a copy exists, and
never stat the ssd-1 path — probing it is the thing that hangs. `query_20.hf`
and `train_{16,32,128,256,512}k.hf` are mirrored there already.

So: regenerate configs rather than reusing an old one from a run directory. A
config written before this change still carries the ssd-1 path.

Confirmed by running `plan_adam_eps1e17_16k_bs64` end to end — all four steps,
scores written. Seven EK-FAC jobs are running now across the fleet.

## Second thing, and it is nastier

`lucia` is **uid 1001** on iris-0, secret-ord-0 and maria-1, and **uid 1000** on
the other seven nodes. CephFS stores the numeric uid and bergson writes mode
0600, so anything one group produces is unreadable to the other. It fails
instantly with `FileNotFoundError` while `ls` shows the file sitting there, which
is how it differs from a hung mount — `head -c 8 <file>` is the cheap probe.

This had already reached the banks: 42 of the 100 retrained models in
`plan_adam_eps1e17_32k_bs256` could not be opened from seven of ten nodes. A
bank is enumerated by directory and nothing checks each model opens, so a filter
run or a HuggingFace publish from those nodes would have used 58 models and
reported success.

Fixed for that bank and for the four bs32 ladder rows. Before trusting a bank
you did not build, run from a node of each uid:

    find <run_dir> -user 1001 ! -perm -o+r
    find <run_dir> -user 1000 ! -perm -o+r

Repair has to run on a node whose uid matches the files, since only the owner can
chmod:

    chmod -R a+rX <run_dir>

The durable fix is `umask 022` in the launch path so artifacts are born readable.
Worth doing before the next bank is built across mixed nodes.

Full write-up in `notes/uid_split.md`.
