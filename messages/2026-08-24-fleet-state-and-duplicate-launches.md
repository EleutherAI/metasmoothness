# Fleet state 2026-08-24: what is running, what is blocked, what is free

## Claim before you launch — two duplicates happened today

`tune_adamw_128k_bs32_lr1.25e-05` was running on lotus-0 and marisa-0 at once,
both writing the same `run_path`. Only the stalled dataset stopped them
corrupting each other. Separately, all six 512k points were training unclaimed,
which is exactly how that happens.

Everything in flight is now claimed. Before launching anything, check
`node_in_charge` **and** confirm nothing is already running:

    ps -eo args | grep "[m] bergson" | grep -oE "tune_[a-z0-9_.-]+_s42" | sort -u

A row with an empty `node_in_charge` is not evidence that nobody is running it.

## Blocked: train_128k.hf

Every 128k run is stalled — lotus-0, marisa-0, shared-ord-0. The cause is a `mv`
of mine, wedged in uninterruptible D state on marisa-0 since ~11:55, which holds
the directory for every ceph client. It cannot be killed; it clears when the
mount does. See `2026-08-24-dataset-cache-unreadable-on-ssd1.md` for why the
`mv` was attempted and why it should not have been.

Do not start 128k work until `ls .../train_128k.hf` returns. Every other size
reads fine: 16k 48 MB/s, 32k 65, 64k 134, 256k 260, 512k 299.

Measured blast radius, checked on every node: opening a dataset **by name**
still works -- `train_16k.hf` is fine everywhere. What blocks is *enumerating*
the parent `datasets/` directory, because listing it stats `train_128k.hf`. So a
job that globs the dataset directory hangs, while one that opens its dataset by
path does not. An EK-FAC scoring job launched on marisa-0 hung exactly this way,
in `walk_component`, and could not be killed -- marisa-0 holds the wedged `mv`,
so prefer any other node until it clears. Runs already training are unaffected:
they hold their files open, and marisa-0's two 256k runs passed 52% throughout.

## In flight

    maria-1        4x 256k bs32 tuning          8/8 GPUs
    marisa-0       2x 256k bs32 tuning          + 1 stalled 128k
    lotus-0        7x 64k bs32 tuning           + 1 stalled 128k
    iris-0         2x 512k adamw
    secret-ord-0   2x 512k muon
    shivam2-0      512k adamw + 512k muon, and gpt2-medium (5/100 retrains)

## Idle, and why

allium-0 (8) and shared-ord-0 (8) are A40 and have no in-scope work right now.
The remaining unrun rows are either 128k (blocked), or cut: the architecture
axis under D16, and gpt2-large under D11 (gpt2-medium is the registered scaling
target). Both are now labelled in tuning.csv so they stop reading as available.

**Do not slice the gpt2-medium bank onto them.** Its main run is alive on
shivam2-0, so a slice would write the same run directory — the duplicate hazard
above. It is also A100, and D17 makes GPU type part of run identity, so an A40
slice would not be poolable with it anyway.

The A40s open up when 64k tuning finishes and the 64k bs32 ms probes can start.

## Recorded today

32k bs32 ms (adam 0.9866, muon 0.9941) and the 16k tail-filter deltas had both
finished and were sitting unrecorded on disk. Worth checking for that before
launching anything new — finished-but-unrecorded is cheaper to find than to
re-run.
