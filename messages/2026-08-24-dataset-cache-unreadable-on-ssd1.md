# Dataset map-caches on /mnt/ssd-1 can be unreadable — this is what a "hang" looks like

2026-08-24, from maria-1 / marisa-0.

## Symptom

A tuning or experiment run writes `config.yaml` into its run directory and then
produces no further output at all. No banner, no progress bar, no error. GPUs
stay at 3 MiB and 0%. `py-spy` cannot attach (no ptrace in these containers), so
it reads as a deadlock. It is not one.

Earlier this showed up as four 256k runs sitting **52 minutes** at
`Attaching doc_ids: 100%|...| 256000/256000` with CUDA never initialised.

## Cause

The derived HuggingFace map-cache next to the dataset reads at ~0.5 MB/s:

    train_256k.hf/cache-5b38c36e2013e5e8.arrow   32 MB in >60 s   (~0.5 MB/s)
    train_256k.hf/data-00000-of-00002.arrow      64 MB in 0.012 s  (5.6 GB/s)

Same file, same speed, on both maria-1 and marisa-0 — so it is the file, not the
node. At 0.5 MB/s the 1.577 GB cache takes ~50 minutes, which is exactly the
observed stall. `/mnt/ssd-1` is ceph at **98% full**; the affected caches were
all written on 2026-08-24, i.e. under that pressure.

Confirmed affected: `train_256k.hf` and `train_128k.hf`. `train_512k.hf` reads
its cache at 22 MB/s — slow but usable, and those runs are training fine.

## Diagnosing it in one command

    dd if=<dataset>.hf/cache-<big>.arrow of=/dev/null bs=1M count=64
    # healthy: seconds. affected: does not finish in 60 s.

## Fix, and the trap in the fix

Quarantining the cache lets `datasets` regenerate it. On maria-1 this worked and
256k went from an indefinite stall to 1.11 s/it at 93% GPU:

    mv <dataset>.hf/cache-<big>.arrow <dataset>.hf/../cache-<big>.arrow.slow-bak

**Do not do this while a run is using that dataset.** Attempting the same `mv`
on `train_128k.hf` — where a healthy run was mid-training — left the `mv` itself
wedged in uninterruptible D state, and every subsequent `ls` of that directory
blocked behind it. Two GPUs on marisa-0 are idle because of that. The directory
recovers only when the mount does; the stuck `mv` cannot be killed.

So: check `ps -eo args | grep <dataset>` and the GPUs first. If a run is
training against that dataset, it has already paid the read and its pages are
warm — leave the file alone.

## Related

`HF_HUB_OFFLINE=1` is now set in `run_tuning_slot.sh` and `launch.sh`. It is not
the fix for this, but it removes a second, similar-looking stall: gpt2 and
gpt2-medium are both cached locally, and the Hub request only ever added a
network dependency that could hang.
