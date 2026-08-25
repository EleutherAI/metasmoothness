# muon + london_32k stalls before it writes anything; adamw on the same config does not

Parked, not solved. Recording it so nobody re-derives the diagnosis.

## Signature

    tune_muon_london32k_bs256_lr{4e-4,8e-4,1.6e-3}
      log 0 bytes even with PYTHONUNBUFFERED=1 and python -u
      GPU 3 MiB (CUDA context only, no allocation), flat for 4+ minutes
      processes alive, no error, no progress

Zero bytes *unbuffered* is the useful part: the process reaches no print at all,
so the stall is very early -- import, config load, or dataset open. A config error
or a missing file would have raised instead.

## What it is not

* **Not the node.** Reproduced on iris-0 and shared-ord-0.
* **Not a port or stale state.** Fresh MASTER_PORTs, fresh GPUs, run directory
  deleted between attempts.
* **Not the launcher.** First seen through an `sh -c` wrapper; reverting to a
  direct `setsid nohup env python` launch reproduced it exactly.
* **Not world size.** nproc 4 stalls the same way as nproc 2.
* **Not muon generally.** All five `tune_muon_london16k_bs256_*` points completed.
* **Not london generally.** All three `tune_adamw_london32k_bs256_*` points run.
* **Not muon-at-32k generally.** `plan_muon_eps1e17_32k_bs256` has a full bank on
  smollm2 at the same optimizer, N and batch.

So it needs muon AND london_32k together, and neither alone.

## The lead worth following

`london_32k.hf` accumulates `cache-*.arrow` files: bergson/HF-datasets write cache
and lock files INSIDE the dataset directory. Three adamw jobs were reading that
directory when the muon jobs were launched. HF datasets guards those writes with
filelock, and a held lock blocks silently and forever -- which matches "no output
at all, indefinitely" far better than anything optimizer-specific does.

If that is right the muon jobs were queued, not hung, and killing them was
premature.

The obvious test -- point muon at a private copy of the dataset -- did NOT settle
it, because copying the arrow file while three jobs have it memory-mapped
produces a torn copy that fails with
`ArrowInvalid: Tried reading schema message, was null or length 0`. A clean test
needs the copy made while nothing else is reading the directory.

## Next steps

1. Re-run one muon 32k point when no adamw job is touching `london_32k.hf`. If it
   starts, it is contention and the fix is per-job dataset copies or a queue.
2. `py-spy dump` on a stalled process would name the lock directly.
3. If it really is filelock, `datasets` can be pointed at a writable cache dir
   outside the dataset directory, which removes the contention entirely.

Cost of leaving it parked: the muon arm of the london N-scaling question. The
adamw arm at 32k is running and answers the same question one optimizer at a time.
