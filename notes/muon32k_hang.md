# muon + london_32k hangs before it writes anything; adamw on the same config does not

Parked, not solved. Recording it so nobody re-derives the diagnosis.

## Signature

    tune_muon_london32k_bs256_lr{4e-4,8e-4,1.6e-3}
      log 0 bytes, GPU 3 MiB (CUDA context only, no allocation), flat for 4+ min
      processes alive, no error, no progress

Reproduced on **two nodes** (iris-0, shared-ord-0), on fresh GPUs, with fresh
MASTER_PORTs, after clearing every stale process and the run directory. So it is
not the node, not a port collision and not leftover state.

## What it is not

* **Not the launcher.** First seen through a `sh -c` wrapper that also chmod-ed
  after the run; reverting to a direct `setsid nohup env python` launch reproduced
  it exactly. The wrapper was reverted anyway -- see scripts/launch_tuning_pair.sh
  -- but it was not the cause.
* **Not the config.** `diff` against the adamw config of the same rung is one
  line: `optimizer: adamw` vs `optimizer: muon`. Everything else -- dataset,
  batch, ga, epochs, seed, distributed block -- is identical.
* **Not muon generally.** `tune_muon_london16k_bs256_*` all five points ran to
  completion on the same corpus and batch size.
* **Not london generally.** All three `tune_adamw_london32k_bs256_*` points
  preprocess and train normally.

So it is specifically muon AT 32k, and the pair (optimizer, N) is what selects it.

## Next things to try

1. nproc 4 instead of 2 -- if it starts, it is a world-size interaction rather
   than the optimizer.
2. muon on smollm2 `train_32k.hf` at bs256, which isolates corpus from N.
3. Attach to the hung process and get a Python traceback (`py-spy dump`), which
   would say whether it is stuck in dataset loading, optimizer init or the
   distributed rendezvous. The 0-byte log means it has not reached any print, so
   the stall is very early.

Cost of leaving it parked: the muon arm of the london N-scaling question. The
adamw arm at 32k is running and answers the same question one optimizer at a time.
