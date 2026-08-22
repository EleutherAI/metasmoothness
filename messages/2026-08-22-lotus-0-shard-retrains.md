# Shard the retrain bank across GPUs: subset_start / subset_stop

From: lotus-0. Date: 2026-08-22. Ruling from Lucia: split retrains across nodes.

The 100 retrains are independent and bergson already slices them
(`ValidationConfig.subset_start/subset_stop`, one `validation_<a>_<b>.csv` per
process). Recipe, tooling and the merge step are in NODES.md "Sharding a bank's
retrains"; `scripts/slice_bank.py` writes the slice configs and prints launch
commands; `scripts/magic_lds.py <run_dir>` merges slices and asserts all 100
subsets are present exactly once.

Two rules learned the hard way on the 8k bank (commit 9620d05):

1. Launch slices one at a time - wait for `Validating` in the previous slice's
   log. Each slice resumes from the last checkpoint and re-saves it; two at once
   corrupt it (`PytorchStreamReader ... miniz error`). Relaunch is clean.
2. Same nproc as the bank (D15), and never two processes on the same subset
   index.

Relevance to the fleet: once a row finishes scoring, every idle GPU on its node
can take a slice of its bank. A bs256 bank at 4 slices drops from ~13 h to ~3.5 h;
the small-batch rows (bs16/32/64, 2000-500 steps per retrain) gain the most.

Live now: 8k adam bank - main process at subset ~60 (will be stopped at 72),
slices 72-86 on GPU 0 and 86-100 on GPU 1.
