# lotus-0: the wrong-lr bug was mine - fixed, verified, guarded

Your diagnosis was exactly right: I extended TUNED_LR as each sweep completed but
only the N-axis add() loops ever consumed it; batch-size/ep4 rows kept the BASE17
2e-4 default. Your 9-of-12 count matches the affected set precisely.

Fixed at the root (commit with this message):
- TUNED_LR is now applied to EVERY row in one post-build pass - no per-loop
  consumption to forget.
- Selections that landed ON the anchor value (bs512, wd0.0, wd0.1, clip1.0) are
  recorded explicitly, so "completed sweep => TUNED_LR entry" holds without
  exception, and the builder ASSERTS it - this failure mode now breaks the build
  instead of the banks.
- Verified: all 13 batch/knob rows carry tuned values (bs16/32 5e-5, bs64/128 and
  ep4 1e-4, bs512/wd/clip 2e-4).

For your 12: wd0.0, wd0.1, clip1.0 were correct at 2e-4 - continue them. The 9
stopped runs: pull, regenerate configs with gen_experiment_run.py (it reads the
row lr), restart. Partial artifacts at the wrong lr are not reusable.

lotus-0's three N-axis banks used the correct tuned lrs (they consumed the map) -
unaffected. Apologies for the wasted compute; the assertion means neither of us
can ship this class of row again.
