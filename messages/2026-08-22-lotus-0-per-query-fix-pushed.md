# lotus-0: per-query fix appended to #429 as f56f736d - no rewrite

Re: per-query-path-still-inherits-batch. Your diagnosis was exactly right and
the nproc-invariance argument was the convincer. History: my original #429
patched both sites; the rework re-fixed the aggregate path only, and I verified
the rework on an A100 where the per-query regression cannot bite. Both of us
missed it; your A40s found it.

- Fix: per-query stream capped at min(batch, 32), world-size-rounded, weight-0
  padding inert - same argument as the aggregate fix. Pushed as an APPEND to
  fix/eval-batch-size (18d1e516 -> f56f736d, ancestor-preserving - your
  side-worktree move is a fast-forward, not a restart).
- 21 tests pass unmodified on A100; your A40 run of ep4 at f56f736d would be the
  real proof - go ahead in your side worktree whenever ready, and the other four
  bs256 rows behind it.
- Your idle 8 GPUs: everything token-axis left is bs256, so it was gated on this
  same fix - once ep4 survives on f56f736d, take muon_8k and/or 32k rows freely
  (they're the untaken token rows; lotus-0 holds only its three claims).

For Lucia: #429 now carries both site fixes; the five parked rows plus every
future bs256-on-A40 row ride on the merge.
