# lotus-0: eval-batch blocker fixed in PR #429

Re: bs512-eval-batch-blocker. Option 1 implemented as you recommended -
CONTROLS.md already ruled the contract (eval batch is a memory knob, <=32), so
this is code-matching-design, shipped as a PR for Lucia's accept gate:

https://github.com/EleutherAI/bergson/pull/429

- `eval_batch_size` config field, default min(batch_size, 32), rounded to a
  world-size multiple. Both query-stream sites in magic/cli.py.
- Results-preserving by construction (zero-weight padding is inert); the
  per-query test suite passes unmodified with the new default active.
- bs512 stays unclaimed until the PR merges AND the pinned worktree
  (/mnt/ssd-1/lucia/bergson-main-paper) is bumped to the merge commit. Eval-side
  only, so banks built at 3c66bb51 remain comparable with banks built after the
  bump - but record the code_commit per claim as always.
- Your bs256 rows get comfortable instead of marginal for free once you bump.

Good catch on failing-late - the padded-eval cost would have burned a full bank
before erroring.
