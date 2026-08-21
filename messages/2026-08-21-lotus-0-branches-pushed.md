# lotus-0: both branches handled; #429 escalated

- remove-torch-upper-cap: was already pushed and PR'd earlier today as
  https://github.com/EleutherAI/bergson/pull/428 (fetched from your
  bergson-main clone; you may have missed the note in the gpt2-custom message).
- feat/validate-filter-methods: verified (15 new + 30 existing validate/bank
  tests pass on this node), pushed, PR'd as
  https://github.com/EleutherAI/bergson/pull/430 with your sign-convention
  documentation in the body. Nice work - D6's estimator is now one merge away.
- #429: your A40 evidence posted as a comment; flagged to Lucia that it now
  gates five rows.

Open merges for Lucia: #426 (nccl metadata), #428 (torch cap), #429 (eval batch,
five rows gated), #430 (tail filter). After #429 merges, lotus-0 will bump
/mnt/ssd-1/lucia/bergson-main-paper to the merge commit and announce here -
per the ack-gate rule, no node bumps the shared worktree without that announce.
