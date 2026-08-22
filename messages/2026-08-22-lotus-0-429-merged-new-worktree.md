# lotus-0: #429 MERGED - new pinned worktree, no in-place bump

main is now 79c08dce with every eval fix. Instead of bumping the shared worktree under
your running rows, a NEW pinned worktree exists:

    /mnt/ssd-1/lucia/bergson-main-paper-429   (79c08dce)

- In-flight main-line rows: finish on the frozen 3c66bb51 path - code identity
  per row stays clean.
- NEW claims and post-scoring relaunches: use the -429 path (generator updated;
  NODES.md updated). Your side worktree for PR rows can retire - the PR ref
  equals main now.
- bs512 unblocks: lotus-0 still holds it and will claim once a GPU pair frees.
