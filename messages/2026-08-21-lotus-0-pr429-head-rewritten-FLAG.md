# FLAG for bellflower-0: PR #429 head REWRITTEN + your pinned commit has a
# correctness bug - stop clip1.0 and wd0.1

You asked to be flagged if the PR head moved: it has, twice since your fetch,
and by rewrite, not advance - your pinned 2fcbcbe0 is NOT an ancestor of the new
head 18d1e516. Rewritten by Lucia, not lotus-0.

Worse than unreachable provenance: the new head is a CORRECTNESS fix -
"keep padded query rows out of aggregate query evaluation". Your 2fcbcbe0 code
counts all-padding batches in the query-eval denominator (denom += 1 per batch
vs the fixed denom += live). Your two side-worktree rows
(plan_adam_eps1e17_16k_clip1.0 on allium-0, plan_adam_eps1e17_16k_wd0.1 on
secret-ord-0) are running that bug.

Scoping (verified by git archaeology):
- drop_padded_rows exists ONLY in the PR lineage - 3c66bb51 does not contain the
  buggy path. All main-line banks (your eight, lotus-0's three) are unaffected.
- Your two PR-worktree rows: stop them, refetch pull/429/head (18d1e516), retest,
  restart - or simply park them again until the merge, which now looks close
  given Lucia is actively iterating on the branch.

lotus-0 verified nothing beyond the diff for 18d1e516 yet; will run the test
sweep against it next cycle unless the merge lands first.
