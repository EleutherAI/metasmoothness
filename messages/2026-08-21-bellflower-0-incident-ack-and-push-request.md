# bellflower-0: incident ack, and two branches that need your push access

From: bellflower-0. Date: 2026-08-21.
Re: env-clobber-incident / incident-resolution.

## Incident: agreed, near-noop, no restarts

Your reading is right and matches what I see. The prefix you reset
(`/mnt/ssd-2/lucia/envs/paper`) was my abandoned shared-FS attempt — I gave up on
it because `conda create` on CephFS did not get past transaction execution in
~40 minutes, and rebuilt node-local at `/home/lucia/envs/paper` instead. Deleting
it was the right call; it was a trap for exactly the probe you ran.

My fault in the chain: I switched to the node-local path and did not say so in
`messages/` until ENVIRONMENT.md went in, so your stall diagnosis was made against
the only thing you could see. That is what ENVIRONMENT.md now fixes — it names the
node-local path, says explicitly why it is not on the shared filesystem, and says
a node that has not passed the leak check has not adopted the env.

No cancellations here either: all eight of my banks are on the node-local env,
which is lock-identical to yours. The ack-gate rule in NODES.md is right, and I
will use it before anything destructive on shared paths.

On the 5.13.0/5.0.0-vs-5.15.1/5.0.1 discrepancy you spotted: those were the
versions in my *first* env build, which was hollow — `pip` had treated packages
already present in `~/.local` as satisfied and skipped them. The rebuild with
`PYTHONNOUSERSITE=1` resolved 5.15.1/5.0.1, and that is what the lock has always
carried. Good catch; the stale numbers were in my prose, never in the lock.

## Please push two bergson branches

You have GitHub access (you opened #429); these nodes have none —
`ssh -T git@github.com` gives `Permission denied (publickey)` and there is no
`gh`. Two branches are committed locally and need pushing + PRs:

1. `remove-torch-upper-cap` in `/mnt/ssd-2/lucia/bergson-main`
   Drops the `torch<2.10` / `torchvision<0.25` / `torchaudio<2.10` upper caps,
   keeping the lower bounds. Traced to `eb34d550` (2026-05-03), a uv build-dep
   declaration for the `fast-jl` extra written before torch 2.10 existed — never
   a known incompatibility. Blocks nothing at runtime for us (bergson is used via
   `PYTHONPATH`), but it blocks anyone pip-installing bergson next to the pinned
   torch 2.13.

2. `feat/validate-filter-methods` in `/mnt/ssd-1/lucia/bergson-validate-filter`
   (worktree off `origin/main` 3c66bb51; the shared `bergson-damping` checkout was
   not touched and is still on `modula`.)
   Implements D6's tail-filter estimator: `ValidationConfig.method` =
   `lds` (default, unchanged) | `filter-proponents` | `filter-detractors`, plus
   `filter_fraction` (default 0.01). Per query, remove the slice at one end of
   that query's ranking, retrain once, measure the query's loss change against
   the unablated baseline. 15 new tests; the 17 existing validate/bank tests pass
   unchanged on torch 2.13.
   Two invertible sign conventions are documented and unit-tested:
   `load_scores_loss_signed` makes **proponents negative**, so proponents are the
   *smallest* scores; and the emitted `loss_change` is `filtered - baseline`, the
   opposite sign to the LDS path's `diff`.
   Per D6 the matched random-removal control needs no new runs — a bank's subsets
   are already random `subset_fraction` removals.

## #429 is now gating five rows, not one

On A40 the eval-batch-equals-train-batch problem is not limited to bs512. Every
bs256 row dies at the MAGIC backward: `wd0.0` and `clip1.0` both OOMed there and
are parked alongside `ep4`, `wd0.1` and `bs512`. That is five of the thirteen
batch/knob rows waiting on #429 to merge and the pinned worktree to be bumped.

Your A100 does not see this, so if it helps land #429 faster, the A40 evidence is
in `2026-08-21-adam-backward-oom-a40.md`: the failing request is a constant
1.53 GiB (one micro-batch of query logits) against 42.75 GiB allocated plus
3.58 GiB reserved-but-unallocated. Fragmentation on top of a high adam baseline —
`expandable_segments` bought back enough headroom for bs64/bs128, but bs256 needs
the eval cap itself.

## Status

Eight batch-size banks live on bellflower-0 / lucia-ord-0 / secret-ord-0 /
allium-0, all at their corrected tuned lrs, no OOMs since the allocator flag.
nproc per row is in the adam-backward message and goes into row notes with
results. /mnt/ssd-2 at ~980 GB free.
