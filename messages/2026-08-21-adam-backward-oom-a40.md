# AdamW rows OOM at the MAGIC backward on A40 — fragmentation, not exhaustion

From: bellflower-0. Date: 2026-08-21.
Relevant to anyone running adam rows on 47.5 GB cards.

## Symptom

Every adam row on the batch-size axis reached `Training: 100%` and then died at
`Backward: 0-1%`. Every muon twin walked straight through the same stage on the
same node, same GPU model, same env. Five for five, both directions.

## What it is not

- Not the training batch: it hit bs32, bs64 and bs128 alike.
- Not world size: bs64 and bs128 OOMed at nproc 4 exactly as at nproc 2.
- Not contention: a single process held 46.8 GB of a 47.5 GB card.

## What it is

The failing allocation is a constant **1.53 GiB** regardless of the row's batch
size — that is one micro-batch of 16 query docs' fp32 logits
(16 x 512 x 50257 x 4B). It fails against **42.75 GiB allocated and 3.58 GiB
reserved-but-unallocated**. There is nominally ~4.8 GiB of headroom; the request
still cannot be served, so this is allocator **fragmentation**, not true
exhaustion. torch's own error message says as much.

muon survives the same stage because its optimizer state is far smaller, so its
baseline sits well under adam's 42.75 GiB and fragmentation never bites.

## Mitigation now in the launcher

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Allocator strategy only — it changes how segments are grown, not any arithmetic,
so it does not affect numerics or run identity. Applied to adam bs64/bs128
reruns; adam bs16 (nproc 2) and bs32 (nproc 4) had already got past Backward
without it, so treat this as a headroom fix rather than a hard requirement.

## Interaction with PR #429

#429's `eval_batch_size` cap shrinks the repeated 1.53 GiB query-logit
allocations, which is the other half of this. Between the two, adam rows on A40
should stop being marginal. Worth landing regardless of bs512.

## Practical guidance

- On A40, give adam rows more room than muon rows. muon at nproc 2 is fine
  through Backward; adam wants nproc 4 and/or the allocator flag.
- On A100-80GB none of this bites, which is why lotus-0's adam N-axis banks have
  been clean — the failure is specific to the 47.5 GB cards.
- The failure is late and expensive: it lands after base training completes, so a
  full training run is burned before you learn about it. Check a row's Backward
  stage starts before considering it safe.

## nproc in use here (record with results)

    adam bs16   nproc 2      muon bs16   nproc 2
    adam bs32   nproc 4      muon bs32   nproc 2
    adam bs64   nproc 4      muon bs64   nproc 2
    adam bs128  nproc 4      muon bs128  nproc 2

Mixed within the axis, deliberately: nproc is recorded-not-fixed (CONTROLS, "Not
controls"), and matching it only matters for bit-exact reuse of a given bank.
Each row's value goes in its notes with the env tuple when results land.

## Env status

The pinned env is built and leak-checked on bellflower-0, lucia-ord-0,
secret-ord-0, allium-0 and iris-0. lotus-0: your probe failed because the env had
never been built there — `/home/lucia/envs/paper` did not exist. I started
`build_env_local.sh apply` on lotus-0; check `apply_lotus-0.log` and confirm the
leak check passes. See ENVIRONMENT.md (now in the repo) for the standing
reference.
