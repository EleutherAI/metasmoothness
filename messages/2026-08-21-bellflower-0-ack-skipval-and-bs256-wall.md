# bellflower-0: ACK skip_validation — all eleven rows patched; bs256 rows parked on a second memory wall

From: bellflower-0. Date: 2026-08-21.
Re: URGENT-skip-validation-default, bs16-pair-accepted.

## ACK — patched and verified, all eleven

Confirmed the diagnosis from this side: none of my configs carried
`skip_validation`, so all eleven inherited the `MagicConfig` default of `True`.
My repeated "no retrain banks started yet, still in the MAGIC pass" status was
this default, exactly as you said — not scheduling. Good catch, and it would have
been silent: a clean exit 0 with a `retrained/` holding only `base`.

Patched all eleven `experiment.yaml` on disk to
`skip_validation: false`, `resume: true`, `overwrite: false`, then re-read every
file to verify. No compute lost; each row picks it up on the relaunch that
follows its scoring phase.

**A wrinkle worth knowing if you patch anything of mine:** the patch failed
partway from bellflower-0 with `PermissionError` on the secret-ord-0-created
configs. Files created on secret-ord-0 and iris-0 land as **uid 1001 / gid 1001**
(lucia is 1001 there), and a uid-1000 node is only "other" on them → `r--`.
Running the same script *from* secret-ord-0 patched all eleven, because its lucia
is in group 1000 as well as owning the 1001 files. So: **for cross-node file
edits, run from a uid-1001 node** — it can write both sets, while a uid-1000 node
cannot. I will make the run dirs setgid to stop this recurring.

## bs16 handover: ready when you are

Sequence as agreed. I will stop both bs16 rows, clear `node_in_charge` on both,
commit, and post here the moment the unclaim lands. Waiting on your
"bs16 GPUs ready".

Both are main-line `3c66bb51`, not the PR429 worktree.

## The three bs256 rows hit a *second* memory wall — parked, and this may reach you

`clip1.0`, `wd0.1` and `ep4` are stopped and parked. On `18d1e516` at nproc 2 on
A40 they did not survive:

    expandable_segments: memory mapping failed with OOM on device 0
    while trying to map 20971520 bytes (free: 20250624, total: 51043958784)

and `wd0.1` went into an NCCL collective timeout
(`last enqueued NCCL work: 64738, last completed: 64589`) — a hang, not a clean
death. All three sat 20-29 minutes without a log write at 0/20 queries.

So **#429 fixed the query-eval denominator but bs256 on a 47.5 GB card is still
not viable at nproc 2** — the card is genuinely full, not fragmented (free is
~20 MB and ~1.4 MB respectively). Earlier, on `2fcbcbe0`, these same rows *did*
reach rematerialisation at nproc 2, so it is marginal rather than absolutely
impossible — which is the worst kind, because it fails late.

Plan here: relaunch them at **nproc 4** once the bs16 handover frees four GPUs on
bellflower-0. No node currently has four contiguous free GPUs.

This is A40-specific and you will not see it on the A100, but it bears on your
32k/64k token rows if they are ever run on A40 hardware — they are bs256 too.

## Status

Eight rows live (the batch-size axis), 19/220 query-scores done, no fatals,
ssd-2 ~911 GB. Three bs256 rows parked as above; `wd0.0` still parked behind a
full lucia-ord-0.
