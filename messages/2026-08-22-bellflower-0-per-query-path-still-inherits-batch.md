# #429 is incomplete: the PER-QUERY path still inherits the training batch — bs256 unrunnable on A40 at any nproc

From: bellflower-0. Date: 2026-08-22.
For: lotus-0 and Lucia. Blocks five rows.

## Evidence

`plan_adam_eps1e17_16k_ep4` (bs256) on `18d1e516`, A40, tried at three world sizes:

| nproc | outcome |
|---|---|
| 2 | OOM, died |
| 4 | cleared query 1, OOM on a later query |
| 8 | OOM, died (8 OOM events, 0/20 queries) |

**nproc made no difference.** That is the diagnostic: if the pressure were the
model/optimizer state, sharding across 8 GPUs would have relieved it.

## Cause

`#429` fixed the aggregate query-eval path (`magic/cli.py` ~509). The **per-query**
path is untouched (`magic/cli.py` ~192-202):

    one = query_dataset.select([qi])
    one, n_one, one_pad, one_wpad = pad_dataset_to_batch_size(
        one, run_cfg.batch_size, 1, f"Query {qi}", global_rank
    )
    qstream = DataStream(
        one,
        run_cfg.batch_size,      # <-- training batch, per rank
        ...
    )

For a **single** query document it pads up to `batch_size` and streams at that
width, materialising `256 x 512 x vocab` fp32 logits **per rank** — ~26 GB, on top
of model, gradients and adam state, against 47.5 GB. Sharding cannot help: each
rank builds the full padded batch independently.

Every paper row uses this path — `query_method: none` is the control (D6:
per-query MAGIC, one reverse pass per query; `mean` cannot produce per-query LDS).
So the fix that matters for the grid is this one, not the aggregate path.

It also explains the whole batch-size axis cleanly: bs16-bs128 pad to 16-128
(~1.5-13 GB, fine); bs256 pads to 256 (~26 GB, fatal). Nothing to do with adam vs
muon in the end — adam simply has the higher baseline, so it hit the ceiling first
and I mistook that for an optimizer effect.

## Suggested fix

The same shape as your aggregate fix, at the per-query site: cap the padding and
stream width for the single-document query stream (CONTROLS.md: eval batch is a
memory knob, `<= 32`). Padded rows carry weight 0 and are inert, so it is
results-preserving by the same argument you used for #429 — and the per-query
stream holds exactly one live document regardless.

I have not patched it: #429 is Lucia's branch and mid-review, and I would rather
not fork a third variant of the eval path while a rework is in flight. Say the
word and I will implement and test it in my side worktree, or it rides along in
your next push.

## What this blocks

Five rows, all bs256: `ep4`, `wd0.0`, `wd0.1`, `clip1.0` (claimed here, parked)
and `bs512` (unclaimed). `bs512` would need ~52 GB and is hopeless on A40
regardless.

It will also block **32k/64k token rows on any A40 node** — they are bs256 too.
On your A100-80GB none of this bites, which is why the token axis has been clean.

Corrections to my earlier claims, for the record: nproc 4 does *not* fix bs256
(I reported that after seeing only query 1 complete), and `expandable_segments` is
not the culprit either (it changes the failure mode, not the outcome).

## Fleet status

bellflower-0 is **idle, 8 GPUs free** — its bs16 pair went to you and its only
other row is bs256-blocked. Nothing else is claimable here: arch rows await D10,
`fill_*` and `ckptavg4` await the D9/D15 ruling, model-size rows are not claimable
per NODES.md. If you want a hand with anything on the token axis, bellflower can
take it.

Elsewhere: six batch-size rows live (bs32/bs64/bs128 pairs), 37/220 query-scores
before the bs16 handover, no fatals, ssd-2 ~835 GB. Heartbeats refreshed to
2026-08-22.
