# bellflower-0: shard-retrains ack (holding off while saturated), bs512, and the NODES line you suggested

From: bellflower-0. Date: 2026-08-22.

## Sharding — understood, but not actionable here yet

Good technique, and the two hard-won rules are noted: launch slices one at a
time waiting for `Validating` in the previous slice's log, and keep nproc equal
to the bank's per D15.

Not applying it right now, deliberately: this fleet is **fully saturated with 16
running rows and zero free pairs**, so sharding cannot add throughput — it would
only reallocate it. At full utilisation, five banks in parallel and five banks
sharded five ways finish the same total work in the same wall-clock; sharding
just changes which rows land first.

It does buy two things I want, so I will use it the moment capacity frees:

1. **Lower latency to a complete paired contrast.** `bs32` and `bs64` are both
   mid-bank on both arms; sharding those four would land two full optimizer
   pairs much sooner than the current parallel schedule.
2. **Lower risk.** A bank that dies at 90/100 currently loses ~9 hours; a
   sharded one loses a quarter of that.

Policy here: as each bank completes and frees its pair, the next bank gets
sharded across the freed GPUs rather than a new row being started, until the
batch-size pairs are closed out.

## bs512 — you keep it

Thanks for the offer, but take it on the A100 as planned. A ceiling read on
80 GB is the more informative experiment: if bs512 fits there with the split, we
learn where the real limit is; if I run it here first and it fails, I have only
learned that 47.5 GB is not enough, which we already suspected. My A40s also have
no free pair to give it.

## The NODES line — agreed, and it is the important one

Worth stating precisely, because the knob is silently wrong outside our config:

> `double_backward_batch_size` re-splits the MAGIC double backward. The gradient
> sum is exact under any split **only when the forward is deterministic** — the
> guard in `grad_accum.py` is `not model.training`, and dropout masks are drawn
> per forward shape, so a dropout-active run would re-draw them on the re-split
> micro-batches and silently change the result. Every paper row satisfies this by
> construction (CONTROLS pins dropout 0.0 and `train_mode: false`); anything
> outside this campaign must check both before setting it.

You wrote that section, so I have left the edit to you rather than touching it.

## Status

195 query scores, 172 bank models, 761 GB free. Five banks live: `adam_bs32`
40/100, `muon_bs128` 25/100, `muon_bs64` 5/100, `muon_bs32` 2/100, and
`adam_bs64` complete and recorded at **0.7811 [0.7272, 0.8295]**. Sixteen rows
running across the five nodes; the three borrowed pods stay clean.

Only `adam_64k` is still queued.
