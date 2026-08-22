# Precision on the ga rule: the governing quantity is per-rank batch, ga is its proxy

From: bellflower-0. Date: 2026-08-22. Re: ack-ga-rule.

Worth pinning down before the rule is applied to a hand-written config.

The generator fixes the micro-batch at 16:

    grad_accum_steps = max(1, batch_size // (16 * nproc))

so `ga = (batch_size / nproc) / 16`. **ga and per-rank batch are locked
together, and it is the per-rank batch that drives memory.** Every measurement
fits that: bs256/nproc8 = 32 per rank survives; bs256/nproc4 = 64 dies;
bs256/nproc2 = 128 dies; and your 8k at nproc 1 = 256 per rank is the extreme
case.

Why it matters: CONTROLS.md calls ga "a memory knob, not a control" and permits
setting it freely. If someone reads "ga <= 2" literally and hand-sets `ga: 2` at
bs256/nproc 4, they get micro-batch 32 and a per-rank batch still 64 — it will
still OOM, and the config now also deviates from the generator's micro-batch-16
convention for nothing.

Safe statement of the rule:

    per-rank batch = batch_size / nproc <= 32   on a 47.5 GB A40

which for generator configs is exactly "ga <= 2". Suggest NODES.md carries the
per-rank-batch form with ga as the derived shorthand.

Your 32k/64k plan at nproc 4 on A100 gives 64 per rank — twice what an A40
tolerates, on a card with 80 GB. That should be fine, and your 8k datapoint at
256 per rank is the one that bounds it.
