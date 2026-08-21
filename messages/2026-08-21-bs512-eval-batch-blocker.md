# plan_adam_eps1e17_16k_bs512 is not runnable as generated (eval batch = train batch)

From: bellflower-0. Date: 2026-08-21.
Status: **NOT claimed, NOT started.** Needs a ruling before anyone picks it up.

## The contradiction

CONTROLS.md, "Attribution / estimator", is explicit:

> eval/validate `batch_size` is set independently of the training batch; keep it
> at 32 or less. Per-document query-loss evaluation materialises fp32 logits of
> batch x 512 x vocab: ~26 GB at batch 256 (measured OOM on 47.5 GB A40s),
> ~52 GB at batch 512. Training batch is the science; eval batch is only a
> memory knob.

But the code has no separate eval-batch knob. `bergson/magic/cli.py` builds the
query stream from the training batch directly:

    query_stream = DataStream(
        query_dataset,
        run_cfg.batch_size,      # <- training batch, not an eval batch
        ...
    )

and the same at the per-query path (line ~188). `gen_experiment_run.py` writes a
single `batch_size` into the config, so for this row it is 512 in both roles.

`query_20.hf` holds 20 documents, but `pad_dataset_to_batch_size` pads up to a
multiple of the batch, so a batch of 512 materialises the full 512 x 512 x 50257
fp32 logit tensor: **~52 GB, against 47.5 GB of A40**. It cannot fit, and it
would fail late — after base training and 100 retrains — not at startup.

Per NODES.md ("If a row's config seems to contradict CONTROLS.md, trust CONTROLS
and ask"), the row is left unclaimed with this note rather than started.

## Scope

- **bs512 (this row): fatal.** ~52 GB > 47.5 GB.
- **bs256 rows: tight but historically fine.** ~26 GB of 47.5 GB. That includes
  the anchor, the wd/clip/ep4 knob rows, and the 32k/64k token rows. They run,
  but they are one config change away from the same cliff — and on a 40 GB A40
  they would not fit at all.
- Everything at bs128 or below is comfortable.

## Options for Lucia

1. **Add a real eval-batch knob to bergson** (`ValidationConfig.eval_batch_size`,
   defaulting to `min(batch_size, 32)`), then generate this row normally. This
   matches what CONTROLS.md already claims exists, and makes the bs256 rows
   comfortable rather than marginal. It is eval-side only — it does not touch
   training numerics, so it does not disturb rows already measured. This is the
   option I would take.
2. **Cap the query stream in the generator only.** Smaller change, but leaves the
   library contradicting CONTROLS.md for anyone not using this generator.
3. **Drop bs512 from the batch-size axis.** The axis still spans 16-256; the
   step-count deconfound loses its "double batch at fixed epochs" arm, which was
   half of the pair that separates steps from batch size (CONTROLS, per-axis table).

Option 1 is a small, eval-only change, but it is a bergson code change during a
measurement campaign, so it is your call, not mine.

## What I ran instead

The freed slot on secret-ord-0 (GPUs 4,5, port 29612) went to
`plan_adam_eps1e17_16k_wd0.1`, which completes the weight-decay pair with
`wd0.0` — a lone wd0.0 has nothing to contrast against. Disk budget is unchanged
at 12 banks.
