# 2026-08-23 — sharded both anchors, gpt2-medium started, volumes rebalanced

## Sharding: both anchors and five grid rows are now parallel

Both anchors were validating serially at roughly 570 seconds per subset with ~80
subsets left, i.e. **13 hours each**. Both are now three-way:

| run | main process | slice A | slice B |
|---|---|---|---|
| `sm_muon_eps1e17_16k_bs256` | 18→ (bellflower-0) | 40–70 (maria-1) | 70–100 (shivam2-0) |
| `sm_adamw_eps1e17_16k_bs256` | 22→ (bellflower-0) | 40–70 (maria-1) | 70–100 (shivam2-0) |

Five more rows got one slice each covering subsets 55–100: `wd0.1`, `wd0.0`,
`clip1.0` (lucia-ord-0) and `scale0.5`, `scale0.25` (bellflower-0). Expect ~5–7
hours instead of ~13 on all seven.

The main processes were left running rather than restarted with a `subset_stop`.
They will redo subsets the slices also cover. That is deliberate: restarting
would have thrown away 3 hours of in-progress validation, and the 8k merge
established that duplicate rows across a shard boundary are identical
(`score_sum` delta exactly 0.0). **Merge with `scripts/magic_lds.py` and check the
merged subset set is 0–99 exactly once before recording any of these.**

The two `scale` slices run against `/mnt/ssd-1/lucia/bergson-logit-scale`
(PR #433), not the pinned `-429` worktree — they carry a `logit_scale` field that
`-429` rejects at parse time. Record `code_commit` for those two rows.

## Disk: the volumes were about to run out, and are rebalanced

`df` showed ssd-2 at 402 GB free with **~667 GB of remaining demand** — all 14
in-flight runs write there, and a finished run is ~69 GB (21 GB checkpoints +
48 GB retrained). They would have collectively hit the wall mid-bank.

Completed banks are being moved to ssd-1 by
`_orchestration/migrate_run.sh`, which copies, verifies file-for-file, and only
then replaces the source with a symlink — so every path that referenced a moved
bank still resolves. `plan_adam_eps1e17_16k_bs32` (83.5 GiB, 634 entries) and
`plan_adam_eps1e17_4k_bs256` are done; two more are queued.

**Do not start a new gpt2-small bank without re-checking `df` on both volumes.**

## gpt2-medium (D11)

Sweep measured on the borrowed A100 pods: 5e-5 → 3.0062, **1e-4 → 3.0019**,
2e-4 → 3.0085. Interior optimum, so no extension needed, but it is very flat —
0.0066 nats across a 4x range, against 0.14 nats in the gpt2-small anchor sweep.
1e-4 is recorded in `TUNED_LR`.

The bank is running on shivam2-0 at nproc 4, writing to
`/mnt/ssd-4/lucia/paper_runs/experiments/` — ~200 GB (100 models at 1.4 GB) does
not fit beside the grid on ssd-2 or ssd-1. **ssd-4 is mounted on the A100 pods
only.** Before those pods go back, the bank must be uploaded to HuggingFace and
`validation.csv` copied to ssd-2, or the result is stranded.

## Two rows that must not be launched as they stand

- **`ckptavg4`** — nothing implements it. bergson has no
  `ckpt_avg_k`/`avg_k`/`averaged_gradient` code and `gen_experiment_run.py` never
  reads the column, so the generated config is byte-equivalent to a plain bs256
  anchor. Launching it spends a full bank reproducing the anchor. Recorded in the
  row notes.
- **`arch_control` / `preact_layernorm`** — still blocked on the same
  architecture design question that put the QK-norm rows in future work (D16).

## Generator and launcher fixes

- `gen_tuning_run.py` emitted `logit_scale` unconditionally, which killed all
  three gpt2-medium sweep points before a single step ran: a no-op *value* does
  not help when the *field* is unknown to `-429`. Same fix as `ca72992`.
- `launch.sh` hard-coded the ssd-2 run root, but new run dirs go to ssd-1 since
  `7118153` — it would have handed bergson a nonexistent `experiment.yaml` with
  the claim already committed. It now resolves both volumes, and honours
  `RUN_ROOT=<dir>` for oversized rows.
