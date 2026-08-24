# Metasmoothness and LDS: Empirical Analysis

Understanding how training-algorithm design choices affect training function **smoothness** w.r.t data-weights and in turn how smoothness affects the **Linear Datamodeling
Score (LDS)** of attribution methods (MAGIC, EK-FAC). Target setting: LLM post-training
(GPT-2 fine-tuned on SmolLM2 512-token chunks), with token-scaling and batch size-scaling axes.

## Start here

| file | what it holds |
|---|---|
| [CONTROLS.md](CONTROLS.md) | The fixed control hyperparameters every ablation deviates from by one factor, each value tied to its evidence; the tuning protocol |
| [DECISIONS.md](DECISIONS.md) | Design decisions D1-D13 with rulings, plus the learning-rate sweep-grid design |
| [EXPERIMENTS_CSV.md](EXPERIMENTS_CSV.md) | Schema and admission policy for the data CSVs; what old data was excluded and why; planned pre-training experiments |
| [NODES.md](NODES.md) | How multiple nodes claim rows, check in, and steal stale claims |

## Build data templates

```
python build_tuning_csv.py        # tuning.csv      — stage 0: lr selection runs (run these first)
python build_experiments_csv.py   # experiments.csv — stage 1: the metasmoothness/LDS grid
```

One row per run. Empty result cells are work to claim (see NODES.md). Edit results in the
builder scripts and regenerate — never in the CSVs, except the two node-claim columns.
Datasets: `EleutherAI/bergson-smollm2-scaling` on the Hub (verified nested train chain +
disjoint held-out/query sets); tooling in `scripts/`.

## Results

```
run                               | optimizer | N docs |  bs | N epochs | N steps | metasmoothness | EK-FAC LDS | MAGIC LDS | status 
----------------------------------+-----------+--------+-----+----------+---------+----------------+------------+-----------+--------
plan_adam_eps1e17_4k_bs256        | adamw     |  4,000 | 256 |        2 |      32 |         0.9946 |     0.3975 |    0.9295 | done   
plan_adam_eps1e17_8k_bs256        | adamw     |  8,000 | 256 |        2 |      63 |         0.9924 |     0.3869 |    0.9163 | done   
plan_adam_eps1e17_16k_bs16        | adamw     | 16,000 |  16 |        2 |    2000 |         0.9133 |     0.3872 |    0.1796 | done   
plan_adam_eps1e17_16k_bs32        | adamw     | 16,000 |  32 |        2 |    1000 |         0.9800 |     0.4586 |    0.9201 | done   
plan_adam_eps1e17_16k_bs64        | adamw     | 16,000 |  64 |        2 |     500 |         0.9853 |     0.4239 |    0.7811 | done   
plan_adam_eps1e17_16k_bs128       | adamw     | 16,000 | 128 |        2 |     250 |         0.9935 |     0.4551 |    0.9441 | done   
plan_adam_eps1e17_16k_clip1.0     | adamw     | 16,000 | 256 |        2 |     125 |         0.9896 |     0.4176 |    0.8982 | done   
plan_adam_eps1e17_16k_scale0.25   | adamw     | 16,000 | 256 |        2 |     125 |         0.9150 |     0.1733 |    0.0456 | done   
plan_adam_eps1e17_16k_scale0.5    | adamw     | 16,000 | 256 |        2 |     125 |         0.9878 |     0.1760 |    0.9448 | done   
plan_adam_eps1e17_16k_wd0.0       | adamw     | 16,000 | 256 |        2 |     125 |         0.9930 |     0.4235 |    0.9410 | done   
plan_adam_eps1e17_16k_wd0.1       | adamw     | 16,000 | 256 |        2 |     125 |         0.9930 |     0.4244 |    0.9414 | done   
sm_adamw_eps1e17_16k_bs256        | adamw     | 16,000 | 256 |        2 |     125 |         0.9930 |     0.4253 |    0.9411 | done   
plan_adam_eps1e17_16k_gpt2-medium | adamw     | 16,000 | 256 |        2 |     125 |         0.8580 |          - |         - | planned
plan_adam_eps1e17_16k_bs512       | adamw     | 16,000 | 512 |        2 |      63 |         0.9950 |     0.4142 |    0.9233 | done   
plan_muon_eps1e17_4k_bs256        | muon      |  4,000 | 256 |        2 |      32 |         0.9037 |     0.3031 |    0.3020 | done   
plan_muon_eps1e17_8k_bs256        | muon      |  8,000 | 256 |        2 |      63 |         0.9962 |     0.3881 |    0.7712 | done   
plan_muon_eps1e17_16k_bs16        | muon      | 16,000 |  16 |        2 |    2000 |         0.9939 |          - |         - | planned
plan_muon_eps1e17_16k_bs32        | muon      | 16,000 |  32 |        2 |    1000 |         0.9952 |     0.4567 |    0.8737 | done   
plan_muon_eps1e17_16k_bs64        | muon      | 16,000 |  64 |        2 |     500 |         0.9939 |     0.4284 |    0.8690 | done   
plan_muon_eps1e17_16k_bs128       | muon      | 16,000 | 128 |        2 |     250 |         0.9944 |     0.4635 |    0.8480 | done   
sm_muon_eps1e17_16k_bs256         | muon      | 16,000 | 256 |        2 |     125 |         0.9964 |     0.4237 |    0.8379 | done   

21 rows shown, 19 with all three metrics. '-' = not yet measured.
```

## What causes a collapse: learning rate or the named knob?

Every row is tuned to its own learning rate, so a row that collapses differs from
the anchor in **two** ways and the collapse cannot be attributed to the named knob
without holding lr fixed. An ms probe costs three trainings and no retrain bank,
so the control is cheap. Raw data: [data/ms_diagnostics.csv](data/ms_diagnostics.csv).

| probe | bs | lr | logit scale | ms | isolates |
|---|---:|---:|---:|---:|---|
| anchor | 256 | 2e-4 | 1.00 | 0.9930 | reference |
| scale0.5 row | 256 | 2e-4 | 0.50 | 0.9878 | logit scale at anchor lr |
| scale0.25 row | 256 | 8e-4 | 0.25 | 0.9150 | scale **and** 4x lr |
| scale0.25 probe | 256 | 2e-4 | 0.25 | **0.9812** | logit scale alone |
| bs16 row | 16 | 5e-5 | 1.00 | 0.9133 | batch **and** 1/4 lr |
| bs16 probe | 16 | 2e-4 | 1.00 | **0.5127** | batch size alone |
| anchor probe | 256 | 5e-5 | 1.00 | **0.9948** | low lr alone |

The two collapses have opposite causes. **scale0.25 is the learning rate**: at the
anchor lr, logit scale 0.25 is healthy (0.9812), and scale0.5 — which already runs
at the anchor lr — has MAGIC 0.9448, unharmed. **bs16 is genuinely the batch size**,
and its tuned 5e-5 is *protecting* it: at the anchor lr it scores 0.5127, the lowest
ms measured anywhere, while low lr alone is harmless (0.9948).

Consequence for the batch axis: bs16's MAGIC of 0.1796 is the *rescued* number.
"Attribution degrades at small batch" means "degrades at small batch, after lr
tuning has already compensated".

## Does metasmoothness predict attributability?

Sorted by ms, the 17 rows with both metrics separate cleanly:

    ms <  0.95  (n=3)    MAGIC  0.046 - 0.302
    ms >= 0.95  (n=14)   MAGIC  0.771 - 0.945

No overlap: the gap between groups (0.302 to 0.771) is wider than the spread within
either. The three low-ms rows are exactly the three lowest-MAGIC rows, and one of
them (bs16, ms 0.9133) was predicted before its ms was measured.

**Do not quote `spearman(ms, MAGIC) = +0.060` as a null result.** Fourteen of
seventeen rows sit inside a 0.017-wide ms band, so fourteen of seventeen ranks are
noise and the correlation is uninformative in both directions. What the data
supports is the narrow claim — ms **detects pathological configurations** — not the
broad one, that it ranks attributability. Inside the healthy band the orderings are
unrelated: bs64 has higher ms than bs32 (0.9853 vs 0.9800) and lower MAGIC (0.7811
vs 0.9201), and muon 8k has the second-highest ms of all with MAGIC of 0.7712.

## How many queries does an LDS estimate need?

MAGIC costs one reverse pass **per query**, so this is the largest cost lever in the
grid. Bootstrap half-width of the per-query Spearman mean, by query count:

| queries | rows inside the D6 threshold (0.06) |
|---:|---|
| 5 | 16 / 21 |
| 10 | 16 / 21 |
| 15 | 18 / 21 |
| 20 | 19 / 21 |

Most rows are already well inside the threshold at 5 queries — the anchor is 0.0163
at n=5 against 0.0083 at n=20. The rows that genuinely need 20 are the low-LDS ones,
where per-query variance is largest: bs16 (0.1014 even at n=20), muon 4k (0.1244),
bs64 (0.0541). Regenerate with `scripts/ci_vs_queries.py`.

## History

[LDS_RESULTS.md](LDS_RESULTS.md), [BASELINE_LDS.md](BASELINE_LDS.md) and
[SHAMPOO_RESULTS.md](SHAMPOO_RESULTS.md) hold all pre-2026-08-20 measurements, most of which
are excluded from the paper CSVs (old shuffle implementation — see the exclusion table in
EXPERIMENTS_CSV.md). They remain the provenance record; read them only when you need the
history behind a decision.

## Differences in final query loss across GPU types

Loss query differences in nats when re-training on the same subsets across GPUs, vs. between
different subsets on the same GPUs.

| varied | held fixed | comparisons | median | p90 | max |
|---|---|---:|---:|---:|---:|
| GPU type (A40 → A100) | run, subset, query | 240 | 6.7e-04 | 2.5e-03 | 3.3e-03 |
| which subset was dropped | run, query, GPU | 720 | 6.8e-04 | 2.4e-03 | 8.1e-03 |

Splitting each difference into the offset shared by all subsets of a query and a residual
that differs between them. Only the residual part affects subset query loss rank/LDS.

| difference | part | comparisons | median | p90 | max |
|---|---|---:|---:|---:|---:|
| GPU type (A40 → A100) | shared offset | 60 | 7.0e-04 | 2.4e-03 | 3.2e-03 |
| GPU type (A40 → A100) | residual | 240 | 5.7e-05 | 1.3e-04 | 2.6e-04 |
| which subset was dropped | shared offset | 120 | 5.9e-04 | 2.0e-03 | 3.3e-03 |
| which subset was dropped | residual | 480 | 4.7e-04 | 1.5e-03 | 5.8e-03 |

This shows that LDS re-trains should not be sharded across different GPU types.
