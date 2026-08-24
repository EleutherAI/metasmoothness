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
run                               | optimizer |    lr | N docs |  bs | N epochs | N steps | metasmoothness | EK-FAC LDS | MAGIC LDS | rand filt Δ | EK-FAC filt Δ | MAGIC filt Δ | train loss | heldout loss | delta L2 | status 
----------------------------------+-----------+-------+--------+-----+----------+---------+----------------+------------+-----------+-------------+---------------+--------------+------------+--------------+----------+--------
plan_adam_eps1e17_4k_bs256        | adamw     | 1e-04 |  4,000 | 256 |        2 |      32 |         0.9946 |     0.3975 |    0.9295 |     0.00004 |       0.01153 |      0.02505 |     3.2064 |       3.3149 |     8.04 | done   
plan_adam_eps1e17_8k_bs256        | adamw     | 2e-04 |  8,000 | 256 |        2 |      63 |         0.9924 |     0.3869 |    0.9163 |     0.00018 |       0.02923 |      0.07060 |     3.1309 |       3.2851 |    20.58 | done   
plan_adam_eps1e17_16k_bs16        | adamw     | 5e-05 | 16,000 |  16 |        2 |    2000 |         0.9133 |     0.3872 |    0.1796 |           - |             - |            - |     3.0698 |       3.2497 |    39.77 | done   
plan_adam_eps1e17_16k_bs32        | adamw     | 5e-05 | 16,000 |  32 |        2 |    1000 |         0.9800 |     0.4586 |    0.9201 |           - |             - |            - |     3.1031 |       3.2473 |    22.45 | done   
plan_adam_eps1e17_16k_bs64        | adamw     | 1e-04 | 16,000 |  64 |        2 |     500 |         0.9853 |     0.4239 |    0.7811 |           - |             - |            - |     3.0715 |       3.2479 |    27.64 | done   
plan_adam_eps1e17_16k_bs128       | adamw     | 1e-04 | 16,000 | 128 |        2 |     250 |         0.9935 |     0.4551 |    0.9441 |           - |             - |            - |     3.1162 |       3.2498 |    19.76 | done   
plan_adam_eps1e17_16k_clip1.0     | adamw     | 2e-04 | 16,000 | 256 |        2 |     125 |         0.9896 |     0.4176 |    0.8982 |           - |             - |            - |     3.0919 |       3.2543 |    27.66 | done   
plan_adam_eps1e17_16k_scale0.25   | adamw     | 8e-04 | 16,000 | 256 |        2 |     125 |         0.9150 |     0.1733 |    0.0456 |           - |             - |            - |     3.2238 |       3.4343 |   123.70 | done   
plan_adam_eps1e17_16k_scale0.5    | adamw     | 2e-04 | 16,000 | 256 |        2 |     125 |         0.9878 |     0.1760 |    0.9448 |           - |             - |            - |     3.1896 |       3.3022 |    30.75 | done   
plan_adam_eps1e17_16k_wd0.0       | adamw     | 2e-04 | 16,000 | 256 |        2 |     125 |         0.9930 |     0.4235 |    0.9410 |           - |             - |            - |     3.1078 |       3.2572 |    27.22 | done   
plan_adam_eps1e17_16k_wd0.1       | adamw     | 2e-04 | 16,000 | 256 |        2 |     125 |         0.9930 |     0.4244 |    0.9414 |           - |             - |            - |     3.1077 |       3.2572 |    27.54 | done   
sm_adamw_eps1e17_16k_bs256        | adamw     | 2e-04 | 16,000 | 256 |        2 |     125 |         0.9930 |     0.4253 |    0.9411 |     0.00023 |       0.05288 |      0.09090 |     3.1078 |       3.2572 |    27.23 | done   
plan_adam_eps1e17_16k_gpt2-medium | adamw     | 1e-04 | 16,000 | 256 |        2 |     125 |         0.8580 |          - |         - |           - |             - |            - |          - |            - |        - | planned
plan_adam_eps1e17_16k_ep4         | adamw     | 1e-04 | 16,000 | 256 |        4 |     250 |         0.9959 |          - |         - |           - |             - |            - |          - |       3.2503 |        - | planned
plan_adam_eps1e17_16k_bs512       | adamw     | 2e-04 | 16,000 | 512 |        2 |      63 |         0.9950 |     0.4142 |    0.9233 |           - |             - |            - |     3.1700 |       3.2751 |    20.83 | done   
plan_adam_eps1e17_32k_bs32        | adamw     | 5e-05 | 32,000 |  32 |        2 |    2000 |         0.9866 |          - |         - |           - |             - |            - |          - |            - |        - | planned
plan_adam_eps1e17_32k_bs256       | adamw     | 2e-04 | 32,000 | 256 |        2 |     250 |         0.9937 |          - |         - |           - |             - |            - |          - |       3.2365 |        - | planned
plan_adam_eps1e17_64k_bs256       | adamw     | 1e-04 | 64,000 | 256 |        2 |     500 |         0.9876 |          - |         - |           - |             - |            - |          - |       3.2314 |        - | planned
plan_muon_eps1e17_4k_bs256        | muon      | 4e-04 |  4,000 | 256 |        2 |      32 |         0.9037 |     0.3031 |    0.3020 |           - |             - |            - |     3.1281 |       3.3114 |    13.17 | done   
plan_muon_eps1e17_8k_bs256        | muon      | 2e-04 |  8,000 | 256 |        2 |      63 |         0.9962 |     0.3881 |    0.7712 |     0.00012 |       0.02258 |      0.03289 |     3.1560 |       3.2841 |    11.95 | done   
plan_muon_eps1e17_16k_bs16        | muon      | 5e-05 | 16,000 |  16 |        2 |    2000 |         0.9939 |          - |    0.8850 |           - |             - |            - |     3.0518 |       3.2440 |    26.13 | done   
plan_muon_eps1e17_16k_bs32        | muon      | 5e-05 | 16,000 |  32 |        2 |    1000 |         0.9952 |     0.4567 |    0.8737 |           - |             - |            - |     3.0978 |       3.2441 |    18.42 | done   
plan_muon_eps1e17_16k_bs64        | muon      | 1e-04 | 16,000 |  64 |        2 |     500 |         0.9939 |     0.4284 |    0.8690 |           - |             - |            - |     3.0648 |       3.2464 |    24.46 | done   
plan_muon_eps1e17_16k_bs128       | muon      | 1e-04 | 16,000 | 128 |        2 |     250 |         0.9944 |     0.4635 |    0.8480 |           - |             - |            - |     3.1204 |       3.2501 |    16.43 | done   
sm_muon_eps1e17_16k_bs256         | muon      | 2e-04 | 16,000 | 256 |        2 |     125 |         0.9964 |     0.4237 |    0.8379 |           - |             - |            - |     3.1135 |       3.2570 |    20.03 | done   
plan_muon_eps1e17_32k_bs32        | muon      | 5e-05 | 32,000 |  32 |        2 |    2000 |         0.9941 |          - |         - |           - |             - |            - |          - |            - |        - | planned
plan_muon_eps1e17_32k_bs256       | muon      | 2e-04 | 32,000 | 256 |        2 |     250 |         0.9948 |          - |         - |           - |             - |            - |          - |       3.2372 |        - | planned
plan_muon_eps1e17_64k_bs256       | muon      | 1e-04 | 64,000 | 256 |        2 |     500 |         0.9947 |          - |         - |           - |             - |            - |          - |       3.2323 |        - | planned

28 rows shown, 19 with all three metrics. '-' = not yet measured.
```

## How many queries does an LDS estimate need?

MAGIC costs one reverse pass per query.  Bootstrapped confidence intervals of mean Spearman rank correlation over queries:

| queries | fraction of rows with confidence intervals <= +/-0.06 |
|---:|---|
| 5 | 16 / 21 |
| 10 | 16 / 21 |
| 15 | 18 / 21 |
| 20 | 19 / 21 |

The baseline model is 0.0163 at n=5 and 0.0083 at n=20. Low-LDS rows have higher query variance and need 
more queries to get low CIs, e.g. a batch size 16 row has +/-0.1014 at n=20 and muon 4k 
has +/-0.1244. Regenerate data with `scripts/ci_vs_queries.py`.

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
