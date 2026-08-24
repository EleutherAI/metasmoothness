# Metasmoothness and LDS: Empirical Analysis

Understanding how training-algorithm design choices affect data-weight **metasmoothness**
(Chang et al. 2024, Def. 2), and in turn how metasmoothness affects the **Linear Datamodeling
Score (LDS)** of attribution methods (MAGIC, EK-FAC). Target setting: LLM post-training
(GPT-2 fine-tuned on SmolLM2 512-token chunks), with token-scaling as a first-class axis.

## Start here (in order)

| file | what it holds |
|---|---|
| [CONTROLS.md](CONTROLS.md) | The fixed control hyperparameters every ablation deviates from by one factor, each value tied to its evidence; the tuning protocol |
| [DECISIONS.md](DECISIONS.md) | Design decisions D1-D13 with rulings, plus the learning-rate sweep-grid design |
| [EXPERIMENTS_CSV.md](EXPERIMENTS_CSV.md) | Schema and admission policy for the data CSVs; what old data was excluded and why; planned pre-training experiments |
| [NODES.md](NODES.md) | How multiple nodes claim rows, check in, and steal stale claims |

## The data (source of truth)

```
python build_tuning_csv.py        # tuning.csv      — stage 0: lr selection runs (run these first)
python build_experiments_csv.py   # experiments.csv — stage 1: the metasmoothness/LDS grid
```

One row per run; empty result cells are work to claim (see NODES.md). Edit results in the
builder scripts and regenerate — never in the CSVs, except the two node-claim columns.
Datasets: `EleutherAI/bergson-smollm2-scaling` on the Hub (verified nested train chain +
disjoint held-out/query sets); tooling in `scripts/`.

## Results

Regenerate with `python scripts/results_table.py --all` (drop `--all` for measured rows only,
add `--ci` for 95% CIs on the LDS columns) any time `experiments.csv` changes.

```
run                               | optimizer | N docs |  bs | N epochs | N steps | metasmoothness | EK-FAC LDS | MAGIC LDS | status 
----------------------------------+-----------+--------+-----+----------+---------+----------------+------------+-----------+--------
plan_adam_eps1e17_4k_bs256        | adamw     |  4,000 | 256 |        2 |      32 |         0.9946 |     0.3975 |    0.9295 | done   
plan_adam_eps1e17_8k_bs256        | adamw     |  8,000 | 256 |        2 |      63 |         0.9924 |     0.3869 |    0.9163 | done   
plan_adam_eps1e17_16k_bs16        | adamw     | 16,000 |  16 |        2 |    2000 |              - |     0.3872 |    0.1796 | done   
plan_adam_eps1e17_16k_bs32        | adamw     | 16,000 |  32 |        2 |    1000 |         0.9800 |     0.4586 |    0.9201 | done   
plan_adam_eps1e17_16k_bs64        | adamw     | 16,000 |  64 |        2 |     500 |         0.9853 |     0.4239 |    0.7811 | done   
plan_adam_eps1e17_16k_bs128       | adamw     | 16,000 | 128 |        2 |     250 |         0.9935 |     0.4551 |    0.9441 | done   
plan_adam_eps1e17_16k_clip1.0     | adamw     | 16,000 | 256 |        2 |     125 |         0.9896 |     0.4176 |    0.8982 | done   
plan_adam_eps1e17_16k_scale0.25   | adamw     | 16,000 | 256 |        2 |     125 |         0.9150 |     0.1733 |    0.0456 | done   
plan_adam_eps1e17_16k_scale0.5    | adamw     | 16,000 | 256 |        2 |     125 |         0.9878 |     0.1760 |    0.9448 | done   
plan_adam_eps1e17_16k_wd0.0       | adamw     | 16,000 | 256 |        2 |     125 |         0.9930 |     0.4235 |    0.9410 | done   
plan_adam_eps1e17_16k_wd0.1       | adamw     | 16,000 | 256 |        2 |     125 |         0.9930 |     0.4244 |    0.9414 | done   
sm_adamw_eps1e17_16k_bs256        | adamw     | 16,000 | 256 |        2 |     125 |         0.9930 |     0.4253 |    0.9411 | done   
plan_adam_eps1e17_16k_ckptavg4    | adamw     | 16,000 | 256 |        2 |     125 |              - |          - |         - | planned
plan_adam_eps1e17_16k_gpt2-large  | adamw     | 16,000 | 256 |        2 |     125 |              - |          - |         - | planned
plan_adam_eps1e17_16k_gpt2-medium | adamw     | 16,000 | 256 |        2 |     125 |              - |          - |         - | planned
plan_adam_eps1e17_16k_ep4         | adamw     | 16,000 | 256 |        4 |     250 |              - |          - |         - | planned
plan_adam_eps1e17_16k_bs512       | adamw     | 16,000 | 512 |        2 |      63 |              - |     0.4142 |    0.9233 | done   
plan_adam_eps1e17_32k_bs256       | adamw     | 32,000 | 256 |        2 |     250 |              - |          - |         - | planned
plan_adam_eps1e17_64k_bs256       | adamw     | 64,000 | 256 |        2 |     500 |              - |          - |         - | planned
plan_muon_eps1e17_4k_bs256        | muon      |  4,000 | 256 |        2 |      32 |         0.9037 |     0.3031 |    0.3020 | done   
plan_muon_eps1e17_8k_bs256        | muon      |  8,000 | 256 |        2 |      63 |         0.9962 |     0.3881 |    0.7712 | done   
plan_muon_eps1e17_16k_bs16        | muon      | 16,000 |  16 |        2 |    2000 |              - |          - |         - | planned
plan_muon_eps1e17_16k_bs32        | muon      | 16,000 |  32 |        2 |    1000 |              - |     0.4567 |    0.8737 | done   
plan_muon_eps1e17_16k_bs64        | muon      | 16,000 |  64 |        2 |     500 |         0.9939 |     0.4284 |    0.8690 | done   
plan_muon_eps1e17_16k_bs128       | muon      | 16,000 | 128 |        2 |     250 |         0.9944 |     0.4635 |    0.8480 | done   
sm_muon_eps1e17_16k_bs256         | muon      | 16,000 | 256 |        2 |     125 |         0.9964 |     0.4237 |    0.8379 | done   
plan_muon_eps1e17_32k_bs256       | muon      | 32,000 | 256 |        2 |     250 |              - |          - |         - | planned
plan_muon_eps1e17_64k_bs256       | muon      | 64,000 | 256 |        2 |     500 |              - |          - |         - | planned

28 rows shown, 16 with all three metrics. '-' = not yet measured.
```

## History

[LDS_RESULTS.md](LDS_RESULTS.md), [BASELINE_LDS.md](BASELINE_LDS.md) and
[SHAMPOO_RESULTS.md](SHAMPOO_RESULTS.md) hold all pre-2026-08-20 measurements, most of which
are excluded from the paper CSVs (old shuffle implementation — see the exclusion table in
EXPERIMENTS_CSV.md). They remain the provenance record; read them only when you need the
history behind a decision.
