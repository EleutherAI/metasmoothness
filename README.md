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

## History

[LDS_RESULTS.md](LDS_RESULTS.md), [BASELINE_LDS.md](BASELINE_LDS.md) and
[SHAMPOO_RESULTS.md](SHAMPOO_RESULTS.md) hold all pre-2026-08-20 measurements, most of which
are excluded from the paper CSVs (old shuffle implementation — see the exclusion table in
EXPERIMENTS_CSV.md). They remain the provenance record; read them only when you need the
history behind a decision.
