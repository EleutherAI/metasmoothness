# OOD mean-query scaling runs

The OOD scaling artifacts live under `/mnt/ssd-2/lucia/ood_mean_scaling`.

## Fixed queries

- WikiText-103: `/mnt/ssd-2/lucia/wikitext103_ood/query_50_mean.hf`
- BioForget: `/mnt/ssd-2/lucia/ood_mean_scaling/bioforget/query_50_mean.hf`

Each query is a fixed set of 50 documents with 512 tokens per document. Both
sets have zero exact overlap with the 512k SmolLM training dataset. Use
`query_aggregation: mean` for EK-FAC scoring and `query_method: mean` for
filter validation.

## Generators

- `scripts/prepare_ood_mean_scaling.py`: score configs
- `scripts/prepare_ood_random_banks.py`: reusable three-random control banks
- `scripts/prepare_ood_filters.py`: top-1% filter configs
- `scripts/reshard_bergson_hessian.py`: exact Hessian repartitioning

The random leave-out models are query-independent. Build one bank of three
random 1% subsets per dataset size and reuse it for every query dataset. To
parallelize a bank, generate disjoint `subset_start:subset_stop` configs that
share the same run path and `subsets.json`. Do not run a serial `0:3` job at the
same time as range jobs after `subset_0` has been saved.

## Eight-GPU scoring

An eight-worker EK-FAC scorer requires eight files in every sharded Hessian
directory. A two-shard Hessian cannot be used by merely changing
`nproc_per_node` to 8. Repartition it with `reshard_bergson_hessian.py`, then
point the score run's `hessian` symlink at the verified derived copy.

`resume: true` checks whether intermediate files exist, not whether their
contents are numerically valid. An interrupted inverse application can leave a
complete-sized, all-zero `kfac_query/gradients.bin`; resuming from it produces
fully written all-zero scores. Before accepting a score run, verify:

1. `kfac_query/gradients.bin` is finite and has nonzero standard deviation.
2. Every `written_0` value in `scores/scores.bin` is true.
3. `score_0` is finite, nonzero, and has nonzero standard deviation.

If the inverse-query intermediate is invalid, use a fresh run directory, link
the healthy raw `query` directory, link the verified Hessian, and recompute
`kfac_query` plus scores. Keep the old job until the replacement has produced a
healthy nonzero inverse-query file, then terminate only the superseded process
group.

## Outputs

Each filter result is summarized by `filter_top1pct/filter_summary.csv`. The
diagnostic plotter `scripts/plot_ood_mean_scaling.py` discovers completed points
from 4k through 512k and includes the three-random mean and standard deviation.
