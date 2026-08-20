# WikiText-2 / GPT-2 SOURCE replication (Bae et al. 2024)

Configs replicating Figure 6 (SOURCE > EK-FAC IF at α = 0.5) end to end from
public data, run with bergson 0.20+. Chain: `prep_dataset.py` (hosted as
`EleutherAI/bergson-wikitext-2-4656-chunks`) → train → source → ekfac →
retrain (100 random halves × 5 seeds, same subsets) → validate (query losses
averaged over seeds before correlating).

LDS, mean Spearman over 481 validation queries ± 95% CI:

| Ground truth | EK-FAC IF | SOURCE |
|---|---|---|
| single seed | 0.418 ± 0.016 | 0.429 ± 0.015 |
| 5-seed averaged | 0.468 ± 0.015 | 0.476 ± 0.015 |

SOURCE > IF on 308/481 queries, Wilcoxon p = 2.3e-11 (5-seed). kronfluence
reports 0.44 for EK-FAC IF on its own 5-seed ground truth. Per-seed EK-FAC
spread across single rulers: 0.4142–0.4183.
