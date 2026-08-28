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

## Research questions

The questions this project exists to answer (Lucia, 2026-08-25). Work is
prioritised against these.

- How various training setup changes break or aid metasmoothness
- LDS correlation with proponent filtering (MAGIC and EK-FAC)
- Whether metasmoothness stays high as dataset size increases (not sure what
  batch size is convincing for this)
- Whether proponent filtering effect stays high as dataset size increases
- Whether Muon and Adam are different in terms of maintaining metasmoothness
  (all these runs are super high ms which doesn't match my wikitext tests; I
  wonder whether our fine-tuning dataset is too similar to the pre-training
  corpus; running some experiments with a corpus of pre-1931 text)

The strongest findings -- meaning the ones cheapest to investigate -- should
generalise up to dataset size **N = 1M** for now.

## Proponent filter scaling (EK-FAC)

```
EK-FAC proponent-filter delta vs corpus size (bs256, 2 epochs)
A = adamw   M = muon   | = 95% CI

 0.245 |
 0.229 |                                                |
 0.214 |                                                A
 0.199 |                                                |
 0.184 |
 0.168 |
 0.153 |
 0.138 |                                       | |
 0.122 |                                       A M
 0.107 |                                       | |
 0.092 |
 0.076 |                     | |
 0.061 |                     | |      | M
 0.046 |            | |      A M      A |
 0.031 |            A |      | |
 0.015 |   A M      | M
 0.000 |     |
       +------------------------------------------------------
            4k       8k      16k      32k      64k      128k  

  N     steps   adamw delta            muon delta             ms(A)   ms(M)
  4k    31      0.01153 [0.00888,0.01508] 0.00943 [0.00662,0.01409] 0.9946    -   
  8k    62      0.02923 [0.01914,0.04631] 0.02258 [0.01304,0.03834] 0.9924  0.9962
  16k   125     0.05288 [0.03626,0.07966] 0.05040 [0.03448,0.07677] 0.9930  0.9964
  32k   250     0.05353 [0.04760,0.05971] 0.05365 [0.04758,0.06008] 0.9937  0.9948
  64k   500     0.12307 [0.11109,0.13477] 0.12397 [0.11170,0.13617] 0.9876  0.9947
  128k  1000    0.21287 [0.19167,0.23260]      retraining          -       -   
```

## Fixed-count filter curve (top 40 documents)

Same rows as the scaling curve above, but removing a fixed 40 documents at
every N instead of 1%. Regenerate with `python scripts/top40_curve.py --readme`.

```
EK-FAC proponent filter, FIXED 40 documents removed (adamw, bs256, 2 epochs)

     N      frac  n_removed           delta [95% CI]    rank1
    4k  0.010000         40 0.01137 [0.00871,0.01492]     20/20
    8k  0.005000         40 0.02078 [0.01161,0.03677]     20/20
   16k  0.002500         40 0.02637 [0.01807,0.03818]     20/20
   32k  0.001250         40 0.01861 [0.01463,0.02327]     20/20
   64k  0.000625         40 0.08129 [0.07166,0.09084]     20/20

F = fixed 40 docs    P = proportional (1% of N)

 0.238 |
 0.221 |
 0.204 |
 0.187 |
 0.170 |
 0.153 |
 0.136 |
 0.119 |                                         P
 0.102 |
 0.085 |                                       F
 0.068 |
 0.051 |                       P        P
 0.034 |              P      F
 0.017 |   F P      F                 F
 0.000 |
       +---------------------------------------------
            4k       8k      16k      32k      64k   
```

## Alternative corpus: london

No bank was built for these rows, so there is no LDS and no filter delta.
ms only. Regenerate with `python scripts/london_table.py --readme`.

```
metasmoothness on london (distribution-shift corpus), 2 epochs

run_id                   opt           N     bs    steps         ms
london16k_bs16_adamw     adamw     16000     16     2000     0.9058
london16k_bs16_muon      muon      16000     16     2000      0.964
london16k_bs256_adamw    adamw     16000    256      125     0.9867
london16k_bs256_muon     muon      16000    256      125     0.9321
london32k_bs256_adamw    adamw     32000    256      250     0.9712
london32k_bs256_muon     muon      32000    256      250     0.9619
london64k_bs256_adamw    adamw     64000    256      500     0.6744
london64k_bs256_muon     muon      64000    256      500     0.4475
london128k_bs256_adamw   adamw    128000    256     1000    pending
london128k_bs256_muon    muon     128000    256     1000    pending
```

At bs256 ms holds up to 32k and then collapses at 64k (0.674 adamw, 0.447 muon)
where smollm2 stays near 0.99. That collapse is confounded with learning rate:
the 64k rows ran at lr 1.6e-3 and re-running at 8e-4 recovers most of it
(+0.26 adamw, +0.42 muon), leaving a smaller genuine N effect at fixed lr
(-0.041 adamw, -0.093 muon from 32k to 64k).

## Results

```
metasmoothness | optimizer |  N docs |  bs | N epochs | N steps | EK-FAC LDS | MAGIC LDS | rand filt Δ | EK-FAC filt Δ | MAGIC filt Δ | train loss | heldout loss |    lr | delta L2 | status  | run                              
---------------+-----------+---------+-----+----------+---------+------------+-----------+-------------+---------------+--------------+------------+--------------+-------+----------+---------+----------------------------------
        0.9637 | adamw     | 256,000 |  32 |        2 |   16000 |          - |         - |           - |             - |            - |          - |            - | 5e-05 |        - | planned | plan_adam_eps1e17_256k_bs32      
        0.9741 | adamw     | 128,000 |  32 |        2 |    8000 |          - |         - |           - |             - |            - |          - |            - | 5e-05 |        - | planned | plan_adam_eps1e17_128k_bs32      
        0.8580 | adamw     |  64,000 |  32 |        2 |    4000 |          - |         - |           - |             - |            - |          - |            - | 3e-05 |        - | planned | gpt2medium_64k_bs32              
        0.9869 | adamw     |  64,000 |  32 |        2 |    4000 |          - |         - |           - |             - |            - |          - |            - | 3e-05 |        - | planned | plan_adam_eps1e17_64k_bs32       
        0.9936 | muon      |  64,000 |  32 |        2 |    4000 |          - |         - |           - |             - |            - |          - |            - | 3e-05 |        - | planned | plan_muon_eps1e17_64k_bs32       
        0.9876 | adamw     |  64,000 | 256 |        2 |     500 |     0.4336 |         - |           - |             - |            - |          - |       3.2314 | 1e-04 |        - | planned | plan_adam_eps1e17_64k_bs256      
        0.9947 | muon      |  64,000 | 256 |        2 |     500 |          - |         - |           - |             - |            - |          - |       3.2323 | 1e-04 |        - | planned | plan_muon_eps1e17_64k_bs256      
        0.9866 | adamw     |  32,000 |  32 |        2 |    2000 |     0.4146 |         - |     0.00021 |       0.05037 |            - |          - |       3.2342 | 5e-05 |        - | planned | plan_adam_eps1e17_32k_bs32       
        0.9941 | muon      |  32,000 |  32 |        2 |    2000 |     0.4281 |         - |    -0.00005 |       0.04959 |            - |          - |       3.2310 | 5e-05 |        - | planned | plan_muon_eps1e17_32k_bs32       
        0.9937 | adamw     |  32,000 | 256 |        2 |     250 |     0.4127 |    0.9529 |     0.00021 |       0.05353 |      0.09777 |     3.1074 |       3.2363 | 2e-04 |    36.51 | done    | plan_adam_eps1e17_32k_bs256      
        0.9948 | muon      |  32,000 | 256 |        2 |     250 |     0.4044 |    0.8715 |     0.00020 |       0.05365 |      0.08312 |     3.1013 |       3.2372 | 2e-04 |    31.95 | done    | plan_muon_eps1e17_32k_bs256      
        0.9133 | adamw     |  16,000 |  16 |        2 |    2000 |     0.3872 |    0.1796 |     0.00040 |       0.05014 |      0.01443 |     3.0698 |       3.2497 | 5e-05 |    39.77 | done    | plan_adam_eps1e17_16k_bs16       
        0.9939 | muon      |  16,000 |  16 |        2 |    2000 |     0.4276 |    0.8850 |     0.00121 |       0.04874 |      0.06771 |     3.0518 |       3.2440 | 5e-05 |    26.13 | done    | plan_muon_eps1e17_16k_bs16       
        0.9800 | adamw     |  16,000 |  32 |        2 |    1000 |     0.4586 |    0.9201 |     0.00020 |       0.05202 |      0.09090 |     3.1031 |       3.2473 | 5e-05 |    22.45 | done    | plan_adam_eps1e17_16k_bs32       
        0.7402 | adamw     |  16,000 |  32 |        2 |    1000 |          - |         - |           - |             - |            - |          - |            - | 5e-05 |        - | planned | gpt2medium_16k_bs32              
        0.9952 | muon      |  16,000 |  32 |        2 |    1000 |     0.4567 |    0.8737 |     0.00025 |       0.04801 |      0.06660 |     3.0978 |       3.2441 | 5e-05 |    18.42 | done    | plan_muon_eps1e17_16k_bs32       
        0.9853 | adamw     |  16,000 |  64 |        2 |     500 |     0.4239 |    0.7811 |     0.00021 |       0.04978 |      0.07609 |     3.0715 |       3.2479 | 1e-04 |    27.64 | done    | plan_adam_eps1e17_16k_bs64       
        0.9939 | muon      |  16,000 |  64 |        2 |     500 |     0.4284 |    0.8690 |     0.00022 |       0.04860 |      0.06916 |     3.0648 |       3.2464 | 1e-04 |    24.46 | done    | plan_muon_eps1e17_16k_bs64       
        0.9935 | adamw     |  16,000 | 128 |        2 |     250 |     0.4551 |    0.9441 |     0.00002 |       0.04861 |      0.08073 |     3.1162 |       3.2498 | 1e-04 |    19.76 | done    | plan_adam_eps1e17_16k_bs128      
        0.9944 | muon      |  16,000 | 128 |        2 |     250 |     0.4635 |    0.8480 |     0.00016 |       0.04735 |      0.06097 |     3.1204 |       3.2501 | 1e-04 |    16.43 | done    | plan_muon_eps1e17_16k_bs128      
        0.9896 | adamw     |  16,000 | 256 |        2 |     125 |     0.4176 |    0.8982 |     0.00021 |       0.05517 |      0.09038 |     3.0919 |       3.2543 | 2e-04 |    27.66 | done    | plan_adam_eps1e17_16k_clip1.0    
        0.8580 | adamw     |  16,000 | 256 |        2 |     125 |     0.4189 |   -0.0407 |     0.00017 |       0.04808 |     -0.00018 |     2.8529 |       3.0019 | 1e-04 |    20.72 | done    | plan_adam_eps1e17_16k_gpt2-medium
        0.9150 | adamw     |  16,000 | 256 |        2 |     125 |     0.1733 |    0.0456 |     0.00010 |       0.04315 |      0.00135 |     3.2238 |       3.4343 | 8e-04 |   123.70 | done    | plan_adam_eps1e17_16k_scale0.25  
        0.9878 | adamw     |  16,000 | 256 |        2 |     125 |     0.1760 |    0.9448 |     0.00014 |       0.02532 |      0.07565 |     3.1896 |       3.3022 | 2e-04 |    30.75 | done    | plan_adam_eps1e17_16k_scale0.5   
        0.9930 | adamw     |  16,000 | 256 |        2 |     125 |     0.4235 |    0.9410 |     0.00022 |       0.05291 |      0.09091 |     3.1078 |       3.2572 | 2e-04 |    27.22 | done    | plan_adam_eps1e17_16k_wd0.0      
        0.9930 | adamw     |  16,000 | 256 |        2 |     125 |     0.4244 |    0.9414 |     0.00022 |       0.05296 |      0.09083 |     3.1077 |       3.2572 | 2e-04 |    27.54 | done    | plan_adam_eps1e17_16k_wd0.1      
        0.9930 | adamw     |  16,000 | 256 |        2 |     125 |     0.4253 |    0.9411 |     0.00023 |       0.05288 |      0.09090 |     3.1078 |       3.2572 | 2e-04 |    27.23 | done    | sm_adamw_eps1e17_16k_bs256       
        0.9964 | muon      |  16,000 | 256 |        2 |     125 |     0.4237 |    0.8379 |     0.00040 |       0.05040 |      0.06586 |     3.1135 |       3.2570 | 2e-04 |    20.03 | done    | sm_muon_eps1e17_16k_bs256        
        0.9959 | adamw     |  16,000 | 256 |        4 |     250 |     0.4730 |    0.9534 |     0.00026 |       0.05340 |      0.09793 |     3.0838 |       3.2505 | 1e-04 |    25.16 | done    | plan_adam_eps1e17_16k_ep4        
        0.9950 | adamw     |  16,000 | 512 |        2 |      63 |     0.4142 |    0.9233 |     0.00017 |       0.04004 |      0.07413 |     3.1700 |       3.2751 | 2e-04 |    20.83 | done    | plan_adam_eps1e17_16k_bs512      
        0.9924 | adamw     |   8,000 | 256 |        2 |      63 |     0.3869 |    0.9163 |     0.00018 |       0.02923 |      0.07060 |     3.1309 |       3.2851 | 2e-04 |    20.58 | done    | plan_adam_eps1e17_8k_bs256       
        0.9962 | muon      |   8,000 | 256 |        2 |      63 |     0.3881 |    0.7712 |     0.00012 |       0.02258 |      0.03289 |     3.1560 |       3.2841 | 2e-04 |    11.95 | done    | plan_muon_eps1e17_8k_bs256       
        0.9946 | adamw     |   4,000 | 256 |        2 |      32 |     0.3975 |    0.9295 |     0.00004 |       0.01153 |      0.02505 |     3.2064 |       3.3149 | 1e-04 |     8.04 | done    | plan_adam_eps1e17_4k_bs256       
        0.9037 | muon      |   4,000 | 256 |        2 |      32 |     0.3031 |    0.3020 |     0.00025 |       0.01844 |      0.01345 |     3.1281 |       3.3114 | 4e-04 |    13.17 | done    | plan_muon_eps1e17_4k_bs256       

34 rows shown, 24 with all three metrics. '-' = not yet measured.
```

## How many queries should an LDS be averaged over?

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
[SHAMPOO_RESULTS.md](SHAMPOO_RESULTS.md) hold pre-2026-08-20 measurements, most of which
are excluded from the paper CSVs (they use an old shuffle implementation — see the exclusion 
table in EXPERIMENTS_CSV.md).

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
