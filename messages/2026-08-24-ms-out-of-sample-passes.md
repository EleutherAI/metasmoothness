# 2026-08-24 — ms detects collapsed configurations: out-of-sample test passes

## The boundary, and the test of it

The `ms < 0.95` boundary was drawn from two rows: muon 4k and scale0.25. `bs16`
did **not** inform it — its ms was still training when the boundary was written
down, and its attributability (MAGIC 0.1796) put it squarely in the collapsed
regime. Prediction: ms below 0.95. Measured: **0.9133**.

Three for three, one of them out of sample.

## All 17 rows, sorted by ms

| run | ms | MAGIC |
|---|---|---|
| muon 4k | 0.9037 | 0.3020 |
| **adam bs16** | **0.9133** | **0.1796** |
| adam scale0.25 | 0.9150 | 0.0456 |
| adam bs32 | 0.9800 | 0.9201 |
| adam bs64 | 0.9853 | 0.7811 |
| adam scale0.5 | 0.9878 | 0.9448 |
| adam clip1.0 | 0.9896 | 0.8982 |
| adam 8k | 0.9924 | 0.9163 |
| adam wd0.0 | 0.9930 | 0.9410 |
| adamw anchor | 0.9930 | 0.9411 |
| adam wd0.1 | 0.9930 | 0.9414 |
| adam bs128 | 0.9935 | 0.9441 |
| muon bs64 | 0.9939 | 0.8690 |
| muon bs128 | 0.9944 | 0.8480 |
| adam 4k | 0.9946 | 0.9295 |
| muon 8k | 0.9962 | 0.7712 |
| muon anchor | 0.9964 | 0.8379 |

    ms <  0.95  (n=3)    MAGIC  0.0456 - 0.3020
    ms >= 0.95  (n=14)   MAGIC  0.7712 - 0.9448

The three lowest-ms rows are exactly the three lowest-MAGIC rows, and the gap
between the groups (0.3020 to 0.7712) is wider than the spread within either.

## Two claims, only one of which the data supports

**Supported — ms detects pathological configurations.** Three collapsed rows,
three ms values below 0.95, no false alarms among fourteen healthy rows.

**Not supported — ms ranks attributability.** Inside the healthy band the
ordering is unrelated to MAGIC, and there are direct contradictions:

- bs64 has HIGHER ms than bs32 (0.9853 vs 0.9800) and LOWER MAGIC (0.7811 vs 0.9201)
- muon 8k has the second-highest ms of all (0.9962) and MAGIC of only 0.7712
- in 3 of 4 optimizer pairs muon has the higher ms and the lower MAGIC

`spearman(ms, MAGIC) = +0.060` over the full set, which is what you get when 14
of 17 rows sit inside a 0.017-wide band. **Do not quote that number as a null
result** and do not quote it as support either — it is uninformative in both
directions.

## Caveats a reader needs

- n=3 in the collapsed group. The boundary at 0.95 is drawn between 0.9150 and
  0.9800; anywhere in that gap fits the data equally well.
- **scale0.25 is confounded**: it runs at its tuned lr of 8e-4 against the
  anchor's 2e-4, so its ms drop is not attributable to the logit scale alone.
  scale0.5 at lr 2e-4 moves ms only 0.9930 -> 0.9878. A diagnostic probe at
  logit_scale 0.25 with lr held at 2e-4 is running in `paper_runs/diagnostics/`
  to separate them. bs16 and muon 4k are NOT confounded this way.
- **bs16 hardware**: its ms ran on A40 while its bank is A100 (lotus-0), per the
  D17 accept-and-label ruling. The GPU effect measured so far is ~9.5e-4 on
  query-loss diffs; whether it moves ms at all is unmeasured.
