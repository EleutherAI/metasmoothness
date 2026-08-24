# 2026-08-25 — what actually causes the two ms collapses (controlled probes)

Raw data: `data/ms_diagnostics.csv`. Probe outputs live in
`paper_runs/diagnostics/<name>/` with config, log and JSON.

## The problem these solve

Every grid row tuned to its own lr differs from the anchor in **two** ways, so a
collapse cannot be attributed to the named knob. Two rows collapse (`bs16`,
`scale0.25`) and both are confounded. An ms probe costs three trainings and no
bank, so holding lr fixed is cheap.

| probe | bs | lr | scale | ms | isolates |
|---|---|---|---|---|---|
| anchor | 256 | 2e-4 | 1.00 | 0.9930 | reference |
| scale0.5 row | 256 | 2e-4 | 0.50 | 0.9878 | logit scale at anchor lr |
| scale0.25 row | 256 | **8e-4** | 0.25 | 0.9150 | scale AND 4x lr |
| **scale0.25 at anchor lr** | 256 | 2e-4 | 0.25 | **0.9812** | **logit scale alone** |
| bs16 row | **16** | 5e-5 | 1.00 | 0.9133 | batch AND 1/4 lr |
| **bs16 at anchor lr** | **16** | 2e-4 | 1.00 | **0.5127** | **batch size alone** |
| **anchor at bs16 lr** | 256 | **5e-5** | 1.00 | **0.9948** | **low lr alone** |

## The two collapses have opposite causes

**scale0.25 is the learning rate.** Holding lr at 2e-4, logit scale 0.25 scores
0.9812 -- healthy. The row scores 0.9150 because it also carries 8e-4.

**bs16 is the batch size, and its tuned lr is protecting it.** Batch 16 at the
anchor lr scores **0.5127**, by far the lowest ms ever measured here and far
below the row's own 0.9133 at its tuned 5e-5. Low lr alone is harmless (0.9948
at bs256), so the direction is unambiguous: small batch breaks smoothness, and
5e-5 partially rescues it.

So "row X collapses because of knob X" is right for bs16 and wrong for
scale0.25. Do not describe the logit-scale axis as damaging MAGIC.

## Two consequences worth noting

**ms has far more dynamic range than the grid suggested.** Healthy rows sit in a
0.017-wide band, which is why rank correlations over the grid are uninformative.
But 0.5127 exists. The band is narrow only because every gently-tuned config
looks alike to the probe -- not because the probe saturates.

**Tuning lr per row partly masks the effects the grid is measuring.** The CONTROLS
protocol measures each config at its own best lr, which is correct for
attributability comparisons, but it means bs16's LDS of 0.1796 is the *rescued*
number. At the anchor lr it would presumably be worse. Any claim of the form
"attribution degrades at small batch" is really "degrades at small batch, after
lr tuning has already compensated".

## Still open

The same confound applies to every non-anchor-lr row: bs32 (5e-5), bs64, bs128,
ep4, 4k, 64k (1e-4). None of those collapsed, so it is less urgent, but a claim
attributing any of their effects to the named knob needs the same probe. Each is
three trainings.
