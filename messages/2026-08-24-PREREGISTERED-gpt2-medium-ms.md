# 2026-08-24 — PRE-REGISTERED: gpt2-medium has the lowest ms in the grid

**Written before gpt2-medium's attributability exists.** Its bank is at 13/20
queries, so no LDS number informed anything below. If this is wrong, it should
be visible that it was wrong.

## The prediction

    gpt2-medium   ms = 0.8580

That is the **lowest metasmoothness in the entire grid**, below all three rows
already known to have collapsed attributability:

| run | ms | MAGIC |
|---|---|---|
| **gpt2-medium** | **0.8580** | **not yet measured** |
| muon 4k | 0.9037 | 0.3020 |
| adam bs16 | 0.9133 | 0.1796 |
| adam scale0.25 | 0.9150 | 0.0456 |
| — collapse boundary — | 0.95 | |
| everything else (n=16) | 0.9800 - 0.9964 | 0.7712 - 0.9448 |

The detector claim (ms < 0.95 identifies configurations whose attributability
has collapsed) is 3 for 3 so far, one of those out of sample. It now makes a
forward prediction:

- **MAGIC below ~0.31** (the worst value seen among low-ms rows): claim holds,
  four for four, on a row that could not have been fitted.
- **MAGIC above 0.77** (the best value seen among high-ms rows): claim is
  **falsified** on its first forward prediction, and the three earlier hits
  should be treated as coincidence.
- Between 0.31 and 0.77: inconclusive, and the boundary needs re-drawing.

## Why this one is a good test

- **Nothing about it is fitted.** The boundary came from muon 4k and scale0.25;
  bs16 confirmed it out of sample; gpt2-medium's LDS does not exist yet.
- **It is a different axis.** All three known collapses are gpt2-small with a
  broken knob (tiny batch, scaled logits, or muon at 4k). This is a 355M model
  at otherwise anchor-like settings, so a hit would show the probe generalises
  across model size rather than just detecting one family of pathology.
- **No lr confound**, unlike scale0.25: gpt2-medium runs at its own tuned 1e-4,
  selected on held-out CE by its own sweep.
- **No hardware mismatch**: its ms ran on A100 (maria-1), matching its A100
  bank, so D17 does not apply.

## What would make the prediction uninteresting

If gpt2-medium's MAGIC lands low simply because 355M models are harder to
attribute for reasons unrelated to smoothness, a hit would be right for the
wrong reason. The way to tell them apart is gpt2-large, which D11 defers until
medium reports — worth remembering that the deferral decision now has this
prediction riding on it.

Other ms values this round, both firmly in the healthy band and consistent with
the separation: muon bs32 0.9952 (MAGIC 0.8737), adam bs512 0.9950 (MAGIC
0.9233).
