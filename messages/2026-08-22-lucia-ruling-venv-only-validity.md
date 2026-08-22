# Ruling relay: venv-only validity is FINAL - grid reset executed

From: lotus-0 relaying Lucia, 2026-08-22.

1. D15 closes by ruling, not forensics: the pinned venv is the sole valid
   environment. Every pre-venv bank is invalid - results struck from
   experiments.csv (historical values remain in LDS_RESULTS.md), artifacts
   deleted (old anchors s16k_*, per-epoch banks, all pre-venv gate/scoring
   dirs). No salvage attempts, no mixing tests.
2. The grid after the reset: 72 rows, 2 done (the clean-env 4k pair), 70
   planned - INCLUDING the 16k anchor pair, which re-runs in the venv like any
   other row (tuned lr 2e-4 both arms, from the standing tuning results, which
   are loss-only and remain valid).
3. D16: qk_norm rows are CUT (fine-tune-grafted arch mods don't answer the
   native-architecture question; pre-training mods in is out of scope for now).
   preact_layernorm/arch_control stay blocked pending the same design question.
4. Practical upshot for claiming: the pool of runnable planned rows just grew -
   everything that was "partial" on old banks is claimable fresh, all at final
   tuned lrs, all on bergson-main-paper-429 + the pinned env per NODES.

This is the campaign's validity floor from here on: one env, one code line, one
estimator per metric, no mixed-provenance cells.
