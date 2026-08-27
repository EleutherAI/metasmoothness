# 2026-08-27 - Hub bank repos renamed to LDS-retrain-bank-*, all public

All 24 retrain-bank repos are renamed, per Lucia's request:

    EleutherAI/metasmoothness-bank-plan_adam_eps1e17_16k_bs32
      -> EleutherAI/LDS-retrain-bank-adamw-N16k-bs32

Scheme: `LDS-retrain-bank-<adamw|muon>-N<docs>-bs<batch>[-<knob>]`, knob only
for the non-bs variants (`-clip1.0`, `-wd0.0`, `-scale0.25`, `-ep4`,
`-gpt2-medium`). No eps token (`=` is not a legal Hub character, hence `N16k`
not `N=16k`). The `plan_`/`sm_` prefixes are gone; both anchors are plain
`...-adamw-N16k-bs256` / `...-muon-N16k-bs256`.

- Every repo had 101 models before renaming; none were skipped.
- All 24 are public (14 were private).
- The Hub redirects old names, so existing snapshot_download calls keep
  working, but create new repos through `bank_repo_id()` in
  scripts/publish_bank.py / scripts/bank_card.py - both now build the new
  name from a run_id. Do not hand-construct `metasmoothness-bank-*` names.
- The data-attribution collection follows the renames (26 items, 0 old-named).
