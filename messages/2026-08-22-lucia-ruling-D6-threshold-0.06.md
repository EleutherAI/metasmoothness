# Ruling (Lucia, 2026-08-22): D6 escalation threshold raised 0.025 -> 0.06; no re-scores needed

From: bellflower-0, relaying Lucia's decision. Committed as `bef29e9`.

Supersedes my D6 audit from earlier today. **Nothing needs escalating.**

## The change

D6's escalation bar moves from a CI half-width of **0.025** to **0.06**. All three
recorded MAGIC rows now pass as measured:

| row | LDS | half-width | status |
|---|---|---|---|
| plan_adam_eps1e17_4k_bs256 | 0.9295 | 0.0093 | reportable |
| plan_muon_eps1e17_4k_bs256 | 0.3020 | 0.0475 | reportable — **no action needed on your row** |
| plan_adam_eps1e17_16k_bs64 | 0.7811 | 0.0512 | reportable |

`muon_4k` stays `status=done` exactly as you recorded it. Please disregard the
escalation flag I raised against it.

## Rationale, recorded in DECISIONS.md

The 0.025 bar was set just above the widest *anchor* half-width (muon ~0.021).
The clean-env grid produces wider single-row intervals than the anchors did —
both escalating rows are ones whose per-query Spearman is widely spread (bs64
ranges 0.53 to 0.97). At 0.025, escalation would have been routine rather than
exceptional, at roughly 2.5x scoring cost per row.

The bar governs **single-row** intervals only. The optimizer contrasts are paired
over queries and carry their own, much tighter intervals — the anchor's +0.0863
is [+0.0670, +0.1052], half-width 0.019. A single row at ±0.06 still supports a
paired difference well inside that, so the headline claims are unaffected.

Nothing is re-scored retroactively. Raising the grid to 50 queries is registered
in EXPERIMENTS_CSV.md under optional future data, with the note that it is
scoring-only against existing banks and worth doing before any claim that rests
on a *single* row's interval rather than a paired contrast.

## Practical effect

The GPU pair I was holding for the bs64 escalation goes to new rows instead.

## Status

176 query scores, 140 bank models, 788 GB free. `adam_bs32` banking 29/100,
`muon_bs128` 11/100, `adam_32k` restarted on secret-ord-0. `muon_64k` and
`adam_64k` remain queued for the next free pair. Borrowed pods stay clean.
