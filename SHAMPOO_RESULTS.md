# Shampoo experiments — apply power, and the Meta truncated pseudo-inverse (PR #384)

Companion to [`LDS_RESULTS.md`](LDS_RESULTS.md), covering the Shampoo-specific knobs: the
preconditioner apply power and the Meta `distributed_shampoo` rank-truncated pseudo-inverse added in
[bergson PR #384](https://github.com/EleutherAI/bergson/pull/384) (`feat/rank-pseudoinverse-inversions`,
head `0a105617`). #384 adds `pseudoinverse_rank` / `pseudoinverse_factored` — the
`torch.linalg.matrix_rank` cutoff `clamp(rank_rtol·max(λ), min=rank_atol)` used by
`PseudoInverseConfig`, applied per Kronecker factor for `pseudoinverse_factored` so a grid cell
survives only if both λ_A and λ_G pass their own factor's threshold — plus a generic
`HessianConfig.apply_power` replacing the ad-hoc `shampoo_quarter` / `shampoo_p025` method names.

**Three conclusions.** The PR's example yaml sits one power rung below what it describes; the
inversion mode is irrelevant on adam at every `rank_rtol` that isn't destructive; but on **muon at
the −1/4 rung it is a small, reproducible improvement** (+0.004 to +0.006 paired, CIs clear of
zero, two independent banks) — still ≤2.6% relative, so not practically meaningful.

## Setup

All rows share one bank and one query set, so they are directly comparable to the `Shampoo −1/*`
rows in `LDS_RESULTS.md`.

- **adam bank:** `/mnt/ssd-2/lucia-adam-shampoo/run_adam/N4k` — GPT-2 ft, adam, eps_root 1e-6, N=4k,
  bs64, 2 epochs, dropout 0.1 (inert), `rep` shuffle; metasmoothness 0.991. 100 subsets @1%.
- **muon bank:** `/mnt/ssd-2/lucia/muon4k/run/N4k` — eps_root 1e-6, lr 5e-5, 4 epochs; metasmooth 0.997.
- **large-update muon bank:** `/mnt/ssd-2/lucia/muon4k/run_3e-4/N4k` — eps_root 1e-6, lr 3e-4,
  4 epochs, 50 subsets. Previously **unlisted in `LDS_RESULTS.md`**. Measured against the `gpt2`
  init: **ΔL1 = 0.0304, ΔL2 = 0.0328**, 5.3× the lr 5e-5 bank and the largest-update muon bank
  available (comparable to adam eps_root=1e-10, 0.0293). The same measurement reproduces the
  recorded 0.0057/0.0061 (lr 5e-5) and 0.0110/0.0118 (lr 1e-4) exactly. Its Shampoo −1/4 LDS is
  0.2140 (damped 0.1) vs the lr 5e-5 bank's 0.4314.
- **No muon `eps_root=1e-8` bank exists.** Checked exhaustively by `retrained/` dir over all of
  `/mnt/ssd-2` (29 banks): every 1e-8 bank is adamw, every muon bank is eps_root 1e-6 or 0. Nothing
  on `/mnt/ssd-1` has a `retrained/` dir; the HF cache holds no bank repos; `muon4k/sweep/lr*` holds
  only `base` (metasmoothness models, not LDS banks). muon's `eps_root` reaches only the
  AdamW-fallback params (121,344 of 163,037,184; see the muon eps_root section of
  `LDS_RESULTS.md`). muon EK-FAC LDS across eps_root: 0.474 @1e-6 vs 0.468 @0.
- **Queries:** `runs/ekfac_vs_n/datasets/query_50.hf` (50 chunks, `chunk_length: 0`);
  **train:** `runs/ekfac_vs_n/datasets/train_4k.hf`.
- LDS = mean per-query Spearman; CIs from a 10k-resample bootstrap over the 50 queries. Δ columns are
  a **paired** bootstrap (same resampled query indices in both arms), much tighter than comparing
  marginal CIs.
- Mean `baseline_loss` = **3.269046** in every adam-bank run here and in the canonical Jul-20
  `shampoo_quarter` run, confirming identical bank/queries.

## `apply_power` is twice the `LDS_RESULTS` label

| # | method | apply_power | inversion | LDS | 95% CI | paired Δ vs row 1 | run dir |
|---|--------|-------------|-----------|-----|--------|-------------------|---------|
| 1 | `shampoo_quarter` (canonical Jul-20, pre-#384) | — | damped 0.1 | 0.3264 | [0.292, 0.359] | — | `lucia-adam-shampoo/shampoo_quarter_adam` |
| 2 | `shampoo` (#384) | −0.5 | damped 0.1 | 0.3262 | [0.292, 0.358] | −0.0001 [−0.0018, +0.0016] | `lucia/shampoo_rohan_adam_p05` |
| 3 | `shampoo` (#384) | −0.5 | damped 0.1 | 0.3246 | [0.291, 0.357] | −0.0018 [−0.0035, −0.0001] | `lucia/shampoo_rohan_adam_p0.5_damped` |
| 4 | `shampoo` (canonical, old) | — | damped 0.1 | 0.3071 | [0.272, 0.341] | −0.0193 [−0.035, −0.003] | `lucia-adam-shampoo/shampoo_adam` |
| 5 | `kfac` (canonical, old) | — | damped 0.1 | 0.3156 | [0.283, 0.347] | −0.0108 [−0.022, −0.000] | `lucia-adam-shampoo/ekfac_adam` |

**Rows 2 and 3 reproduce row 1.** #384's `apply_power = -0.5` is the same operator as the old,
never-committed `method: shampoo_quarter`, which `LDS_RESULTS.md` labels **Shampoo −1/4**. Row 4 (old
plain `method: shampoo`, labelled **−1/2**) is a different and significantly worse operator, so the
match is not an artifact of the arms being trivially close.

```
LDS_RESULTS label  =  apply_power / 2
  −1/2  ↔  apply_power −1.0  (#384 default)
  −1/4  ↔  apply_power −0.5
  −1/8  ↔  apply_power −0.25
```

The fitted Shampoo Hessian is stored as the eigenvalue grid `λ_G ⊗ λ_A`, so a joint power `p` is a
per-factor power `p/2`, and the `LDS_RESULTS` labels are per-factor.

Rows 2 and 3 are **independent replicates** of the same config — row 2 refit the Shampoo factors in
its own `run_path`, row 3 reused a shared fit — and they bracket the canonical value
(0.3262 / 0.3246 vs 0.3264). So the ~0.002 offset is run-to-run fit variation, **not** cross-version
drift between the Jul-20 code and PR head. The same holds at the lower rung: 0.2630 (refit) vs
0.2626 (shared fit), spread 0.0004.

## The PR's example yaml is one rung low

`examples/pipelines/rohan_shampoo_gpt2_lds.yaml` sets `apply_power: -0.25`, which is Shampoo
**−1/8**, not the quarter-power −1/4 the PR describes and not what the Meta config it cites
(per-side `L^{†1/4} G R^{†1/4}`) calls for. **Use `apply_power: -0.5` for −1/4.**

| bank | rung (`apply_power`) | inversion | LDS | 95% CI | run dir |
|------|------|-----------|-----|--------|---------|
| adam eps1e-6 4k | −1/8 (−0.25) | `pseudoinverse_factored`, default rtol — **the yaml as committed** | 0.2625 | [0.229, 0.294] | `shampoo_rohan_adam` |
| adam eps1e-6 4k | −1/8 (−0.25) | damped 0.1 | 0.2630 | [0.230, 0.294] | `shampoo_rohan_adam_dampedref` |
| adam eps1e-6 4k | −1/8 (−0.25) | damped 0.1 | 0.2626 | [0.229, 0.294] | `shampoo_rohan_adam_damped` |
| muon eps1e-6 5e-5 4k | −1/8 (−0.25) | `pseudoinverse_factored`, default rtol | 0.3024 | [0.260, 0.343] | `shampoo_rohan_muon` |
| muon eps1e-6 5e-5 4k | −1/8 (−0.25) | damped 0.1 | 0.3020 | [0.260, 0.343] | `shampoo_rohan_muon_damped` |

Confirmed independently on the muon bank: the yaml's config gives 0.3024 against the canonical
`shampoo_p025` (−1/8) row's 0.3026. Taken at face value the yaml looks like a 0.064 regression versus
the existing grid (paired Δ vs row 1 = −0.0639 [−0.077, −0.051]), but it is simply evaluating a
different power. This is a config bug; the `apply_power` machinery itself is exact.

Row 3 of the table above is the first **Shampoo −1/8** measurement for adam eps1e-6 4k —
`LDS_RESULTS.md` previously had −1/8 only for adam eps0 (0.1111) and muon (0.3026 / 0.3010).

## The inversion mode: null except at −1/4 on muon

Paired against `damped_inverse(0.1)` on identical code, same fit, same power. All 7 comparisons:

| bank | ΔL1 | rung | pinv | damped | paired Δ (`pseudoinverse_factored` − damped) |
|------|-----|------|------|--------|---------------------------------------------|
| adam eps1e-6 4k | 0.0016 | −1/4 (−0.5) | 0.3241 | 0.3246 | −0.0005 [−0.0020, +0.0009] |
| adam eps1e-6 4k | 0.0016 | −1/8 (−0.25) | 0.2625 | 0.2626 | −0.0001 [−0.0005, +0.0004] |
| muon 1e-6 lr5e-5 4k | 0.0057 | −1/4 (−0.5) | 0.4356 | 0.4314 | **+0.0042 [+0.0017, +0.0067]** |
| muon 1e-6 lr5e-5 4k | 0.0057 | −1/8 (−0.25) | 0.3024 | 0.3020 | +0.0004 [−0.0002, +0.0011] |
| muon 1e-6 lr3e-4 4k | 0.0304 | −1/2 (−1.0) | 0.3455 | 0.3495 | −0.0040 [−0.0162, +0.0082] |
| muon 1e-6 lr3e-4 4k | 0.0304 | −1/4 (−0.5) | 0.2196 | 0.2140 | **+0.0056 [+0.0030, +0.0082]** |
| muon 1e-6 lr3e-4 4k | 0.0304 | −1/8 (−0.25) | 0.1253 | 0.1249 | +0.0005 [−0.0003, +0.0013] |

Five of seven straddle zero. The two that do not are **both muon at −1/4**, on two independent
banks, both positive. But the effect does **not** extend along the power axis on the bank where all
three rungs were measured: at −1/2 the point estimate is negative and at −1/8 it is +0.0005. So it
is rung-specific and non-monotone in rung. The largest effect anywhere is 0.006 LDS on a
0.12–0.44 base.

**Shampoo rung profile, `damped_inverse(0.1)`, muon banks.** The two banks have the same ordering;
the 5.3× larger update shifts the whole profile down by roughly a third without reordering it.

| bank | ΔL1 | −1/2 | −1/4 | −1/8 |
|------|-----|------|------|------|
| muon 1e-6 lr5e-5 4k | 0.0057 | 0.5217 | 0.4314 | 0.3026 |
| muon 1e-6 lr3e-4 4k | 0.0304 | 0.3495 | 0.2140 | 0.1249 |

(lr5e-5 −1/2 and −1/8 are the canonical Jul-20 `shampoo` / `shampoo_p025` runs; −1/4 is the
same-code control from this campaign.)

Caveat: every row is a single run per arm, and §"`apply_power` is twice the label" shows run-to-run
fit variation of ~0.002 on the adam bank. The paired Δ is computed on a shared fit, so fit variation
cancels within a pair, but a replicate of each pair would be worth having before leaning on the
sign — particularly given 2 of 7 comparisons reach significance with no monotone pattern in rung.

**`rank_rtol` sweep** (adam eps1e-6 4k, `apply_power −0.25`). Monotone harmful: the best setting is
the *least* truncating one, and it beats damping by +0.0011, i.e. noise. There is no tuned
`rank_rtol` that makes the pseudo-inverse a win. "% zeroed" = fraction of the `λ_G ⊗ λ_A` grid sent
to 0.

| `rank_rtol` | % of grid zeroed | LDS | 95% CI |
|-------------|------------------|-----|--------|
| 1e-6 | 0.14 | 0.2637 | [0.230, 0.295] |
| 1e-4 | 6.99 | 0.2637 | [0.231, 0.295] |
| default `numel(λ)·eps` (≈2e-3) | 52.70 | 0.2625 | [0.230, 0.294] |
| 1e-3 | 61.37 | 0.2564 | [0.223, 0.288] |
| 1e-2 | 98.10 | 0.2214 | [0.190, 0.251] |
| 1e-1 | 99.99 | 0.1440 | [0.110, 0.177] |
| — (`damped_inverse` 0.1) | — | 0.2626 | [0.229, 0.294] |

The Meta default is **not** a mild cutoff at this scale: `rank_rtol = numel(λ)·eps` with I up to
18432 is ≈2.2e-3, zeroing over half the grid. LDS moves by 0.0001 nonetheless. The scores do differ
(corr 0.99998, max rel diff 1%).

## On the cited tweet

The yaml cites `_arohan_/status/2064631528806908134`, which gives *optimizer* hyperparameters for
distributed_shampoo on a speedrun task (`lr=0.01, wd=0.1, beta2=0.9, eps=1e-15, freq=1`) and says
only qualitatively that pseudo-inverse tuning mattered for rank-deficient matrices; it specifies no
rtol/atol. Those HPs do not transfer here — nothing in this pipeline trains with Shampoo. (X 402s on
direct fetch; read via a text mirror, so treat as second-hand.)

## Reproduce

Worktree at PR head `0a105617`, 6 GPUs (`nproc_per_node: 6`; the committed yaml says 8):

```sh
cd <worktree-at-0a105617>
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 PYTHONPATH=$PWD \
  python -m bergson examples/pipelines/rohan_shampoo_gpt2_lds.yaml
```

`bergson` is not installed into site-packages — it imports from `PYTHONPATH`/cwd only, so the
`PYTHONPATH` (or `python -m` from the worktree) is what selects PR code over another worktree's.

Per run: ~40s query gradients, ~110s Shampoo fit, ~3min apply+score, ~3min validate over the
100-model bank. #384's 32 inversion / preconditioner unit tests pass on GPU.

Run dirs: `/mnt/ssd-2/lucia/shampoo_rohan_{adam,muon}*` (`_damped`, `_dampedref`, `_p05`,
`_p0.5_damped`, `_p0.5_pinv`, `_rtol*`), `/mnt/ssd-2/lucia/shampoo_muon5e5_p05_{pinv,damped}`,
`/mnt/ssd-2/lucia/shampoo_muon3e4_{pinv,damped}` and `shampoo_muon3e4_p{1.0,0.25}_{pinv,damped}`.
Metasmoothness for the lr3e-4 bank: `/mnt/ssd-2/lucia/muon4k/metasmooth_muon_3e-4` (running).
LDS + paired bootstrap over any set of
`validate/summary.csv` via a scratch `lds_compare.py` derived from
`scripts/ekfac_vs_n/bootstrap_table.py`.

## Caveats

- Canonical rows (1, 4, 5) were fit with `nproc_per_node: 8`, all #384 rows with 6. Row 2 reproducing
  row 1 to ±0.0002 shows the shard count does not affect the result at this scale.
- The old `shampoo_quarter` implementation is unrecoverable: it exists on no ref
  (`git log --all -S shampoo_quarter -- bergson` is empty) and was a local uncommitted edit. Its
  fitted factors under `shampoo_quarter_adam/scores/hessian/` are mode 600/uid 1001 and could not be
  diffed against the new fit. The equivalence above is behavioural, not by code inspection.
- The `rank_rtol` sweep and the muon arms come from a parallel session's runs
  (`_rtol*`, `_damped`, `_p0.5_*`, `shampoo_rohan_muon*`); the `_p05` and `_dampedref` replicates and
  all paired statistics here were computed independently from the stored `summary.csv` files.
