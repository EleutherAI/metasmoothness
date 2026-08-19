# LDS results — EK-FAC / Shampoo / MAGIC / SOURCE / Trackstar

We use GPT-2 for most experiments. We use Olmo-2 for pre-training experiments because it has a modern architecture similar to the GPT-Nano speedrun competition winner that lets us achieve reasonable pre-training losses quickly. We also use it to test QK-norm, an architectural feature it implements that's rumored to improve metasmoothness.

## LDS Evaluation

We produce leave-k-out banks containing re-trained models over 100 subsets @1% held out (50 for Muon SmolLM2 banks). We compute CIs with a 10k-resample bootstrap.

## Results

### Empirical metasmoothness ↔ EK-FAC LDS grid (all models / knobs)

Consolidated view, sorted by metasmoothness. Detailed per-axis sweeps are in the sub-sections
below. `shuffle` = data-order-per-epoch implementation (see note): **rep** = shuffle-once-then-repeat
(same order every epoch, now an unsupported setup). Of course, epochs=1 rows are comparable.

`train loss` = final base-model loss on its training set (mean per-token CE, ≤2000-doc sample). `ΔL1` = relative
parameter-update L1, ‖θ_final−θ_init‖₁ / ‖θ_init‖₁ (init = gpt2, or the OLMo2 reinit for scratch).

| model | opt | eps_root | N | bs | epochs | steps | empirical metasmooth | EK-FAC LDS | MAGIC LDS | train loss | ΔL1 | ΔL2 | dropout | shuffle | ms shuffle |
|-------|-----|----------|-----|-----|--------|-------|-----------|-----------|-----------|-----------|------|------|---------|---------|---|
| OLMo2 scratch | muon | 1e-6 | 16k | 128 | 6 | 750 | −0.000 | 0.0175 | — | 2.92 | 4.56 | 4.10 | 0.0 | rep | per-epoch |
| GPT-2 ft | adam | 0 | 16k | 64 | 2 | 500 | 0.4269 | 0.1097 | — | 2.65 | 0.086 | 0.087 | 0.1 | rep | per-epoch |
| GPT-2 ft | adam | 0 | 8k | 64 | 2 | 250 | 0.6226 | 0.1410 | — | 2.61 | 0.065 | 0.067 | 0.1 | rep | per-epoch |
| GPT-2 ft | adam | 0 | 4k | 64 | 2 | 125 | 0.7724 | 0.1740 | — | 2.61 | 0.051 | 0.053 | 0.1 | rep | per-epoch |
| GPT-2 ft | adam | 1e-10 | 4k | 64 | 2 | 125 | 0.7883 | 0.2097 | −0.02 (n=20) | 2.71 | 0.029 | 0.029 | 0.1 | rep | per-epoch |
| GPT-2 ft | adam | 1e-8 | 4k | 32 | 1 | 125 | 0.837 | 0.1781 | 0.05 (n=20) | 3.07 | 0.009 | 0.010 | 0.1 | rep |  |
| GPT-2 ft | adam | 1e-8 | 4k | 64 | 2 | 125 | 0.8755 | 0.3033 | 0.17 (n=20) | 3.02 | 0.008 | 0.009 | 0.1 | rep | per-epoch |
| GPT-2 ft | adam | 1e-8 | 4k | 64 | 2 | 125 | 0.8755 | 0.3203 | 0.18 (n=20) | 3.01 | 0.008 | 0.009 | 0.0 | rep | per-epoch |
| GPT-2 ft | adam | 1e-6 | 8k | 64 | 2 | 250 | 0.9786 | 0.3019 | 0.98 (n=20) | 3.18 | 0.0024 | 0.003 | 0.1 | rep | per-epoch |
| GPT-2 ft | adam | 1e-8 | 4k | 128 | 4 | 125 | 0.9822 | 0.3369 | 0.43 (n=20) | 2.98 | 0.0074 | 0.009 | 0.1 | rep | per-epoch |
| GPT-2 ft | adam | 1e-6 | 4k | 64 | 2 | 125 | 0.9952 | 0.3173 | 0.86 (n=20) | 3.18 | 0.0016 | 0.002 | 0.1 | rep | per-epoch |
| GPT-2 ft | muon | 0 | 4k | 64 | 4 | 250 | 0.9960 | 0.4683 | — | 3.08 | 0.0057 | 0.006 | 0.1 | rep | per-epoch |
| GPT-2 ft | muon | 1e-6 | 4k | 64 | 4 | 250 | 0.9962 | 0.4738 | 0.76 (n=20) | 3.08 | 0.0057 | 0.006 | 0.1 | rep | per-epoch |

**Per-epoch shuffle note:** the run checkout `feat/magic-grad-accum` shuffles the train set once then
`.repeat(num_epochs)`, so every epoch sees the **same order** (`rep`). The fix — commit `1e6eea7f`
"Shuffle each training epoch independently" (#352), on `origin/main` — reshuffles each epoch. Not yet
rebased; recording which runs use which. All rows above are `rep`; epochs=1 rows are shuffle-agnostic.

**MAGIC code-version note:** MAGIC values depend on the metagradient code, which changed via a rebase on 2026-07-24 ~08:00 that landed `c0f11ba8 "Fix metagrad replay correctness under CUDA dropout and DDP"` (+ grad_accum). eps1e-8 MAGIC = 0.37 on the pre-fix code (07-23) vs 0.17 on the fixed code (07-24). **All MAGIC values in the grid are now on the FIXED code:** eps1e-8 (0.17), eps1e-8 dropout0 (0.18), bs128 (0.43), eps1e-6 4k (0.86 [0.844, 0.877], n=20), eps1e-6 8k (0.98 [0.980, 0.987], n=20), muon (0.76), and — recomputed on the fixed code (were −0.08 / 0.099 pre-fix) — eps1e-10 (−0.02 [−0.065, 0.023], n=20) and bs32 (0.05 [−0.054, 0.145], n=20). eps1e-10 and bs32 sit at ≈0 with CIs spanning zero.

**Dropout:** all GPT-2 runs use dropout **0.1** (gpt2 default; `model_kwargs` empty); OLMo2 from-scratch uses **0.0** (`attention_dropout: 0.0`). Dropout is why the metagrad replay needed the fix (RNG-mask reproduction). Disable via `model_kwargs="resid_pdrop=0.0,attn_pdrop=0.0,embd_pdrop=0.0"`. For eps1e-8 4k bs64 (the two adjacent rows above): metasmoothness 0.876→0.8758, ΔL2 0.009→0.0091, EK-FAC 0.3033→0.3203 (within bootstrap CI). **The MAGIC leg of this comparison is void:** the two MAGIC runs (`magicroll_eps1e8_4k` vs `magicroll_eps1e8_drop0`) produce **bit-identical** score tensors — all 20 queries, max abs difference exactly 0 — because the trainer ran `model.eval()` in both arms, so dropout was inactive regardless of the configured rate (`train_mode` defaulted False at `7b223e31` and was rejected outright for metagradient runs). The 0.17 vs 0.1822 difference comes from scoring identical scores against two different retrain banks, not from dropout. `334fcead` later removed the guard (RNG restore reproduces the masks) and PR #359 re-adds opt-in `train_mode`. **Measured on WikiText with dropout actually active** (`train_mode: true`, gpt2 default 0.1) — see the WikiText table below: MAGIC LDS ≈ 0 with dropout vs 0.9681 without. Note dropout is only active when `train_mode: true`; with the default the trainer calls `model.eval()` and the configured rate is inert.

### Per-epoch shuffle: replication of the headline grid

Re-runs of the headline GPT-2 fine-tuning rows under the **per-epoch shuffle** setup (commit
`1e6eea7f` "Shuffle each training epoch independently", now on the trainer/bank path), to
compare against the published `rep` (shuffle-once-then-`.repeat`) numbers. Same configs, seeds,
datasets and 50-subset @1% banks; one bank per training config, reused for EK-FAC scoring. The
`epochs=1` bs32 row and the OLMo2-scratch endpoint are omitted (shuffle-agnostic / already
recorded at per-epoch −0.000). Reproduction: configs + drivers under
`/mnt/ssd-1/lucia/perepoch/` (`gen_configs.py`, `run_all.sh`, `build_table.py`) on bergson HEAD
`source-wikitext-replication`.

Metasmoothness was also patched to shuffle per epoch (`bergson/magic/metasmoothness.py` now
calls `shuffled_epochs`, matching `run_magic`), so the metasmoothness column and the bank
retrains share the same per-epoch training — previously metasmoothness alone still used `rep`.

**Both metasmoothness and EK-FAC LDS are invariant to the shuffle change** (all 11 configs
measured). Every per-epoch metasmoothness lands within ~0.01 of its `rep` value across the whole
axis (0.44 → 0.99; largest gap is 8k eps0, 0.615 → 0.6226), and every per-epoch EK-FAC LDS sits
inside the bootstrap CI of its `rep` value — the `rep` point is contained in all 11 CIs, and the
largest shift is +0.02 (bs128, 0.337 → 0.358). At 2–4 epochs, one repeated data order vs. that
many independent orders does not move the movement-weighted sign-agreement metric or the
attribution quality it predicts. (`adam eps1e-8 4k` and its `drop0` twin give an identical
metasmoothness 0.8755 — dropout is inert in metasmoothness, which does not set `train_mode`, so
the two train identically.)

| model | opt | eps_root | N | bs | ep | rep ms | per-epoch ms | rep EK-FAC | per-epoch EK-FAC LDS | 95% CI | n | ΔL1 | ΔL2|
|-------|-----|----------|-----|-----|----|--------|--------------|------------|----------------------|--------|---|------|------|
| GPT-2 ft | adam | 0 | 16k | 64 | 2 | 0.437 | 0.4269 | 0.1097 | 0.1186 | [0.074, 0.164] | 50 | 0.0938 | 0.0964|
| GPT-2 ft | adam | 0 | 8k | 64 | 2 | 0.615 | 0.6226 | 0.1410 | 0.1555 | [0.121, 0.190] | 50 | 0.0714 | 0.0742|
| GPT-2 ft | adam | 0 | 4k | 64 | 2 | 0.766 | 0.7724 | 0.1740 | 0.1540 | [0.106, 0.202] | 50 | 0.0565 | 0.0587|
| GPT-2 ft | adam | 1e-10 | 4k | 64 | 2 | 0.781 | 0.7883 | 0.2097 | 0.2020 | [0.164, 0.240] | 50 | 0.0272 | 0.0283|
| GPT-2 ft | adam | 1e-8 | 4k | 64 | 2 | 0.876 | 0.8755 | 0.3033 | 0.3095 | [0.269, 0.351] | 50 | 0.0068 | 0.0084|
| GPT-2 ft | adam | 1e-8 | 4k | 64 | 2 | 0.876 | 0.8755 | 0.3203 | 0.3048 | [0.264, 0.346] | 50 | 0.0068 | 0.0084|
| GPT-2 ft | adam | 1e-8 | 4k | 128 | 4 | 0.982 | 0.9822 | 0.3369 | 0.3576 | [0.317, 0.397] | 50 | 0.0065 | 0.0080|
| GPT-2 ft | adam | 1e-6 | 8k | 64 | 2 | 0.978 | 0.9786 | 0.3019 | 0.2950 | [0.253, 0.338] | 50 | 0.0024 | 0.0028|
| GPT-2 ft | adam | 1e-6 | 4k | 64 | 2 | 0.991 | 0.9952 | 0.3173 | 0.3076 | [0.268, 0.346] | 50 | 0.0015 | 0.0021|
| GPT-2 ft | muon | 0 | 4k | 64 | 4 | 0.996 | 0.9960 | 0.4683 | 0.4648 | [0.422, 0.504] | 50 | 0.0053 | 0.0061|
| GPT-2 ft | muon | 1e-6 | 4k | 64 | 4 | 0.997 | 0.9962 | 0.4738 | 0.4630 | [0.420, 0.503] | 50 | 0.0053 | 0.0061|

The grid spans the full metasmoothness axis (0.44 → 0.997 ms, EK-FAC 0.11 → 0.47) including the
lowest-metasmoothness config (16k eps0), which has the most headroom for a shuffle effect and
still shows none (0.1097 → 0.1186, `rep` inside [0.074, 0.164]). ΔL1/ΔL2 are the per-epoch run's;
they track the `rep` values, running a hair higher at low metasmoothness (e.g. eps0 4k ΔL2 0.0587
vs `rep` 0.0528) — independent per-epoch orders explore marginally more, but the smoothness metric
is unaffected. **Caveat:** this covers GPT-2 fine-tuning at 2–4 epochs only; the shuffle change
could still matter at more epochs, or for the from-scratch OLMo2 pre-training run (full trajectory),
where the `rep`→per-epoch metasmoothness was already recorded at ≈0 → −0.000 (a dead endpoint, so
no headroom to see a difference either way).

### SmolLM2 (`bergson-smollm2-lds-chunks`, `train_{4k,8k,16k,32k}.hf`)

| Optimizer | eps_root | lr | N | epochs | metasmooth | Method | LDS | 95% CI | n | train loss | ΔL1 | ΔL2 | dropout | shuffle | ms shuffle |
|-----------|----------|-----|-----|--------|-----------|--------|-----|--------|---|-----------|------|------|---------|---------|---|
| adam | 1e-6 | 8e-4 | 4k | 2 | 0.9952 | EK-FAC | 0.3173 | [0.285, 0.348] | 50 | 3.18 | 0.0016 | 0.0021 | 0.1 | rep | per-epoch |
| adam | 1e-6 | 8e-4 | 8k | 2 | 0.9786 | EK-FAC | 0.3019 | [0.267, 0.338] | 50 | 3.18 | 0.0024 | 0.0028 | 0.1 | rep | per-epoch |
| adam | 1e-6 | 8e-4 | 16k | 2 | 0.9954 | EK-FAC | 0.3815 | [0.352, 0.412] | 50 | 3.17 | 0.0039 | 0.0042 | 0.1 | rep | per-epoch |
| adam | 1e-6 | 8e-4 | 32k | 2 | 0.9979 | EK-FAC | 0.3575 | [0.325, 0.394] | 50 | 3.19 | 0.0069 | 0.0072 | 0.1 | rep | per-epoch |
| adam | 1e-6 | 8e-4 | 4k | 2 | 0.9952 | Shampoo −1/2 | 0.3071 | [0.270, 0.342] | 50 | 3.18 | 0.0016 | 0.0021 | 0.1 | rep | per-epoch |
| adam | 1e-6 | 8e-4 | 4k | 2 | 0.9952 | Shampoo −1/4 | 0.3264 | [0.294, 0.358] | 50 | 3.18 | 0.0016 | 0.0021 | 0.1 | rep | per-epoch |
| adam | 1e-8 | 8e-4 | 4k | 2 | 0.8755 | EK-FAC | 0.3033 | [0.274, 0.334] | 50 | 3.02 | 0.0079 | 0.0091 | 0.1 | rep | per-epoch |
| adam | 1e-10 | 8e-4 | 4k | 2 | 0.7883 | EK-FAC | 0.2097 | [0.182, 0.239] | 50 | 2.71 | 0.0293 | 0.0295 | 0.1 | rep | per-epoch |
| adam | 0 | 8e-4 | 4k | 2 | 0.7724 | EK-FAC | 0.1740 | [0.140, 0.208] | 50 | 2.61 | 0.0509 | 0.0528 | 0.1 | rep | per-epoch |
| adam | 0 | 8e-4 | 8k | 2 | 0.6226 | EK-FAC | 0.1410 | [0.113, 0.169] | 50 | 2.61 | 0.0646 | 0.0668 | 0.1 | rep | per-epoch |
| adam | 0 | 8e-4 | 16k | 2 | 0.4269 | EK-FAC | 0.1097 | [0.084, 0.136] | 50 | 2.65 | 0.0855 | 0.0872 | 0.1 | rep | per-epoch |
| adam | 0 | 8e-4 | 4k | 2 | 0.7724 | Shampoo −1/2 | 0.2145 | [0.178, 0.249] | 50 | 2.61 | 0.0509 | 0.0528 | 0.1 | rep | per-epoch |
| adam | 0 | 8e-4 | 4k | 2 | 0.7724 | Shampoo −1/4 | 0.1562 | [0.122, 0.192] | 50 | 2.61 | 0.0509 | 0.0528 | 0.1 | rep | per-epoch |
| adam | 0 | 8e-4 | 4k | 2 | 0.7724 | Shampoo −1/8 | 0.1111 | [0.076, 0.145] | 50 | 2.61 | 0.0509 | 0.0528 | 0.1 | rep | per-epoch |
| muon | 1e-6 | 5e-5 | 4k | 4 | 0.9962 | EK-FAC | 0.4738 | [0.432, 0.513] | 50 | 3.08 | 0.0057 | 0.0061 | 0.1 | rep | per-epoch |
| muon | 1e-6 | 5e-5 | 4k | 4 | 0.9962 | Shampoo −1/2 | 0.5217 | [0.481, 0.561] | 50 | 3.08 | 0.0057 | 0.0061 | 0.1 | rep | per-epoch |
| muon | 1e-6 | 5e-5 | 4k | 4 | 0.9962 | Shampoo −1/4 | 0.4304 | [0.388, 0.471] | 50 | 3.08 | 0.0057 | 0.0061 | 0.1 | rep | per-epoch |
| muon | 1e-6 | 5e-5 | 4k | 4 | 0.9962 | Shampoo −1/8 | 0.3026 | [0.260, 0.343] | 50 | 3.08 | 0.0057 | 0.0061 | 0.1 | rep | per-epoch |
| muon | 1e-6 | 1e-4 | 4k | 4 | 0.9930 | EK-FAC | 0.4514 | [0.416, 0.486] | 50 | 2.94 | 0.0110 | 0.0118 | 0.1 | rep | per-epoch |
| muon | 0 | 5e-5 | 4k | 4 | 0.9960 | EK-FAC | 0.4683 | [0.427, 0.508] | 50 | 3.08 | 0.0057 | 0.0061 | 0.1 | rep | per-epoch |
| muon | 0 | 5e-5 | 4k | 4 | 0.9960 | Shampoo −1/2 | 0.5208 | [0.479, 0.561] | 50 | 3.08 | 0.0057 | 0.0061 | 0.1 | rep | per-epoch |
| muon | 0 | 5e-5 | 4k | 4 | 0.9960 | Shampoo −1/4 | 0.4306 | [0.389, 0.471] | 50 | 3.08 | 0.0057 | 0.0061 | 0.1 | rep | per-epoch |
| muon | 0 | 5e-5 | 4k | 4 | 0.9960 | Shampoo −1/8 | 0.3010 | [0.260, 0.343] | 50 | 3.08 | 0.0057 | 0.0061 | 0.1 | rep | per-epoch |
| muon | 0 | 1e-4 | 4k | 4 | 0.9931 | EK-FAC | 0.4544 | [0.419, 0.489] | 50 | 2.94 | 0.0110 | 0.0118 | 0.1 | rep | per-epoch |
| muon | 0 | 1e-4 | 4k | 4 | 0.9931 | Shampoo −1/2 | 0.5206 | [0.479, 0.560] | 50 | 2.94 | 0.0110 | 0.0118 | 0.1 | rep | per-epoch |
| muon | 0 | 1e-4 | 4k | 4 | 0.9931 | Shampoo −1/4 | 0.4093 | [0.371, 0.447] | 50 | 2.94 | 0.0110 | 0.0118 | 0.1 | rep | per-epoch |
| muon | 0 | 1e-4 | 4k | 4 | 0.9931 | Shampoo −1/8 | 0.2682 | [0.229, 0.307] | 50 | 2.94 | 0.0110 | 0.0118 | 0.1 | rep | per-epoch |
| adam | 1e-6 | 8e-4 | 4k | 4 | 0.9989 | — | — | — | — | 3.18 | 0.0016 | 0.0021 | 0.1 | rep | per-epoch |
| adam | 0 | 8e-4 | 4k | 4 | 0.6836 | — | — | — | — | 2.61 | 0.0509 | 0.0528 | 0.1 | rep | per-epoch |

- adam eps1e-6 4k EK-FAC row is the original run (0.3173); re-score with the `1ba43f92` scoring code = 0.3156. muon eps1e-6 5e-5 EK-FAC 0.4738 = will's reported 0.474.
- muon eps_root acts on Muon's AdamW-fallback parameters (embeddings / lm_head / 1D params); the 2D weights use Newton-Schulz, which eps_root does not touch. So muon EK-FAC is nearly flat across eps_root (5e-5: 0.474 @1e-6 vs 0.468 @0; 1e-4: 0.451 @1e-6 vs 0.454 @0), unlike adam (0.317 @1e-6 → 0.174 @0).

#### Shampoo inversion + apply power (PR #384) — see [`SHAMPOO_RESULTS.md`](SHAMPOO_RESULTS.md)

Full write-up of the Meta truncated pseudo-inverse (`pseudoinverse_rank` /
`pseudoinverse_factored`) and the `apply_power` knob lives in the companion file. Headlines:

- `apply_power` is **twice** the `Shampoo −1/*` label here (−1.0 / −0.5 / −0.25 = −1/2 / −1/4 / −1/8).
  The PR's example yaml uses −0.25, i.e. **−1/8**, one rung below the −1/4 it claims.
- The inversion mode is a null on adam (paired Δ ≈ −0.0005) but **small and reproducibly positive on
  muon at −1/4**: +0.0042 [+0.0017, +0.0067] (lr 5e-5) and +0.0056 [+0.0030, +0.0082] (lr 3e-4).
  All effects ≤0.006 LDS, so not practically meaningful.
- `rank_rtol` is monotone harmful; no tuned value beats damping.

#### adam metasmoothness vs eps_root (4k bank config: adamw, lr 8e-4 poly, bs64, epochs=2, betas 0.95/0.975)

Sweep between the eps1e-6 (ms 0.991) and eps0 (ms 0.766) endpoints. Single direction_seed, h=0.1 (noisy).

| optimizer | eps_root | metasmooth | EK-FAC LDS | train loss | ΔL1 | ΔL2 |
|-----------|----------|-----------|-----------|-----------|------|------|
| adam | 1e-6 | 0.991 | 0.3173 | 3.18 | 0.0016 | 0.0021 |
| adam | 1e-7 | 0.978 | — | — | — | — |
| adam | 1e-8 | 0.876 | 0.3033 [0.274, 0.334] | 3.02 | 0.0079 | 0.0091 |
| adam | 1e-9 | 0.907 | — | — | — | — |
| adam | 1e-10 | 0.781 | 0.2097 [0.182, 0.239] | 2.71 | 0.0293 | 0.0295 |
| adam | 0 | 0.766 | 0.1740 | 2.61 | 0.0509 | 0.0528 |

#### adam metasmoothness vs training steps (N-sweep, epochs=2, lr 8e-4, bs64)

Measured at eps_root=0 (un-saturated); eps_root=1e-6 shown for contrast (pinned near the 1.0
ceiling, hides the trend). Steps = N·epochs/bs.

| optimizer | N | steps | metasmooth (eps0) | EK-FAC LDS (eps0) | train loss (eps0) | ΔL1 (eps0) | ΔL2 (eps0) | metasmooth (eps1e-6) |
|-----------|-----|-------|-------------------|-------------------|-------------------|------------|------------|----------------------|
| adam | 4k | 125 | 0.766 | 0.174 | 2.61 | 0.0509 | 0.0528 | 0.991 |
| adam | 8k | 250 | 0.615 | 0.141 | 2.61 | 0.0646 | 0.0668 | 0.978 |
| adam | 16k | 500 | 0.437 | 0.110 | 2.65 | 0.0855 | 0.0872 | 0.995 |
| adam | 32k | 1000 | — | — | — | — | — | 0.998 |

At eps0, LDS falls monotonically with steps, tracking metasmoothness (both ~halve 4k→16k).

Same direction with data held fixed (4k, eps0, epochs 2→4 = 125→250 steps): 0.766 → 0.663.

#### muon metasmoothness vs training steps and eps_root (N-sweep, epochs=4, lr 5e-5 poly, bs64)

Extends the muon 4k bank rows along two axes. Steps = N·epochs/bs = N·4/64. Metasmoothness only —
no leave-k-out banks were built for the new points.

**N × eps_root grid.** Every cell is muon, lr 5e-5 poly, bs64, ep4, betas 0.95/0.975, wd 0.01,
seed 42, fd_step 0.1, direction_seed 0 — only `data.dataset` and `eps_root` vary.

| N | steps | metasmooth (eps0) | metasmooth (eps1e-8) | metasmooth (eps1e-6) | EK-FAC LDS (eps1e-6) | train loss | ΔL1 | ΔL2 |
|-----|-------|-------------------|----------------------|----------------------|----------------------|-----------|------|------|
| 4k | 250 | 0.996 | 0.9961 | 0.9965 | 0.4738 [0.432, 0.513] | 3.08 | 0.0057 | 0.0061 |
| 8k | 500 | 0.9956 | — | 0.9957 | — | — | — | — |
| 16k | 1000 | 0.9951 | — | 0.9952 | — | — | — | — |
| 32k | 2000 | — | — | 0.9947 | — | — | — | — |

Unfilled cells are not-yet-run, not failures: eps0 32k was cancelled mid-run to free GPUs; eps1e-8
was only run at 4k. Resume with `run_muon_ms_eps.sh 0 eps0 32` / `run_muon_ms_eps.sh 1e-8 eps1e8 8 16 32`
(both skip any point that already has `metasmoothness.json`). No leave-k-out banks were built for the
new points, so their LDS/loss columns are empty.

**Muon is flat on BOTH axes.** Across steps: 0.9965→0.9947 over an 8× step increase (250→2000), a
drift of 0.0018. Across eps_root at 4k: 0.996 / 0.9961 / 0.9965 for eps 0 / 1e-8 / 1e-6 — a 0.0005
spread over six orders of magnitude. The eps0 and eps1e-6 columns agree to ~1e-4 at every N where
both were measured. All of this is within single-direction_seed fd_step=0.1 noise.

Muon's split is on `param.ndim` (`optim.py`): `ndim == 2` → Newton-Schulz, everything else → an
AdamW branch, and `adamw_eps_root` appears only inside that AdamW branch. For GPT-2 as the trainer
builds it (`lm_head` untied from `wte`, so both are separate 2D tensors), that is 162,915,840 params
on the Newton-Schulz path and 121,344 (98 tensors: LayerNorm weights/biases, Linear biases) on the
AdamW path — eps_root reaches 0.07% of parameters. **Note:** embeddings and `lm_head` are 2D and go
through Newton-Schulz, *not* the AdamW branch. muon EK-FAC LDS is likewise near-flat across eps_root
(0.468 @eps0 vs 0.474 @eps1e-6; 1e-4 lr: 0.454 vs 0.451).

Caveat: every muon point sits at ~0.995, so these do **not** distinguish "muon is robust to run
length" from "the metric has no headroom left to fall". eps_root is *not* the knob that un-saturates
muon — the whole axis is pinned. Batch size does not un-saturate it either: muon eps0 at bs16 is
0.9932 (vs 0.996 at bs64), where adam at eps1e-8 collapses to 0.500 over the same bs64→bs16 change.
The knob that still moves the metric elsewhere in the grid is output logit scale (0.609 at 0.25).
Contrast adam at eps0, which roughly halves over a comparable steps range (0.766→0.437, 125→500).

#### MAGIC score finiteness and magnitude vs eps_root (GPT-2 4k)

Checked every MAGIC score tensor on disk under `/mnt/ssd-2/lucia/muon4k`. 560,000 score elements
across 8 runs: **100% finite (0 NaN, 0 Inf)**.

| run dir | optimizer | eps_root | bs | NaN | finite | score range |
|---------|-----------|----------|----|-----|--------|-------------|
| magicroll_eps1e6_4k | adamw | 1e-6 | 64 | 0 | 100% | ±0.0083 |
| magicroll_eps1e8_4k | adamw | 1e-8 | 64 | 0 | 100% | ±2.59 |
| magicroll_eps1e10_4k | adamw | 1e-10 | 64 | 0 | 100% | −3.43 … 3.92 |
| magicroll_eps1e8_drop0 | adamw | 1e-8 | 64 | 0 | 100% | ±2.59 |
| magic_eps1e8_4k | adamw | 1e-8 | 64 | 0 | 100% | −0.59 … 0.19 |
| magicroll_bs32 | adamw | 1e-8 | 32 | 0 | 100% | −2.22 … 2.96 |
| magicroll_bs128 | adamw | 1e-8 | 128 | 0 | 100% | −0.123 … 0.099 |
| magicroll_muon_eps1e6_5e5 | muon | 1e-6 | 64 | 0 | 100% | −0.0084 … 0.0024 |

Score magnitude grows monotonically as eps_root falls (adamw, bs64): ±0.0083 @1e-6 → ±2.59 @1e-8
(~300×) → 3.92 @1e-10.

**MAGIC at eps_root=0, batch size matched to the bank: 98.4% finite.** The earlier
`magic_eps0_muon_5e-5` attempt used bs16 while its bank `run_eps0_5e-5/N4k` was trained at bs64, so
it replayed a trajectory no retrain in that bank took; it wrote no scores and its checkpoints have
been deleted. Re-run at **bs64**, one variable changed from the known-finite
`magicroll_muon_eps1e6_5e5/q0` config (`eps_root` 1e-6 → 0.0), same code `6c597de0`:

| run | optimizer | eps_root | bs | NaN | finite | score range |
|-----|-----------|----------|----|-----|--------|-------------|
| magic_eps0_bs64_q0 | muon | 0 | 64 | 64/4000 | 98.4% | −6.58e-4 … 5.26e-4 |

The 64 NaN are scattered (indices 122, 175, 192 … 3912, 3963), not one contiguous batch. Finite mean
−1.40e-7. Magnitude is the same order as the eps1e-6 muon reference (−0.0084 … 0.0024, 0 NaN).

**The `Score summary` log line does not distinguish "some NaN" from "all NaN".** This run logs
`DescribeResult(nobs=4000, minmax=(nan, nan), mean=nan, ...)` while its saved tensor is 98.4%
finite — `scipy.stats.describe` propagates a single NaN to every field. The earlier
"100% NaN, all 4000 scores" reading of `magic_eps0.log:1038` was that same log line, and that run
saved no tensor, so its NaN fraction was never measured.

Caveats on this row: `nproc_per_node: 4` (the reference used 8), one query (`query_q0.hf`), and
`num_subsets: 0`, so it establishes finiteness only — no LDS. The run exits non-zero because
`validate_scores` calls `pearsonr` on fewer than 2 subsets; scores are written before that.
Run dir `/mnt/ssd-2/lucia/muon4k/magic_eps0_bs64_q0`.

**EK-FAC at eps_root=0.** The muon eps0 EK-FAC score matrix
(`ekfac_eps0_muon_5e-5/scores/scores/scores.bin`) is 4000×50 with **0 NaN, all `written` flags True**,
range [−1090, 1902], mean 0.52 — and yields LDS 0.4683. EK-FAC never forms the unregularized
second-moment reciprocal that MAGIC's replay divides by.

**Metasmoothness across the muon eps_root / batch-size cells.**

| config | metasmooth |
|--------|-----------|
| muon, eps0, 4k, bs16, ep4 (1000 steps) | 0.9932 |
| muon, eps0, 4k, bs64, ep4 (250 steps) | 0.996 |
| muon, eps1e-6, 4k, bs64, ep4 (250 steps) | 0.9965 |

**Per-parameter-group split (muon, eps0, bs16).** `metasmoothness.json` carries a `groups` breakdown
for muon runs, scoring the Newton-Schulz and AdamW paths separately; the decomposition is exact
(`score = Σ share·score`, verified to 1e-9):

| group | score | movement share | numel |
|-------|-------|----------------|-------|
| muon_2d (Newton-Schulz) | 0.9932 | 0.99960 | 162,915,840 |
| adamw_1d (eps_root's only reach) | 0.9867 | 0.00040 | 121,344 |

The 1D group carries 0.04% of total L1 movement, so it could have scored −1.0 and moved the
aggregate only to ~0.992. At eps_root=0 that group scores 0.9867 on its own normalization.
Per-coordinate movement is also smaller for the 1D group (1.13e-4 vs 2.11e-4).

bs16 barely moves muon metasmoothness (0.996 at bs64 → 0.9932 at bs16, −0.003), unlike adam at
eps1e-8 where bs16 collapses the metric to 0.500.

#### adam metasmoothness — other knobs (eps_root=1e-8, N=4k, epochs=2; baseline bs64/wd0.01/scale1.0 = 0.876)

| knob | values → metasmooth |
|------|---------------------|
| batch size (steps=8000/bs) | 16→0.500, 32→0.810, **64→0.876**, 128→0.979, 256→0.997 |
| weight decay | 0→0.867, **0.01→0.876**, 0.1→0.877, 0.3→0.862 |
| output logit scale | **1.0→0.876**, 0.5→0.840, 0.25→0.609 |

Weight decay: no effect over 0–0.3. Output logit scale (fixed
`lm_head` forward multiply, non-trainable): scaling outputs down lowers metasmoothness.

**Batch size deconfounded — metasmoothness at FIXED steps=125** (eps1e-8, 4k, epochs = bs/32):

| optimizer | batch size | epochs | steps | metasmooth | EK-FAC LDS | train loss | ΔL1 | ΔL2 | ms shuffle |
|-----------|------------|--------|-------|------------|-----------|-----------|------|------|---|
| adam | 32 | 1 | 125 | 0.837 | 0.1781 | 3.07 | 0.0087 | 0.0100 |  |
| adam | 64 | 2 | 125 | 0.8755 | 0.3033 | 3.02 | 0.0079 | 0.0091 | per-epoch |
| adam | 128 | 4 | 125 | 0.9822 | 0.3369 | 2.98 | 0.0074 | 0.0087 | per-epoch |
| adam | 256 | 8 | 125 | 0.9984 | — | — | — | — | per-epoch |

Batch size raises metasmoothness even at fixed step count (0.837→0.992) — a genuine effect, not just
a step-count proxy. (LDS at bs256/ep8 is **infeasible on the current code**: the leave-k-out bank build
OOMs on the 47.5 GiB A40s — the fp32 LM-head logits for 256×512 tokens alone are ~26 GB, and grad
checkpointing doesn't touch that. Effective-batch-256 needs micro-batching via `grad_accum_steps`, which
is not a field on the current `Magic`/trainer (it lives only on the unmerged `feat/magic-grad-accum`
branch). Switching to it would change the metagradient code and break MAGIC-value comparability across
the grid, so bs256 LDS is left out. The bs32→128 rows already establish the fixed-steps trend.)

#### Non-eps ms-knobs — LDS + update magnitude (eps1e-8, 4k, steps=125, dropout 0.1)

Built banks for the fixed-steps ms-knobs to test the confounder analysis: does a knob vary metasmoothness *without* changing update magnitude (ΔL2)?

| optimizer | knob | metasmooth | EK-FAC LDS | train loss | ΔL1 | ΔL2 |
|-----------|------|-----------|-----------|-----------|------|------|
| adam | output scale 1.0 (ref) | 0.876 | 0.303 | 3.02 | 0.008 | 0.009 |
| adam | output scale 0.5 | 0.840 | 0.220 | 3.10 | 0.013 | 0.016 |
| adam | output scale 0.25 | 0.609 | 0.169 | 3.36 | 0.016 | 0.019 |
| adam | gradient clip 1.0 | 0.876 | 0.307 | 3.03 | 0.006 | 0.007 |

Output-scale lowers metasmoothness and EK-FAC LDS but **raises** ΔL2 (0.009→0.019) — so it is NOT a clean isolate-metasmoothness knob (update magnitude moves too, opposite direction to the eps_root case). Gradient clipping (max_grad_norm 1.0) is effectively a **no-op** at these settings: metasmoothness is unchanged (0.876, identical to ref) and EK-FAC LDS barely moves (0.307 vs 0.303) — grad norms rarely exceed 1.0, so clipping almost never fires.

### OLMo2 from-scratch (`olmo2_reinit` 124M, SmolLM2 corpus)

Re-initialized OLMo2 (124M; hidden 768, 12 layers) trained **from scratch** (not fine-tuned) — the
pre-training proxy that motivated this investigation. muon, 6 epochs, bs128, lr 9e-3, eps_root 1e-6,
betas 0.95/0.975, wd 0.1, 50 subsets.

| optimizer | N | steps | metasmooth | Method | LDS | 95% CI | n | train loss | ΔL1 | ΔL2 | dropout | shuffle |
|-----------|-----|-------|-----------|--------|-----|--------|---|-----------|------|------|---------|--------|
| muon | 16k | 750 | 0.010 | EK-FAC | 0.0175 | [−0.036, 0.071] | 50 | 2.92 | 4.56 | 4.10 | 0.1 | rep |

Both metasmoothness (0.010) and LDS (0.018) ≈ 0 — the extreme low-metasmoothness endpoint of the
grid.

#### Pre-training is metasmooth only in its *tail* — exclude the early steps (main result)

A from-scratch pre-training run is unscoreable over its **full** trajectory, but the **last epoch**
is metasmooth. Attributing only over the tail — and evaluating against a leave-k-out bank that drops
each subset's docs only in that tail — makes it attributable, at the model's **full 3.23-nats loss**
(the trained model is unchanged; only the attribution window moves).

**Headline — same scorer (EK-FAC), same aggregation (per-query mean), window off → on:**

| attribution window | frac | metasmooth | Method | LDS | 95% CI | n | train loss | shuffle |
|---|---|---|---|---|---|---|---|---|
| full run (0–749) | 0.0 | 0.010 | EK-FAC (per-query) | 0.0175 | [−0.036, 0.071] | 28×50 | 2.92 | rep |
| **last epoch (624–749)** | **0.833** | **0.984** | **EK-FAC (per-query)** | **0.161** | **[0.123, 0.198]** | 50×50 | 3.23 | per-epoch |

EK-FAC uses no metagradient, so this comparison is immune to the MAGIC code-version issue below, and
both rows are the mean-across-50-queries Spearman with a query-bootstrap CI. The CIs are **disjoint**
(full-run ≤ 0.071 < tail ≥ 0.123): a ~9× gain from excluding the early steps, with the *same*
trajectory-agnostic scorer. 46/50 queries have positive tail rho. The full-run row is the original
`rep` bank (metasmooth 0.010); the per-epoch full run is metasmooth −0.000 — both ≈ 0, so the shuffle
change does not move the dead endpoint.

**MAGIC (matched tail-metagradient) — per-query run in progress.** MAGIC is the exact tail
metagradient, so it should predict the tail bank better than the trajectory-agnostic EK-FAC. The
per-query MAGIC LDS (matched to the 0.161 EK-FAC number above) is being computed — one backward per
query. _(An earlier aggregate-query MAGIC number is **not** used here; aggregate-query metrics are
non-standard and quarantined in [the appendix](#appendix--aggregate-query-numbers-do-not-cite).)_
- **Intervention** — `weight_start_frac` / `weight_start_step`, following arXiv 2503.13751 App. C.3
  (data weights enter the loss only from step *k*, chosen to maximize metasmoothness; DataComp *k* =
  2800/3125, IFT *k* = "150 steps from the end"). `DataStream` pins weights to a constant 1 before
  *k*, so the forward trajectory — and the loss — are identical across the window; only the backward
  and the bank's leave-out window change. Branch `feat/ms-pretrain` (off `fix/per-epoch-shuffle`),
  commit `5833a9b3`, 13 regression tests. MAGIC backward over the tail unrolls only the last epoch
  (~6× cheaper) and is numerically clean, where a full-run backward NaNs on the chaotic trajectory.

#### metasmoothness vs attribution window (16k, fixed model, loss 3.23 throughout)

Moving the attribution start forward, on the one fixed 16k/750-step run. `predicted LDS` applies the
`LDS = 0.284·ms^0.637` law (spearman 0.98 over the 11 both-measured configs above). Coverage =
whether every doc still appears in the window (needed for a valid bank): the tail spans (1−frac)×6
epochs, so full coverage requires frac ≤ 0.833.

| frac | window (steps) | epochs in window | metasmooth | predicted LDS | coverage |
|---|---|---|---|---|---|
| 0.0 | 0–749 | 6.00 | −0.000 | 0.00 | full |
| 0.25 | 187–749 | 4.50 | 0.025 | 0.03 | full |
| 0.5 | 375–749 | 3.00 | 0.355 | 0.15 | full |
| 0.6 | 450–749 | 2.40 | 0.669 | 0.22 | full |
| 0.75 | 562–749 | 1.50 | 0.793 | 0.24 | full |
| **0.833** | **624–749** | **1.00** | **0.984** | **0.28** | **full (boundary)** |
| 0.896 | 672–749 | 0.62 | 0.986 | 0.28 | 38% docs unseen |
| 0.95 | 712–749 | 0.30 | 0.993 | 0.28 | 70% unseen |
| 0.99 | 742–749 | 0.06 | 0.990 | 0.28 | 94% unseen |

`frac=0.833` (exactly the last epoch) is the operating point: near-ceiling metasmoothness at the
largest window with full coverage. Confirmed at a second `direction_seed`: `window_0.75` scores 0.793
(seed 0) / 0.838 (seed 1), `total_movement_l1` 0.24% apart — a real effect, not the sign-statistic
noise that dominates near zero.

#### Window sweep repeated on a 188-step (4k) run (control)

The `window4k` control repeats the window sweep on a 188-step (N=4k) run. Both runs are 6 epochs.

| epochs in window | frac | 4k (188-step) ms | 16k (750-step) ms |
|---|---|---|---|
| 4.50 | 0.25 | 0.154 | 0.025 |
| 3.00 | 0.50 | 0.147 | 0.355 |
| 2.40 | 0.60 | 0.512 | 0.669 |
| 1.50 | 0.75 | 0.971 | 0.793 |
| 1.00 | 0.833 | 0.993 | 0.984 |
| 0.30 | 0.95 | 0.995 | 0.993 |

The **same** 188-step 4k model (loss 4.98) scores **0.0095** over its full 6-epoch window and
**0.993** over its last 1-epoch window.

#### metasmoothness vs pre-training length (steps axis) — flat at ~0

Full-run attribution, muon eps_root 1e-6, per-epoch shuffle, `direction_seed=0`. Shortening the run
does **not** help. Every full run is 6 epochs; all sit at ~0 while loss ranges over 1.9 nats.

| N | steps | metasmooth | final-epoch loss |
|---|---|---|---|
| 4k | 188 | 0.0095 | 4.98 |
| 8k | 375 | 0.0177 | 3.95 |
| 16k | 750 | −0.0002 | 3.23 |
| 32k | 1500 | 0.0051 | 3.09 |

#### Optimizer / architecture knobs (full-run) — nothing helps without wrecking loss

One-factor from the 16k baseline (muon, eps_root 1e-6, lr 9e-3, wd 0.1, bs128, 6ep). `opt_adamw`'s
0.647 is at loss 6.18. Every knob at a usable loss stays ~0.

| knob | metasmooth | final-epoch loss |
|---|---|---|
| baseline | −0.0002 | 3.23 |
| optimizer → adamw (lr 8e-4) | 0.647 | 6.18 |
| lr → 3e-3 | 0.019 | 2.69 |
| weight_decay → 0 | 0.003 | 3.27 |
| eps_root → 1e-4 | 0.004 | 3.34 |
| batch_size 64 (3ep) | 0.005 | 4.31 |
| batch_size 256 (12ep) | 0.006 | 1.34 |

#### Caveats / reproduction

- **Def. 2 is ill-conditioned near zero.** The score is a movement-weighted average of ±1 sign
  agreements; when its true value is ~0 the signs are near coin flips, so a tiny numerical change
  flips the score's sign while `total_movement_l1` barely moves. Concretely the 16k full run scored
  0.0101 (bergson 0.10.0, `rep`) vs −0.000165 (0.13.1+, per-epoch) with movement agreeing to 0.12%.
  So: differences below ~0.02 carry no information; confirm any promising cell at a second
  `direction_seed`; quote `total_movement_l1` alongside. `use_tf32_matmuls` was verified bit-identical
  here (not the cause of that gap) but is kept off as a precaution — all rows here are fp32.
- The full-run OLMo2 bank's per-query Δloss sd is 0.066 (vs ~0.001 for GPT-2 fine-tuning banks).
  Re-scoring the surviving index (`/mnt/ssd-2/lucia/scratch_olmo/N16k_scores`)
  with different damping/SOURCE/Trackstar does not change it.
- **Code / configs (branch `feat/ms-pretrain` @ `/mnt/ssd-1/lucia/bergson-ms-pretrain`):** window
  sweep + analysis `experiments/pretrain_metasmoothness/{run_sweep,analyze}.py`; tail-only bank
  generator `gen_tail_bank.py` (frac=0.833, single `magic` step → tail MAGIC scores + tail-only 50-
  subset bank + inline LDS). Tail bank + models: `runs/tail_bank_083_full/` (reusable). Full narrative
  writeup: `experiments/pretrain_metasmoothness/RESULTS.md`.

### WikiText (`bergson-wikitext-512-chunks`)

Two banks, both adamw, 4 epochs, betas 0.95/0.975.

metasmooth measured for each bank's training config (bs64, 4 epochs): lotus 0.998, epsroot0 0.609.

**Dropout collapses MAGIC LDS on this dataset.** The `dropout*` rows use `train_mode: true` (PR #359,
`334fcead`) so gpt2's default 0.1 dropout is actually active during training and the metagradient
replay; every other row has dropout configured but inert (`model.eval()`). Same model / dataset /
lr 8e-4 / bs64 / ep4 / eps_root 1e-6 as lotus. Both dropout runs are statistically indistinguishable
from zero, against lotus's 0.9681 (per-query 0.92–0.99, p ~1e-250). The two dropout runs share
bit-identical MAGIC scores — they differ only in bank construction, so subset construction is not
what drives the result: ragged subsets (2–47 docs) gave 0.1862 and fixed 46-doc subsets gave −0.2286.

Caveats: the dropout rows are **1 query** and 15/99 subsets, vs lotus's 50 queries and 400 subsets.
At N=15 the 95% CI on −0.2286 is roughly [−0.7, +0.35] — wide enough that its sign is meaningless,
but narrow enough to exclude ~0.9. MAGIC scores themselves are finite under dropout (4608 scores,
0 NaN), so this is a prediction-quality collapse, not a numerical failure.

`chunk_length` is **not** a factor here: re-running `lotus_final_q01_50` with `chunk_length: 0`
instead of 512 reproduces 0.9681 exactly (mean 0.9681, median 0.9815, min 0.5977 vs 0.5976).

| optimizer | Bank | eps_root | metasmooth | Method | Variant | LDS | n | train loss | ΔL1 | ΔL2 | Run dir | dropout | shuffle |
|-----------|------|----------|-----------|--------|---------|-----|---|-----------|------|------|---------|---------|--------|
| adam | lotus | 1e-6 | 0.998 | MAGIC | full q01–50 | 0.9681 | 50 | 3.06 | 0.0033 | 0.0040 | `runs/lotus_final_q01_50` | 0.1 (inert) | rep |
| adam | dropout_s15 | 1e-6 | — | MAGIC | q0, 15 subsets @46 | −0.2286 (p=0.41) | 1 | 3.00 | — | — | `runs/gpt2_wikitext_dropout_s15` | 0.1 **active** | rep |
| adam | dropout | 1e-6 | — | MAGIC | q0, 99 ragged subsets | 0.1862 (p=0.065) | 1 | 3.00 | — | — | `runs/gpt2_wikitext_dropout` | 0.1 **active** | rep |
| adam | lotus | 1e-6 | 0.998 | MAGIC | bwd eval | 0.9688 | 50 | 3.06 | 0.0033 | 0.0040 | `runs/lotus_bwd_eval` | 0.1 | rep |
| adam | lotus | 1e-6 | 0.998 | SOURCE | damp0 | 0.3902 | 50 | 3.06 | 0.0033 | 0.0040 | `runs/lotus_source_q50_damp0_validate` | 0.1 | rep |
| adam | lotus | 1e-6 | 0.998 | SOURCE | adam | 0.2068 | 50 | 3.06 | 0.0033 | 0.0040 | `runs/lotus_source_adam_q50_validate` | 0.1 | rep |
| adam | lotus | 1e-6 | 0.998 | SOURCE | default | −0.3871 | 50 | 3.06 | 0.0033 | 0.0040 | `runs/lotus_source_q50_validate` | 0.1 | rep |
| adam | lotus | 1e-6 | 0.998 | EK-FAC | docspace | 0.2588 | 50 | 3.06 | 0.0033 | 0.0040 | `runs/lotus_ekfac50q_docspace_vs_lotus_bank` | 0.1 | rep |
| adam | lotus | 1e-6 | 0.998 | EK-FAC | allium-0 | 0.0543 | 50 | 3.06 | 0.0033 | 0.0040 | `runs/lotus_scores_ekfac50q_allium-0_validate` | 0.1 | rep |
| adam | lotus | 1e-6 | 0.998 | Trackstar | docs p32 noopt | 0.2002 | 50 | 3.06 | 0.0033 | 0.0040 | `runs/gpt2_lotus_trackstar50q_docs_p32_noopt_vs_lotus_bank` | 0.1 | rep |
| adam | lotus | 1e-6 | 0.998 | Trackstar | docs | 0.1838 | 50 | 3.06 | 0.0033 | 0.0040 | `runs/gpt2_lotus_trackstar50q_docs_vs_lotus_bank` | 0.1 | rep |
| adam | lotus | 1e-6 | 0.998 | Trackstar | default | 0.1767 | 50 | 3.06 | 0.0033 | 0.0040 | `runs/lotus_trackstar_q50_validate` | 0.1 | rep |
| adam | epsroot0 | 0 | 0.609 | SOURCE | source2 | 0.1531 | 50 | 1.86 | 0.0834 | 0.0848 | `runs/epsroot0_source2_q50_validate` | 0.1 | rep |
| adam | epsroot0 | 0 | 0.609 | SOURCE | source2 adam hybrid | 0.1446 | 50 | 1.86 | 0.0834 | 0.0848 | `runs/epsroot0_source2_adam_hybrid_validate` | 0.1 | rep |
| adam | epsroot0 | 0 | 0.609 | SOURCE | source2 adam | 0.0811 | 50 | 1.86 | 0.0834 | 0.0848 | `runs/epsroot0_source2_adam_q50_validate` | 0.1 | rep |
| adam | epsroot0 | 0 | 0.609 | EK-FAC | allium-0 | −0.0109 | 50 | 1.86 | 0.0834 | 0.0848 | `runs/epsroot0_scores_ekfac50q_allium-0_validate` | 0.1 | rep |
| adam | epsroot0 | 0 | 0.609 | MAGIC | spotcheck | NaN | 1 | 1.86 | 0.0834 | 0.0848 | `runs/epsroot0_bank` | 0.1 | rep |

- Excluded (n=1 spotchecks): lotus MAGIC `lotus_mq_eval` 0.9893, `lotus_q01_spotcheck` 0.9818, `lotus` 0.9177, `lotus_mq_eval_prefix_backup` 0.9665; and `lotus_interim_q01_08` (n=8, 0.9675).
- The `gpt2_epsroot0_trackstar50q*` runs are excluded: their config has `retrained_dir=runs/lotus`, so they score epsroot0 gradients against the lotus bank.

#### WikiText eps_root=1e-8 batch-size sweep — recovering the MAGIC paper's regime

The MAGIC paper reports WikiText LDS >=0.9 at **eps_root=1e-8** by tuning the batch size (its
batch size was lost, so `examples/magic/gpt2_wikitext.yaml` had substituted eps_root=1e-6 / bs64).
Sweeping batch size at eps_root=1e-8 (lotus recipe: adamw, lr 8e-4 poly, betas .95/.975, wd .01,
ep4 fixed so the data is identical, dropout inert; per-query MAGIC, 30 subsets @1% x 5 queries;
`grad_accum` used to hold the per-GPU metagradient micro-batch ~16 — see note):

| global bs | steps | metasmooth | per-query MAGIC LDS | 95% CI |
|-----------|-------|------------|---------------------|--------|
| 64  | 288 | 0.8947 | 0.169 | [0.02, 0.41] (n=4) |
| 128 | 144 | — | 0.483 | [0.28, 0.69] |
| 192 | 96  | — | 0.644 | [0.54, 0.75] |
| 224 | 83  | — | 0.407 | [0.22, 0.62] |
| 256 | 72  | — | **0.952** | [0.936, 0.969] |

The bs64 metasmoothness (0.8947; ΔL1 0.0121, ΔL2 0.0144, fd_step 0.1, direction_seed 0, code
`37d7b386` — the same commit the sweep banks/scores ran on; run dir
`/mnt/ssd-2/lucia/wikitext_ms_eps1e8_bs64/`) makes this row a measured high-ms/low-MAGIC point:
ms 0.89 with MAGIC 0.17, right in line with the SmolLM2 eps1e-8 bs64 cell (ms 0.8755, MAGIC 0.17).
The other batch sizes' metasmoothness is not yet measured.

LDS is low at small batch (bs64 0.17, matching the SmolLM2 bs64 eps1e-8=0.17 grid value) and jumps
to 0.95 at bs256. The intermediate points (128-224) have wide **overlapping** CIs — estimator noise
at 30 subsets / 5 queries, not a clean monotonic ramp; the high regime onsets sharply near bs256,
whose tight per-query spread (0.93-0.98) marks it genuine. num_epochs is fixed at 4, so every batch
size trains on the same data — the driver is the larger batch, not more/less training.

**bs256 confirmed at scale — definitive: N=100/m=50 = 0.9519 [0.9435, 0.9592]** (per-query
0.83-0.98, all high). Converges with every smaller estimate: N=30/m=5 = 0.9520, N=82/m=11 =
0.9501, N=82/m=21 = 0.9520. So eps_root=1e-8 recovers the paper's >=0.9 with batch size as the
knob, no eps_root=1e-6 substitute. Committed as `examples/magic/gpt2_wikitext.yaml` (bs256,
grad_accum 2, N=100, subset_fraction 0.01; LDS in header). The N=100 run: bank resumed 82->100
after a crash, then per-query MAGIC (query_method none) over test[1:51]; all on bergson
`37d7b386` (0.10.1 — the code the scores/bank were built on, not current main). Sweep
code/results: `experiments/batchsize_eps1e8/`.

**grad_accum note:** `grad_accum_steps>1` rescales the MAGIC metagradient (ga2 ~= 0.68x ga1,
Spearman rank corr ga1-vs-ga2 = 0.9995, top-50 movers identical) but preserves rank, so the
per-query Spearman LDS is unchanged — this is what lets the high-batch metagradient fit few GPUs
(hold per-GPU micro-batch ~16 = ~50 GB via ga, vary global batch as the science). Raw MAGIC score
**magnitudes are NOT comparable across ga** — the canonical "grad_accum breaks comparability"
caveat is about magnitudes, not LDS. Verified bs64 single-doc q1, nproc4, ga1 vs ga2.

# Provenance / reproduction

- SmolLM2 eps_root=1e-6 4k adam bank: HF `EleutherAI/bergson-smollm2-lds-4k` (+ `run_config.yaml`, `subsets.json`). Size-scaling banks: `runs/ekfac_vs_n/N{4,8,16,32}k`. adam eps_root=0 bank: `/mnt/ssd-2/lucia-adam-shampoo/epsroot0_4k_bank/` (code `b3790ba9`). muon banks: `/mnt/ssd-2/lucia/muon4k/{run,run_1e-4,run_eps0_5e-5,run_eps0_1e-4}/N4k` (differ only in eps_root and lr).
- SmolLM2 scoring summary.csv under `/mnt/ssd-2/lucia-adam-shampoo/*/validate/` and `/mnt/ssd-2/lucia/muon4k/**/validate/`; scoring code `1ba43f92` worktree (+ `feat/shampoo-quarter-power` for the Shampoo power variants). WikiText run dirs under `runs/`.
- All banks above: `data.chunk_length = 0`.
- muon N × eps_root metasmoothness sweep (2026-07-27): `/mnt/ssd-2/lucia/muon_ms_steps/` — driver
  `run_muon_ms_eps.sh <eps_root> <tag> [sizes...]`, per-point dirs `{eps1e6→msmuon_*k, eps0/, eps1e8/}`,
  each holding the generated `ms.yaml`, `ms.log` and `metasmoothness.json` (`["score"]`). The
  eps1e-6 column is under the top-level `msmuon_{8,16,32}k/` (driver `run_muon_ms_steps.sh`).
  Run from `/mnt/ssd-1/lucia/bergson-damping`; **requires `PYTHONPATH=<repo>`** — the `bergson`
  console script puts its own bin dir on `sys.path`, not the cwd, so it raises
  `ModuleNotFoundError: No module named 'bergson'` on every rank even from the repo root.
- MAGIC finiteness audit: score tensors read from `/mnt/ssd-2/lucia/muon4k/{magicroll_*,magic_*}/q*/scores.pt`.
  EK-FAC score matrices are `scores/scores/scores.bin` + `info.json` — a structured dtype with
  explicit `offsets`/`itemsize` (float32 `score_i` + bool `written_i`, 8-byte stride); reading it
  without the offsets silently mis-strides and fabricates NaNs.
- Reading muon eps0 configs: `muon4k/ekfac_eps0_muon_{5e-5,1e-4}/validate/config.yaml` report
  `optimizer: adamw, eps_root: 1.0e-08`. Those are unused defaults on the *validate* step — the bank
  was pre-built and supplied via `retrained_dir: muon4k/run_eps0_5e-5/N4k`, whose config is the real
  one (`optimizer: muon`, `eps_root: 0.0`, lr 5e-5, bs64, ep4). Grepping the scoring configs by
  directory name gives the wrong optimizer/eps_root.

## Appendix

### Appendix — aggregate-query numbers (do not cite)

**Aggregate-query metrics are non-standard and must not be used** (see the CLAUDE.md rule: LDS is
per-query only). They are recorded here, out of the main results, purely so a value already computed
is not silently lost or mistaken for a headline number. An aggregate-query LDS averages the query
gradients into one query and then correlates over subsets — a single, noisy correlation over few
points, **not comparable** to the per-query means in the main tables.

OLMo2 tail-only bank (frac=0.833), aggregate-query:

| window | Method | aggregation | LDS | 95% CI | n |
|---|---|---|---|---|---|
| last epoch (0.833) | EK-FAC | aggregate-query | 0.054 | [−0.246, 0.357] | 50 |
| last epoch (0.833) | MAGIC | aggregate-query | 0.705 | [0.521, 0.824] | 50 |

The per-query EK-FAC number for the same bank is 0.161 (main table); the per-query MAGIC number is
being computed to replace the 0.705 here. The 0.705 was verified unaffected by the `c0f11ba8`
metagrad fix (OLMo2 dropout 0.0; single-shot `backward()` already carries the 1/world_size
correction; no grad-accum path on `feat/ms-pretrain`), so it is a valid *aggregate-query* number —
it is quarantined for being aggregate-query, not for being wrong.

### Information for Coding Agents

Some invalid banks trained with `chunk_length = 512` are listed in the [Appendix](#appendix--invalid-banks-chunk_length--0), so coding agents don't accidentally pull invalid related data.

On 8xA40s MAGIC OOMs at batch size = 256 without the WIP gradient accumulation branch.

### Additional configuration.

Shampoo −1/2 / −1/4 / −1/8 = methods `shampoo` / `shampoo_quarter` / `shampoo_p025`
(apply power −1.0 / −0.5 / −0.25 on the fitted factors).

metasmooth = empirical metasmoothness (Chang et al. 2024, Def. 2; h=0.1, direction_seed 0),
measured at each row's exact training config. 

All SmolLM2 size banks are epochs=2 (`runs/ekfac_vs_n/configs/N{4,8,16,32}k.yaml`); muon banks epochs=4.
The two adam epochs=4 rows have no corresponding bank.

### Invalid banks (`chunk_length ≠ 0`)

Trained with `data.chunk_length = 512` (WikiText data is already 512-chunked). These
banks and any LDS scored against them are invalid — **do not add them to the tables
above.** Kept here only for the record.

**stdadam** (adamw, betas 0.9/0.999, eps_root 0, `chunk_length=512`):

| Method | Variant | LDS | n | Run dir |
|--------|---------|-----|---|---------|
| EK-FAC | docs | 0.0700 | 50 | `runs/gpt2_stdadam_ekfac50q_docs_vs_stdadam_bank` |
| EK-FAC | default | 0.0232 | 50 | `runs/gpt2_stdadam_ekfac50q_vs_stdadam_bank` |
| Trackstar | docspace | 0.0417 | 50 | `runs/gpt2_stdadam_trackstar50q_docspace_vs_stdadam_bank` |

(other near-0 stdadam variants under `runs/gpt2_stdadam_*`)

**eps / batch-size sweep banks** (`chunk_length=512`): `gpt2_wikitext_eps{1e-8,2e-8,5e-8,5e-9,7e-9}`, `gpt2_wikitext_bs{448_eps1e-7,448_eps1e-8,448_eps1e-9,448_eps5e-9,480_eps1e-8,512_eps1e-8}`, `gpt2_wikitext_A40s`, `gpt2_wikitext_paper_bs32`, `gpt2_wikitext_metasmoothness_eps1e-4`, `batch_probe`. LDS for these are in `runs/eps_search/RESULTS.md`.

Old/exploratory (not chunk-verified): `examples/exp_log.md` (Jun 30, token-length sweeps).
