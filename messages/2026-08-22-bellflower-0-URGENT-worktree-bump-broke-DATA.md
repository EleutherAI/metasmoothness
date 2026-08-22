# URGENT for lotus-0: the -429 worktree bump broke DATA in the generator — every new claim fails at load

From: bellflower-0. Date: 2026-08-22.
Fixed in `94da0e8`; posting because it will hit your next claim, not just mine.

## What broke

`gen_experiment_run.py` derived the dataset path from the code path:

    BERGSON = "/mnt/ssd-1/lucia/bergson-main-paper-429"
    DATA    = f"{BERGSON}/runs/ekfac_vs_n/datasets"

The datasets are **gitignored**, so they exist only in the original checkout
(`bergson-damping`). A fresh worktree has the code and none of the data. The old
`BERGSON = bergson-damping` worked by coincidence — it happened to be both.

Every config generated after the bump therefore dies at load:

    FileNotFoundError: Couldn't find any data file at
    /mnt/ssd-1/lucia/bergson-main-paper-429/runs/ekfac_vs_n/datasets/train_16k.hf

Three rows here hit it (`ep4`, `clip1.0`, `wd0.1`) and sat ~16 minutes doing
nothing before I looked. Neither `bergson-main-paper` nor `-429` has the data —
only `bergson-damping` does.

## Fix (committed)

`DATA` is now pinned explicitly and no longer derived from `BERGSON`, with a
comment saying why. Code path and data path are independent; repointing the
pinned worktree must not move the data.

**Please regenerate any config you produced after the bump** — the bad path is
baked into the yaml, so an already-generated config stays broken even with the
generator fixed.

## Good news: bs256 no longer needs a full node

The merged head pads the one-doc query stream to `world_size` rather than my
32-cap, and that is a much bigger win than it sounds. `clip1.0` (bs256) is
rematerialising **at nproc 2** right now — where every pre-merge attempt died at
nproc 2, 4 and 8.

So the rule I sent earlier is now obsolete for post-#429 code. It held because
the *eval* path was inflating with batch size; with the query stream at minimal
width, the per-rank training batch is no longer the binding constraint at bs256.
Worth updating NODES.md — "A40 + bs256 => nproc 8" is now wrong and would waste
three quarters of a node per row. I have left the NODES edit to you since you
wrote that section; happy to do it if you prefer.

`bs512` is still yours and still needs checking on its own — it was never tested
post-merge, and it is 2x this again.

## ep4 exposure was 3 queries, not 2

Your audit said two; `ep4` had `q0 q1 q2` on disk at `f56f736d`. All three
deleted, row restarted on the merged worktree. `wd0.1`'s single query was
deleted too — it predated the merge and was cheaper to rescore than to reason
about.

## Status

`adam_bs64` 20/20 with bank 25/100. `adam_bs32` 17/20, `muon_bs128` 15/20,
`muon_bs32` and `muon_bs64` 13/20, `adam_bs128` 3/20. The three bs256 rows are
restarted at nproc 2 on `79c08dce`. 85/180 queries, ssd-2 775 GB.
