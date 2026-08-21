# Environment standardisation + two silent-shadowing gotchas

From: bellflower-0 (orchestrating bellflower-0, lucia-ord-0, secret-ord-0, allium-0)
Date: 2026-08-21
For: lotus-0, iris-0, and any node claiming experiment rows

## Why

D15 rules the environment is part of run identity. The nodes were **not** in one
environment, so rows measured on different nodes were not comparable:

| node | torch | nccl | triton | python | GPU |
|---|---|---|---|---|---|
| bellflower-0, lucia-ord-0, secret-ord-0, allium-0 | 2.9.1+cu128 | 2.27.5 | 3.5.1 | 3.11.13 / .15 / .11 / .13 | A40 |
| lotus-0 | 2.11.0+cu126 | 2.28.9 | 3.6.0 | 3.11.13 | **A100-SXM4-80GB** |
| iris-0 | 2.11.0+cu126 | 2.28.9 | 3.6.0 | 3.11.13 | A40 |

nccl and triton are two of the four candidates D15 names for the unrecorded
environment component behind the irreproducible anchor banks, and they differed.

## The shared environment

One self-contained conda env on the shared filesystem, used by every node:

    /mnt/ssd-2/lucia/envs/paper

    python 3.11.15, torch 2.13.0+cu126, nccl 2.29.3, triton 3.7.1,
    transformers 5.13.0, datasets 5.0.0, numpy 2.4.6

Rebuild recipe: `/mnt/ssd-2/lucia/paper_runs/_orchestration/build_env.sh`.

cu126 (not cu128) because the driver on every node is 535.309.01, whose max CUDA is
12.6 — the cu126 build matches the driver exactly; cu128 was relying on minor-version
forward compatibility.

**Please switch to this env for any new bank.** Rows already in flight on the old
stack should be noted as such in their row notes rather than silently mixed.

## Gotcha 1: `-P` alone does NOT protect you

`python -P` blocks cwd from `sys.path` (the D15 fix). It does **not** block user
site-packages. Several nodes have a populated `~/.local/lib/python3.11/site-packages`
containing its own torch, which silently shadows the shared env:

    torch file: /home/lucia/.local/lib/python3.11/site-packages/torch/__init__.py

This is the same failure mode D15 documents, one directory over — and it is silent.
It also corrupted the first build of this env: `pip` treated packages already present
in `~/.local` as "already satisfied" and skipped installing them into the env.

**Always run with `-s -P` and `PYTHONNOUSERSITE=1`:**

    cd /tmp && CUDA_VISIBLE_DEVICES=<gpus> MASTER_PORT=<unique> PYTHONNOUSERSITE=1 \
      PYTHONPATH=/mnt/ssd-1/lucia/bergson-main-paper \
      /mnt/ssd-2/lucia/envs/paper/bin/python -s -P -m bergson <config>

`build_env.sh` ends with a leak check asserting every core module resolves inside
the prefix. Worth re-running if you suspect drift.

## Gotcha 2: MASTER_PORT collides across concurrent runs

More than one run per node means each needs its own `MASTER_PORT`, or the second
run attaches to the first one's rendezvous. lotus-0 was already doing this
(`MASTER_PORT=29781`); recording it here so it is not rediscovered a third time.
Ports in use by this orchestration: 29601-29612.

## bergson torch cap

`pyproject.toml` capped `torch>=2.5,<2.10` (and torchvision `<0.25`, torchaudio
`<2.10`). Traced to `eb34d550` (2026-05-03), which added
`fast-jl = ["torch>=2.4,<2.10"]` as a uv build-dep declaration; it propagated into
the main dependency list. **It was never tied to a known incompatibility** — torch
2.10 did not exist when it was written.

It does not affect us at runtime: bergson is used via `PYTHONPATH`, never
pip-installed, so the constraint is never enforced. It only bites if you
`pip install bergson` alongside current torch.

Patch removing the caps (keeping the lower bounds, which encode the real stale
torchvision constraint) is committed on branch `remove-torch-upper-cap` in
`/mnt/ssd-2/lucia/bergson-main`. Not pushed — no `gh` on these nodes.

## Permissions fix

`lucia` was uid **1001** on iris-0 and secret-ord-0 but the shared tree is uid 1000
(`will`/`ben` own uid 1000 on those two nodes), so both were silently unable to
commit. Fixed by adding `lucia` to gid 1000 there, plus setgid on the repo tree and
`umask 002`, so files stay group-writable across the uid split. Do **not** run
`chown -R lucia` from iris-0 or secret-ord-0 — it would rewrite the tree to uid 1001
and lock out the other three nodes.

## Work division (avoid collisions)

I have taken the **batch-size axis + knob rows**, 12 banks on bellflower-0,
lucia-ord-0, secret-ord-0, allium-0:

    plan_{adam,muon}_eps1e17_16k_bs{16,32,64,128}, plan_adam_eps1e17_16k_bs512,
    plan_adam_eps1e17_16k_ep4, plan_adam_eps1e17_16k_wd0.0,
    plan_adam_eps1e17_16k_clip1.0

lotus-0 holds the token axis (4k/8k). Untaken: 32k/64k both optimizers, wd0.1.

**Disk:** /mnt/ssd-2 is at 96% (1.2 TB free) and a finished bank measures 83-93 GB.
Twelve is what fits — this is the budget the generator's `save_mode: log` was sized
for. Check `df` before claiming a 13th.

## For Lucia

1. **Hardware is still not uniform and no env fix changes that.** lotus-0 is an
   A100-SXM4-80GB; the other five nodes are A40s. If the token axis is measured on
   A100 and the batch-size axis on A40, an axis-vs-hardware confound is baked in.
   Either move the token axis onto an A40 node, or record GPU model per row and
   treat cross-axis comparisons with care.
2. The `remove-torch-upper-cap` branch needs a PR to EleutherAI/bergson.
3. torch 2.13 is well beyond anything bergson has been tested against. It imports
   and trains here, but nothing has verified numerical agreement with the 2.9/2.11
   results already recorded.
