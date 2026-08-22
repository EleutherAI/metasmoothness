# bellflower-0: logit-scale implemented — please push feat/logit-scale and open the PR

From: bellflower-0. Date: 2026-08-22.
Branch: `feat/logit-scale` at `73f5fa26`, in `/mnt/ssd-1/lucia/bergson-logit-scale`
(worktree off merged main `79c08dce`). These nodes still have no GitHub
credentials — `gh` exists on shared-ord-0 but is not logged in — so this needs
your push, as with #428 and #430.

## What it does

Adds `ModelConfig.logit_scale` (default `1.0`, a no-op) multiplying the language
model head output. This is the logit-scale ablation axis, which had no
implementation in bergson at all and was blocking two planned rows.

A forward hook on the output embedding, not a wrapper module, so the model stays
a plain `PreTrainedModel` and FSDP, PEFT and `save_pretrained` keep seeing the
architecture they expect. The hook is deliberately not persisted by
`save_pretrained` — the scale belongs to the run config, and every load path
re-applies it from there.

## The part worth reviewing

It is applied at **both** load sites:

- `setup_model_and_peft` — training and scoring;
- `validate._load_banked_model` — the leave-k-out bank.

The bank bypasses `setup_model_and_peft` entirely. Scaling only the training
path would have measured the bank's query losses at a different temperature than
the models were trained at — silently corrupting the attribution ground truth for
every scaled row, while the training loss still looked correct. That is the same
shape as the `skip_validation` and per-query-width bugs: wrong in the eval path
only, and invisible from the training side.

Ten tests: the default registering no hook at all, logits scaled, gradients
scaled (so training sees it, not just eval), persistence across repeated calls,
the softmax actually flattening, and a clear error when a non-causal-LM is given
a scale.

Also `d849470` in the metasmoothness repo: the generator now emits `logit_scale`
from the row, which it previously did not.

Both rows are already running against the branch worktree on shared-ord-0 and
show `logit_scale: 0.25` / `0.5` in their configs, training clean.

## On the "failing tests" — there are none

Worth recording, because it looks alarming and is not. Running

    tests/test_per_query_magic.py tests/test_multi_query_validate.py
    tests/test_bank_loss_cache.py

on bellflower-0 gives **5 failed, 17 passed** — identically on merged main and on
my branch, so nothing to do with the change. `test_bank_loss_cache.py` alone
passes 6/6.

The failures are `torch.OutOfMemoryError` and `AcceleratorError: CUDA error: out
of memory`: the suite grabs GPU 0, and on these nodes GPU 0 is saturated by
production runs (one process holding 45.97 GiB of 47.54). On an idle node the
same four files give **32 passed**.

So: run the suite on a node with free GPUs, or pin `CUDA_VISIBLE_DEVICES` to one.
If you have seen sporadic failures on lotus-0 while its own rows were running,
this is very likely the same cause rather than a real regression.

## Fleet

18 rows now in flight. New this cycle: the 16k anchor pair, `muon_8k`, the 32k
pair, the 64k pair, and the two logit-scale rows. Three additional nodes brought
up with the pinned environment and leak-checked — shared-ord-0, louis-ord-0,
soar-ord-0 (the last needed `lucia` added to gid 1000; it was created as uid
1003, the same mismatch iris-0 and secret-ord-0 had).
