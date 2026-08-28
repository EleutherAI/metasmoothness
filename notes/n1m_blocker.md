# N=1M needs new data pulled; the local pool is exhausted at 512k

The token-scaling chain is nested by construction:

    train_512k = train_256k ++ fill from train_scratch_512k

and `train_scratch_512k.hf` holds exactly 512,000 documents, all of which are
already in `train_512k.hf`. Headroom for a longer chain is **zero**. So the
registered 512k rung is the end of the road with what is on disk, and the N=1M
target cannot be reached by rearranging existing files.

## What building 1M requires

More documents from the source corpus (smollm2), pulled and packed the same way,
then appended under the existing rule so that `train_1M` keeps `train_512k` as a
byte-identical prefix. The nesting is what makes the token axis a clean
comparison -- every smaller N is a prefix of every larger one -- so a 1M built by
re-sampling from scratch would not be comparable to the rows already measured.

The exclusion pool must stay clean: `heldout_4k`, `query_20` and `query_50` are
disjoint from every train_* set, and the 512k build verified measured overlap 0.
Any 1M build has to re-verify that, not assume it.

## Cost note before anyone starts one

At bs32 and 2 epochs, 1M documents is 62,500 optimiser steps. The 512k rung is
32,000 steps and its lr sweep alone is six runs. A bank at 1M is not plausible --
100 retrains at 62,500 steps each -- so a 1M row would be ms-only, or ms plus a
proponent-filter delta via `gen_filter --no-bank`, which is exactly the regime
that flag was added for.

Network is available from the pods (huggingface.co returns 200), so the pull
itself is not blocked.

## 2026-08-28: what a 1M build would need, and the piece that is missing

Established tonight while planning the path to 1M at bs256:

* **Schema is known.** Every `train_*.hf` row is `input_ids` (`List(int32)`, exactly
  512 tokens) plus a `length` field. So docs are fixed 512-token packed sequences,
  not variable-length text.
* **The Hub copy does not help.** `EleutherAI/bergson-smollm2-scaling` publishes
  `train_4k` through `train_256k`, `heldout_4k`, `query_20` and `query_50` — nothing
  at 512k, and no scratch pool. Its README is frontmatter only, with no prose
  describing how the corpus was packed.
* **The scratch pool's builder is gone.** `train_scratch_512k.hf` exists on disk but
  nothing in `metasmoothness/` or `/mnt/ssd-1/lucia/bergson-damping` references
  `scratch_512k`, and neither tree names a `HuggingFaceTB/*` source or a smollm
  corpus subset. The packing recipe — which smollm2 subset, in what order, with
  which tokenizer — is not recorded anywhere I can find.

**This is the actual blocker, and it is not GPU time.** The nesting rule is what
makes the token axis a clean comparison: every smaller N is a byte-identical prefix
of every larger one. A 1M built by pulling from a *different* smollm2 subset, or
packing in a different order, would still satisfy the prefix check — `train_512k`
would remain a prefix — while quietly changing the distribution of the appended
half. The curve would then confound "more data" with "different data" at exactly
the point where the result is most load-bearing.

So a 1M build should not start by guessing the source. It needs either the original
packing script recovered, or an explicit decision to define the 1M tail from a named
subset and to record that the tail is drawn differently from the head.

Cost note: the earlier estimate in this file was for bs32 (62,500 steps). At bs256
and 2 epochs, 1M is 8,000 steps — about 9.9 h per training run on an A40 pair. A
proponent filter is 23 of those, so ~193 pair-hours, plus EK-FAC scoring, which the
128k measurement shows is far more expensive than previously assumed.
