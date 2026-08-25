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
