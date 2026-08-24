# The HF model cache is node-local — copy it, don't re-download

Slicing the gpt2-medium bank onto lotus-0 failed immediately:

    OSError: We couldn't connect to 'https://huggingface.co' to load ...

Not a network problem. `HF_HUB_OFFLINE=1` is set in both launchers now (it
removes a class of open-ended Hub stalls), and gpt2-medium simply was not in
lotus-0's cache. `~/.cache/huggingface` is **per node**, even though
`/home/lucia/envs/paper` is the shared pinned venv:

    shivam2-0   models--gpt2-medium   1.5G
    lotus-0     NOT CACHED

## What to do

Copy the cache from a node that has it, through the shared volume:

    # on the node that has it
    cd ~/.cache/huggingface/hub && tar czf /mnt/ssd-2/lucia/_hfshare/<model>.tgz models--<model>
    # on the node that needs it
    mkdir -p ~/.cache/huggingface/hub && cd ~/.cache/huggingface/hub
    tar xzf /mnt/ssd-2/lucia/_hfshare/<model>.tgz

Prefer this over dropping `HF_HUB_OFFLINE`. A fresh download can resolve to a
different revision than the one the bank's main run used, and for a bank being
sharded across nodes every slice must load byte-identical weights — the same
reason D17 forbids mixing GPU types within one bank. Copying the cache
guarantees the same snapshot hash; downloading only probably does.

`/mnt/ssd-2/lucia/_hfshare/gpt2-medium.tgz` (1.3 GB) is there now, so any A100
node can join the gpt2-medium bank without touching the network.

## Which models are where

gpt2 is cached widely. gpt2-medium was only on shivam2-0 before this, and is now
also on lotus-0. gpt2-large is nowhere, which is one more reason the gpt2-large
rows are not runnable as they stand (they are also non-target under D11).
