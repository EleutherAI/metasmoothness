# Publishing a bank, and only then releasing its disk

Five banks are now public and their local `retrained/` directories are gone.
This is the procedure, and the reasons each guard exists.

## The rule

A bank leaves the node when **everything derived from it exists** (LDS for both
scorers, and the tail-filter deltas) **and it is definitely public**. Definitely
is the operative word: `upload_folder` that half-finishes leaves a repo that
looks complete in the web UI, and at that moment the local copy is the only
other one.

    python scripts/publish_bank.py <run_id>                 # publish + verify
    python scripts/bank_card.py <run_id>                    # dataset card
    python scripts/publish_bank.py <run_id> --delete-local  # re-verify, then release

`--delete-local` re-runs the scoreability check and the file diff before it
deletes anything, so a stale success from an earlier log cannot authorise a
deletion.

## What is checked before the disk is touched

1. **100/100 models.** A partial bank is not publishable.
2. **It scores cleanly.** `magic_lds.py` asserts every subset appears exactly
   once. A bank can have 100 model directories and still be unscoreable -- the
   8k adam bank had subsets 72-86 duplicated across a shard boundary, which a
   file count passes happily. Publishing an unscoreable bank is worse than not
   publishing, because a consumer cannot tell from the listing.
3. **Every local file meant for upload is in the repo.** Compared by name, with
   the same ignore list used for the upload. Any miss aborts before deletion.

The five checks reproduced their recorded numbers exactly, which is a useful
independent confirmation that the published artifact is the measured one:

    plan_adam_eps1e17_4k_bs256       0.9295 [0.9195, 0.9381]
    plan_adam_eps1e17_8k_bs256       0.9163 [0.9013, 0.9280]
    plan_muon_eps1e17_8k_bs256       0.7712 [0.7477, 0.7904]
    sm_adamw_eps1e17_16k_bs256       0.9411 [0.9326, 0.9477]
    plan_adam_eps1e17_16k_scale0.25  0.0456 [0.0038, 0.0861]

## What is deleted, and what is not

Only `retrained/` -- the part that is published. `checkpoints/` stays: it is the
base training trajectory, it is deliberately NOT uploaded, and it is what a
MAGIC replay needs. `validation.csv`, `subsets.json` and the
`filter_proponents_*` results stay too; they are small and they are the reason
the bank is interpretable.

## Where they are

Collection: `EleutherAI/data-attribution-6a8d013ccc372b7fd6abce3e`
Repos: `EleutherAI/metasmoothness-bank-<run_id>`, public, dataset type.

The card leads with the fact that the repo contains 100 retrained models rather
than scores, since that is the reusable asset and it is not obvious from a file
listing of 500 safetensors.

## Not yet publishable

A bank still needs both filter arms before it qualifies. As of writing:
`sm_muon` and `muon 4k` have MAGIC deltas but not EK-FAC; the bs16/bs32/bs64/
bs128/wd/clip/scale0.5 rows have LDS for both scorers but no filter deltas yet.
Those runs are what the filter queues are working through.
