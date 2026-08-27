# UPDATE: all shards claimed and running; query sharding CONFIRMED working

Nothing here is unclaimed any more - do not launch q0_7, it is running on
allium-0 [4,5]. Superseding the "UNCLAIMED" section below.

## Query sharding needed a second fix, and now works

Slicing only the query dataset is NOT enough. bergson checks the score matrix
against the query count and dies AFTER the first retrain with

    ValueError: scores has 20 query columns but the query dataset has 6 documents

The fix is scripts/shard_scores.py, which slices scores.bin (a flat structured
array of score_i/written_i fields) down to the shard's query range and renumbers
the fields to a dense 0..k-1. Confirmed working: all three shards are now past
the validation step and into the per-query phase, showing 0/7, 0/7 and 0/6 -
matching their slices exactly.

If you shard a filter, you must slice BOTH the query dataset and the scores.

## Live as of this update

    adam 64k     secret-ord-0 [0-3]   9/20 queries   ~3.7h
    muon 64k     allium-0 [4,5] q0_7, allium-0 [6,7] q7_14, iris-0 [6,7] q14_20
                 all in per-query phase                     ~4.4h
    4k muon      shared-ord-0 [0,1]  10/20 queries   ~25min  (replacement at
                 lr 2e-4, ms 0.9968, replacing the collapsed ms 0.9036 point)
    scoring      4000-step ekfac x2 (shared-ord), 8000-step ekfac (secret-ord),
                 MAGIC x2 (lucia-ord) - all healthy, all writing to ssd-2

---

# Proponent-filter scaling curve: state, and one shard needs a home

Goal (Lucia): a filter-delta scaling curve at bs256 with matching training runs,
5 points per optimizer (4k, 8k, 16k, 32k, 64k). Under D22 we collect proponent
filter deltas only - no new LDS, no new 100-retrain banks.

## What already exists

    N       adam ekfac filt D      muon ekfac filt D
    4k      0.01152                0.01843   <- muon point is COLLAPSED, ms 0.9036
    8k      0.02922                0.02257
    16k     0.05288                0.05039
    32k     0.05352                0.05365
    64k     RUNNING                RUNNING (3 shards)

Note the curve already looks saturated above 16k: adam moves +0.0006 from 16k to
32k. The 64k points will most likely confirm a plateau, not extend a trend.

## In flight

    adam 64k    secret-ord-0 [0-3]   4/20 retrains, 20:17 each  -> ~5.4h
                uses the completed bank as its control, so 20 retrains not 23
    muon 64k    3 query shards, 7/7/6 queries + 3 random controls each:
                  q14_20  iris-0   [6,7]   running
                  q7_14   allium-0 [6,7]   running
                  q0_7    UNPLACED - see below
    4k muon     replacement at lr 2e-4 (ms 0.9968 vs 0.9036 at the row's 4e-4).
                base chained on allium-0 [4,5] behind the 8000-step base.

## UNCLAIMED: muon 64k shard q0_7

    config: plan_muon_eps1e17_64k_bs256/filter_proponents_ekfac_q0_7.yaml
    needs:  one pair of GPUs, nproc_per_node=2 (do NOT change - world size is
            part of run identity and the other two shards use 2)
    cost:   7 queries + 3 random controls = 10 retrains, ~38 min each, ~6.3h

    cd /tmp && setsid nohup env CUDA_VISIBLE_DEVICES=<a,b> MASTER_PORT=<unique> \
      PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      BERGSON_DIST_TIMEOUT_MIN=1440 \
      PYTHONPATH=/mnt/ssd-2/lucia/bergson-filter \
      /home/lucia/envs/paper/bin/python -s -P -m bergson <cfg> > <log> 2>&1 &

Claim it under _claims/plan_muon_eps1e17_64k_bs256__filt_q0_7 first. It is the
critical path: without it the curve waits for allium-0 [4,5] to clear the 4k
pipeline, costing ~1.2h.

## Sharding note (why the query slices exist)

bergson has no query_start/query_stop. `subset_start`/`subset_stop` exist but are
read only by validate_scores (the bank path), NOT by tail_filter_retrain. The
query set is a plain dataset path though, so slicing it shards the work with no
code change - scripts/shard_filter.py does this.

Caveat that matters for the table: each shard draws its OWN 3 random controls,
because num_subsets=0 is rejected unless a bank is supplied. Serially the same 3
random models are scored against all 20 queries; sharded, each group gets its
own. Per-query deltas stay valid, the mean is unaffected, but a sharded delta is
not bit-comparable with an unsharded one.

## Do not use ssd-1 for output (D23)

Three D-state wedges so far, all writes under /mnt/ssd-1, all
`wchan = ceph_mdsc_wait_request`, all unkillable. Two of them ate a 9-hour
EK-FAC scoring each and permanently stranded a GPU. Use
PYTHONPATH=/mnt/ssd-2/lucia/bergson-filter and keep run_path on ssd-2.
