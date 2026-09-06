# Experiment tiers — standing backlog

Policies (Lucia, 2026-09-04): holding pairs idle for an upcoming 8-GPU window
is acceptable. When another user's job appears on a node, do not use that node
until it clears — guillaume has priority. Tier 2 is preemptible: fill idle
pairs from the top; evict Tier 2 whenever Tier 1 needs the slot. Tier 3 is
backlog — not running, resume when prioritized.

## Tier 1 (in priority order)

NOTE (2026-09-04): qwen queries need NO held-out variant -- verified 0 overlap
with train_8k/32k/64k_qwen. The qwen chain is nested by construction (query =
chunks [0,20), train = chunks [20,20+N)), so queries are disjoint from every
train set at any N. In-distribution qwen filter results ARE valid held-out
measurements. (The gpt2 contamination came from re-chunking query+train from the
same source; the qwen chain avoids this.)


1. top-40 MAGIC 4k-64k (filter_method_appendix)
2. held-out-query EK-FAC 1% + top-40 at 64k-512k (held-out figure-1 variant;
   keep in-distribution results too -- separate plots)
3. qwen15b EK-FAC 1% + top-40, rungs 4k-64k (qwen figure-1 replication)
4. qwen heldout-4k eval against heldout_4k_qwen_v2 (qwen heldout trend)

## Tier 2 (preemptible; take from the top)

HELD-OUT (OOD-query) RE-RUNS for other gpt-2 figures' 64k+ points (Lucia
2026-09-04): any gpt-2 filter point at N>=64k used the contaminated query_20;
re-run with query_20_heldout for clean numbers. Do NOT touch <64k points (queries
verified disjoint there). Priority within this:
  1. BM25 scaling: held-out BM25 filters at 64k-512k. Scores READY
     (bm25_scores_heldout computed for 64k/128k/256k/512k); just gen+launch
     held-out BM25 filters (bm25_scores_heldout + query_20_heldout).
  2. Muon scaling (filter_muon_appendix): muon rows at 64k+ need held-out EK-FAC
     scores (reuse each muon row's Hessian, re-score query_20_heldout) then
     held-out filters. Same pipeline as the adamw held-out campaign.
(Batch-size appendix struck: it is a 16k-doc sweep, no 64k+ points, so no
contamination -- no held-out re-run needed.)
Slice held-out scores with scripts/slicefix.py (aligned; self-checks).


1. muon 512k row: correction pass + 3-random bank + 1% filter (filter_muon_appendix)
2. jina-v5 baseline: 16k filters, then gpt2 128k/256k scores + filters
3. BM25 held-out: scores 64k-512k (CPU), then filters
4. jina held-out variants at 64k+
5. (add more here)

## Tier 3 (backlog, not running)

- qwen15b 128k rung: EK-FAC pipeline (fit was ~80% of correction when cancelled;
  restarts from scratch -- hessian dir must be cleared first, see RESUME.md)
- qwen15b 256k rung: same state as 128k
- 64k+ LDS validation runs (needs bank re-download from HF)

## Done

- ms probes: 256k adamw 0.947, 256k muon 0.973, 512k adamw 0.487
- seed-43 replication of the 16k figure point: 0.0525 vs 0.0529
- top-40 MAGIC 4k (0.0249), 8k (0.0489)

## Dropped by decision

- MAGIC 1% beyond 64k; all MAGIC at 512k
- muon/BM25 MAGIC top-40 outside the 4k-64k scope
