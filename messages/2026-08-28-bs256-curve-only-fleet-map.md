# 2026-08-28 01:15 — bs256 curve is the only priority; here is who holds what

All bs32 work is stopped (see the preservation note at the bottom). Every GPU on
the allowlist is now either on the bs256 scaling curve or free. If you are picking
up work, read this before launching anything.

## Live right now

| Node | GPUs | Job | Ends |
|---|---|---|---|
| lucia-ord-0 | 0-7 | **EK-FAC scoring, adam 128k bs256** | ~03:00 |
| lotus-0 | 0-7 | 256k bs256 lr sweep, 4 points | ~02:47 |
| shared-ord-0 | 0,1 / 4,5 / 6,7 | 512k bs256 lr sweep, 3 of 4 points | ~06:05 |
| bellflower-0 | 0,1 | 512k bs256 lr sweep, 4th point | ~06:05 |
| allium-0 | 0,1 | muon 128k bs256 base | ~02:10 |
| allium-0 | 2,3 | 32k top-40 filter, 15/20 | ~02:45 |
| secret-ord-0 | 0,1 / 6,7 | 64k top-40 filter shards q0_7, q7_14 | ~02:40 |
| bellflower-0 | 4,5 | 64k top-40 filter shard q14_20 | ~02:40 |
| shared-ord-0 | 2,3 · secret-ord-0 2,3 · bellflower-0 6,7 | 64k MAGIC filter shards | ~03:40 |

Free: allium 4-7, secret-ord 4,5, bellflower 2,3, iris 3-7. **iris 0,1,2 are another
user's — do not touch.** marisa-0 and shivam2-0 remain permanently off.

## Reserve this capacity

The adam 128k proponent filter launches the moment scoring finishes (~03:00) and
wants **as many pairs as it can get**. Do not start anything long on the free pairs
before then.

## Two launcher traps that cost time tonight

1. **bergson clears `run_path` at startup and unlinks any log written into it.**
   The run keeps writing to a dead inode, so the job looks silent and a `ls` shows
   no log — while it is in fact training. Write logs to
   `paper_runs/_logs/` or `paper_runs/experiments/_logs/`, never inside the run dir.
2. **Launch with `setsid`.** Plain `nohup ... &` under `kubectl exec` can look like
   it died when the session tears down. It does not always die — I launched a set
   of four believing they had failed, relaunched, and had two full sets training on
   the same GPUs and the same `run_path` for eight minutes.

Related: `ps` right-aligns PIDs, so `${line%% *}` yields an empty string for a
6-digit PID and every `kill` silently no-ops. Parse with `read -r pid rest`. A
kill sweep that reports success is not evidence; re-check with `nvidia-smi`.

## Preserved, resumable

`/mnt/ssd-2/lucia/paper_runs/_preserved/2026-08-28/` holds completed per-query rows
for the stopped bs32 filters (adam_128k q7_14=7, q14_20=6; muon_128k q7_14=2). The
CSV writer opens mode `"w"`, so a naive restart truncates these — merge them back
with `scripts/merge_filter_shards.py` rather than re-running the queries.

## 1M is blocked on data, not GPUs

`notes/n1m_blocker.md` still holds: `train_scratch_512k.hf` is exactly 512,000 docs
and all of them are already in `train_512k.hf`. The Hub copy of
`EleutherAI/bergson-smollm2-scaling` publishes only up to `train_256k`, so 1M needs
new documents pulled from the source corpus and packed under the same nesting rule.
Network from the pods is fine (the Hub API answers); it is only git-over-SSH that
fails, which is why pushes still go through the bundle bridge.
