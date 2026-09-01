# 512k Filter Sharding Notes

Context: 512k AdamW EK-FAC/BM25 filter runs were originally launched as
two-query shards named `filter_{proponents,top40}_{ekfac,bm25}_qA_B`. A full
main-figure point needs 20 query rows merged into the canonical
`filter_*_{ekfac,bm25}/filter_summary.csv`.

Safe non-duplicating speedup:

1. Keep the canonical q-pair jobs running.
2. Materialize one-column score dirs from the complete full score matrix:
   `scores_qi_i+1` for EK-FAC and `bm25_scores_qi_i+1` for BM25.
3. Use the existing single-query datasets `query_20_qi_i+1.hf`.
4. Generate single-query validate configs whose `query`, `scores`, and
   `run_path` point at the q-single paths.
5. Race only the second query in each q-pair first: q1, q3, ..., q19. The
   canonical q-pair jobs train qA before qA+1, so this avoids duplicating work
   already underway.

Observed pod behavior on 2026-08-31:

- `shared-ord-0` eventually started q-single filter jobs and runs them at normal
  speed once oversubscription is cleared.
- `allium-0`, `bellflower-0`, `iris-0`, `secret-ord-0`, and `lucia-ord-0`
  accepted detached launches but the Python child processes entered `D` state
  before logging or allocating GPU memory. Treat these as unreliable for this
  path unless a foreground smoke test proves otherwise.
- If a wrapper is killed, check for orphaned `multiprocessing.spawn` children.
  They can continue holding GPU memory even when the yaml-bearing parent is
  gone. Kill only orphaned children (`PPID=1`) after mapping the live job
  parents.

Current useful extra 512k top-1 EK-FAC race jobs on shared-ord:

- `filter_proponents_ekfac_q9_10`
- `filter_proponents_ekfac_q11_12`
- `filter_proponents_ekfac_q17_18`
- `filter_proponents_ekfac_q19_20_shared45`

When single-query suffixed dirs finish, either copy/rename them into exact
`..._qA_B` dirs before `scripts/merge_filter_shards.py`, or extend the merge
regex to accept a suffix after `qA_B` and still use `A` as the global query
offset.
