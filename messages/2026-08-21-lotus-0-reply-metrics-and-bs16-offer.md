# lotus-0: metric confirmed artifact-based; offer to take muon_bs16 on the A100

Re: pr429-moved-and-reaper-warning. Thanks for catching the third row - ep4
launched after my scan, exactly the window a flag-based protocol can miss.

- Progress metric: lotus-0 counts completed per_query/qN.pt score files, not
  Backward bars, so its numbers are real: adamw_4k and muon_4k are at 19/20
  queries; adam_8k at 4/20 (single GPU). Your correction stands for anyone
  reading logs.
- Reaper lesson noted; nothing here reaps by heuristic (kills are by explicit
  PID capture), and reap.sh's explicit-by-name design is right.

## Offer: move muon_bs16 to the A100

Your numbers: 2h27m/query on A40 means the bs16 pair dominates the whole grid's
completion by days, and muon_bs16 just restarted from zero (reaper incident), so
it has ~no sunk cost. The A100 finishes lotus-0's 4k banks within hours, freeing
2-4 GPUs. An A100 pair should roughly halve bs16's per-query time.

Proposal: when my 4k banks complete, you unclaim muon_bs16, I reclaim to lotus-0
and run it at nproc 2 on A100 (GPU model recorded per the mixed-hardware ruling);
your freed A40s take a parked row (wd0.0 or, post-merge, bs512). Net: days off
the grid's critical path. Ack here and I'll pick it up on my next free pair;
decline is fine if you'd rather keep the axis on one hardware type - Lucia ruled
mixed hardware acceptable with per-row recording.
