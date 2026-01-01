
## 2026-08-26: the NCCL timeout knob is not the one that sounds right

The 64k_bs32 rows abort with "Watchdog caught collective operation timeout",
checkTimeout at ProcessGroupNCCL.cpp:733, right after "Collecting gradients: 100%".

TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC does NOT fix it -- that governs the monitor
thread. The collective timeout is an init_process_group argument, 10 minutes by
default for NCCL, with no environment override. Setting the heartbeat variable to
7200 changed nothing and both rows failed again in exactly the same place.

Use TORCH_NCCL_ASYNC_ERROR_HANDLING=0, which disables the watchdog teardown so a
slow collective blocks instead of aborting the process.

32k scores fine and 64k has failed twice, with the abort at the rank
synchronisation after the gradient collection. The 64k trajectory is twice as
long, so a rank imbalance at that barrier is the standing suspect.
