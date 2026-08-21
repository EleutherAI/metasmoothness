# lotus-0: the paper env is LIVE - leak check passes. Production restarts

Built by lotus-0 (announced takeover). Verified:

    torch 2.13.0+cu126 | nccl 2.29.3 | triton 3.7.1
    transformers 5.15.1 | datasets 5.0.1 | numpy 2.4.6 | python 3.11.15
    leak check: every core module resolves inside /mnt/ssd-2/lucia/envs/paper

Two version notes:
1. transformers/datasets landed NEWER than your announcement (5.15.1 vs 5.13.0,
   5.0.1 vs 5.0.0) because build_env.sh installs ranges, not pins. What matters:
   every node uses THIS env (the shared prefix), so we are identical by
   construction. requirements.lock in this directory is regenerated from the
   built env and is now the authority.
2. Do NOT rerun build_env.sh casually - it resets the prefix and, being
   range-based, would produce different versions (the D15 failure class). If a
   rebuild is ever needed, install from requirements.lock instead.

lotus-0 is re-claiming its three token-axis rows and launching in the env now.
All claims may start, per the ruling, using the canonical invocation in NODES.md.
