# Ruling relay: mixed hardware is acceptable

From: lotus-0 relaying Lucia, 2026-08-21. Re: env-standardisation point 1.

Mixed hardware across the fleet is fine; do not reshuffle axes across nodes for GPU
uniformity. Record the GPU model with every claim (NODES.md updated) so cross-axis
comparisons stay auditable. lotus-0 keeps the token axis (A100) and will claim
32k/64k next; the batch-size + knob division stands. lotus must and will match the
paper env's settings exactly once the env boots (still incomplete at last probe -
`import torch` fails; ping here when the leak check passes).
