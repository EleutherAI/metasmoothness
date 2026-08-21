# lotus-0: environment + generator state for in-flight coordination

To: bellflower-0 (and any node joining). From: lotus-0, 2026-08-21.

## Provisional runs in flight on lotus-0 (pre-venv)

Three banks are building outside the pinned venv (it did not exist at launch):
`plan_adam_eps1e17_4k_bs256`, `plan_muon_eps1e17_4k_bs256`, `plan_adam_eps1e17_8k_bs256`.
Code: bergson main `3c66bb51` via worktree `/mnt/ssd-1/lucia/bergson-main-paper`.
Env fingerprint: torch 2.11.0+cu126, nccl 2.28.9 (runtime-verified), datasets 5.0.0,
numpy 2.4.6, transformers 5.1.0, python 3.11 (/opt/conda + ~/.local overlay).
Per Lucia these are keep-but-replace: usable now, superseded by pinned-venv reruns.

## Venv construction requests

When you pin the venv, please pin explicitly (not transitively) the four packages
D15 identified as numerics-relevant drift since the stored banks: nvidia-nccl-cu12,
datasets, numpy, triton — plus torch and transformers. Note torch 2.11.0+cu126
hard-requires nccl 2.28.9 symbols (2.26.x will not import). Publish the venv as a
requirements lock + creation command in this directory; lotus-0 will adopt for all
subsequent runs and mark the three banks above for rerun.

## Generator fixes you must have (both committed)

- `8ed894b` — experiment banks use save_mode: log (fits 12 banks on ssd-2).
- `58733df` — save_optimizer_state is AdamW-only; muon rows skip it (the saver
  raises on muon state). Your bs16 mirror (3b46460) post-dates both; regenerate
  only if you generated configs before them.

## Execution rules that bite silently

- Never run with a bergson checkout as cwd; `cd /tmp && python -P -m bergson ...`
  (cwd shadows PYTHONPATH; this voided a day of gate experiments here).
- Record nproc with your claim; world size is measured to change training bits.
- A `fix/nccl-metadata` branch on EleutherAI/bergson makes every run self-record
  its runtime NCCL version; merging it before the venv era starts would let the
  venv policy be verified from run configs rather than trusted.
