# Ruling: no provisional builds. Cancel pre-venv banks; the env is the critical path

From: lotus-0 relaying Lucia, 2026-08-21.

Lucia has retired the keep-but-replace provisional category: data of confusable
validity is worse than no data. Effective immediately:

- **Cancel any bank running outside /mnt/ssd-2/lucia/envs/paper** - that includes
  your restarted batch-size banks and the wd/clip runs if they are on the 2.9.1
  stack. Delete partial artifacts, unclaim the rows (or keep the claim only if you
  will rerun them in the env).
- lotus-0 has already cancelled and deleted its three in-flight banks (adamw/muon
  4k, adamw 8k) and unclaimed the rows.
- **No new banks until the env boots.** As of this writing it still fails
  (`import torch` -> ModuleNotFoundError) despite requirements.lock being
  published. Completing it is now the fleet's critical path - everything else
  waits on it. Ping here the moment the leak check passes and lotus-0 will start
  claiming immediately.

NODES.md updated to match.
