#!/usr/bin/env python3
"""Mirror every small text artifact of the experiment runs into the repo under runs/ (Lucia 2026-09-06: "ensure all
the csvs, logs, yamls in this repo are committed" after 33 finished Qwen banks were deleted from scratch disk).

    python scripts/mirror_run_text.py            # -> runs/<run>/<relative path>, then commit

Copies *.yaml, *.csv, *.json under /mnt/ssd-2/lucia/paper_runs/experiments (depth <= 4, < 4 MB) preserving the path
relative to the experiments root, skipping checkpoints/log_history.json (per-step training curves, tens of MB per run).
Training logs (/mnt/ssd-2/lucia/paper_runs/_logs, up to 60 MB each) are not mirrored here; they go to cold storage
gzipped (/mnt/cold-1/lucia/logs_<date>/). Idempotent: unchanged files are skipped by size+mtime.
"""
import os, shutil, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = "/mnt/ssd-2/lucia/paper_runs/experiments"; DST = os.path.join(ROOT, "runs")
EXT = (".yaml", ".yml", ".csv", ".json"); MAX = 4 * 1024 * 1024; SKIP_NAMES = {"log_history.json"}
t0 = time.time(); copied = skipped = 0
for dirpath, dirnames, filenames in os.walk(SRC):
    depth = dirpath[len(SRC):].count("/")
    if depth >= 4:
        dirnames[:] = []
    dirnames[:] = [d for d in dirnames if not d.endswith((".ckpt", ".hf")) and d not in ("kfac_query", "query", "query.part", "hessian", "traces")]
    for f in filenames:
        if not f.endswith(EXT) or f in SKIP_NAMES: continue
        s = os.path.join(dirpath, f)
        try:
            st = os.stat(s)
        except OSError:
            continue
        if st.st_size > MAX: continue
        d = os.path.join(DST, os.path.relpath(s, SRC))
        try:
            dst_st = os.stat(d)
            if dst_st.st_size == st.st_size and int(dst_st.st_mtime) >= int(st.st_mtime):
                skipped += 1; continue
        except OSError:
            pass
        os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(s, d); copied += 1
print(f"runs/: copied {copied}, unchanged {skipped}, {time.time()-t0:.0f}s")
