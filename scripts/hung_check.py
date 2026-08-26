"""Report live bergson jobs whose output has stopped advancing.

check_runs.py and filter_health.py both ask "does someone own this row and is the
claim fresh". Neither can see a hang, because a hung process is alive and a hang
produces no output at all -- and no output was being scored as no problem.

Two failures made that concrete on 2026-08-26:

  * thirteen london tuning runs at 32k/64k sat hung for ten hours, holding
    fourteen GPUs, with 130 threads (39 in futex_wait) and zero-byte logs
  * their logs go to /tmp/tune_<name>.log rather than the run directory, so every
    sweep reported "no log" and moved on

So this looks at the process table first and the filesystem second: for each live
bergson process it finds the newest byte that process could have written --
anywhere it might write -- and reports the ones that have gone quiet.

Node-local: it reads this node's /proc, so run it on each node.

    python hung_check.py [--minutes 45] [--kill-hint]
"""
import argparse
import glob
import os
import re
import time

AP = argparse.ArgumentParser()
AP.add_argument("--minutes", type=int, default=45,
                help="quiet for longer than this counts as hung")
AP.add_argument("--kill-hint", action="store_true",
                help="print the diagnostic capture command for each hung job")
args = AP.parse_args()

NOW = time.time()
THRESH = args.minutes * 60


def read(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def newest_mtime(paths):
    best = 0.0
    for p in paths:
        try:
            best = max(best, os.stat(p).st_mtime)
        except OSError:
            pass
    return best


def candidates(cfg_path):
    """Every file this job might be writing."""
    out = []
    run_dir = os.path.dirname(cfg_path)
    stem = os.path.splitext(os.path.basename(cfg_path))[0]
    out += glob.glob(os.path.join(run_dir, "*.log"))
    out += glob.glob(os.path.join(run_dir, "*.csv"))
    out += glob.glob(os.path.join(run_dir, "**", "*.safetensors"), recursive=True)
    out += glob.glob(os.path.join(run_dir, "**", "*.json"), recursive=True)
    # The tuning launcher writes here, not to the run directory. This is the
    # lookup that was missing when thirteen hung runs read as healthy.
    out += glob.glob("/tmp/%s*.log" % stem)
    out += glob.glob("/tmp/*%s*.log" % stem)
    return out


rows = []
for d in glob.glob("/proc/[0-9]*"):
    pid = os.path.basename(d)
    cmd = read(os.path.join(d, "cmdline")).replace("\0", " ")
    if "bergson" not in cmd:
        continue
    status = read(os.path.join(d, "status"))
    state = re.search(r"^State:\s+(\S)", status, re.M)
    if not state or state.group(1) == "Z":
        continue
    m = re.search(r"(\S+\.yaml)", cmd)
    if not m:
        continue
    cfg = m.group(1)
    devs = ""
    env = read(os.path.join(d, "environ"))
    dm = re.search(r"CUDA_VISIBLE_DEVICES=([^\0\n]*)", env)
    if dm:
        devs = dm.group(1)
    files = candidates(cfg)
    mt = newest_mtime(files + [cfg])
    quiet = (NOW - mt) / 60.0 if mt else float("inf")
    threads = re.search(r"^Threads:\s+(\d+)", status, re.M)
    wchan = read(os.path.join(d, "wchan")).strip()
    rows.append({
        "pid": pid, "cfg": cfg, "devs": devs, "quiet": quiet,
        "threads": threads.group(1) if threads else "?", "wchan": wchan,
        "nfiles": len(files),
    })

# One entry per config: a run is many ranks, and reporting each rank separately
# turns one hang into eight lines that look like eight problems.
best = {}
for r in rows:
    k = r["cfg"]
    if k not in best or r["quiet"] < best[k]["quiet"]:
        best[k] = r

hung = sorted((r for r in best.values() if r["quiet"] > args.minutes),
              key=lambda r: -r["quiet"])
live = [r for r in best.values() if r["quiet"] <= args.minutes]

host = os.uname().nodename
print("%s: %d live bergson run(s), %d quiet > %dm"
      % (host, len(best), len(hung), args.minutes))
if hung:
    print("\n%-8s %-6s %5s %8s %-14s %s"
          % ("pid", "gpu", "thr", "quiet", "wchan", "config"))
    for r in hung:
        q = "%.0fm" % r["quiet"] if r["quiet"] != float("inf") else "never"
        print("%-8s %-6s %5s %8s %-14s %s"
              % (r["pid"], r["devs"] or "-", r["threads"], q,
                 r["wchan"] or "-", r["cfg"]))
        if r["nfiles"] == 0:
            print("         (no output file found at all -- it may never have started)")
    if args.kill_hint:
        # D19: capture before killing. A kill with no capture destroys the only
        # evidence and the next occurrence starts from zero.
        print("\nD19 -- capture BEFORE killing:")
        for r in hung:
            print("  cat /proc/%s/wchan; "
                  "for t in /proc/%s/task/*/wchan; do cat $t; echo; done | sort | uniq -c; "
                  "ls -l /proc/%s/fd | head; "
                  "py-spy dump --pid %s" % (r["pid"], r["pid"], r["pid"], r["pid"]))
for r in sorted(live, key=lambda r: r["quiet"]):
    print("  ok  %5.0fm  %s" % (r["quiet"], os.path.basename(r["cfg"])))
raise SystemExit(1 if hung else 0)
