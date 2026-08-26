"""Run bergson with a self-scheduled traceback dump.

py-spy needs ptrace and the container has no CAP_SYS_PTRACE, as root or otherwise.
SIGABRT with PYTHONFAULTHANDLER=1 produced nothing either. But faulthandler can
schedule its own dump from inside the process, which needs no external
permission at all:

    faulthandler.dump_traceback_later(N, repeat=True, exit=False)

fires every N seconds on a dedicated thread and writes every thread's Python
stack to stderr. That is exactly what is needed for a hang that produces no
output and cannot be attached to.

    python hang_probe.py <config.yaml> [interval_seconds]
"""
import faulthandler
import runpy
import sys

cfg = sys.argv[1]
interval = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0

faulthandler.enable(file=sys.stderr, all_threads=True)
faulthandler.dump_traceback_later(interval, repeat=True, exit=False, file=sys.stderr)

print("[probe] dumping all thread stacks every %.0fs" % interval, flush=True)
sys.argv = ["bergson", cfg]
runpy.run_module("bergson", run_name="__main__", alter_sys=True)
