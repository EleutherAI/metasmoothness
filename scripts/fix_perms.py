"""Make run artifacts readable across the fleet's two uids.

`lucia` is uid 1001 on maria-1/iris-0/secret-ord-0 and uid 1000 on the other
seven, and CephFS stores the numeric uid. A file written 0600 by one side is
invisible to the other -- it fails instantly with FileNotFoundError while `ls`
shows it, which is the confusing part.

umask 022 is NOT sufficient. huggingface/safetensors saves through a temp file,
and Python creates those 0600 regardless of umask, so the renamed result is 0600
even under a permissive umask. That is why models kept coming out unreadable
after the umask fix went in. The only reliable repair is chmod after the fact.

Only the owning uid can chmod, so run this from a node of EACH uid.

    python scripts/fix_perms.py [--root DIR] [--dry-run]
"""
import argparse
import os
import stat
import sys

DEFAULT_ROOTS = [
    "/mnt/ssd-2/lucia/paper_runs",
    "/mnt/ssd-1/lucia/paper_runs",
    "/mnt/ssd-2/lucia/datasets_local",
]

ap = argparse.ArgumentParser()
ap.add_argument("--root", action="append", default=None)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

me = os.getuid()
roots = a.root or DEFAULT_ROOTS
fixed = skipped = 0

for root in roots:
    if not os.path.isdir(root):
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode):
                continue
            if st.st_uid != me:
                skipped += 1
                continue
            mode = stat.S_IMODE(st.st_mode)
            want = mode | 0o044 | (0o011 if stat.S_ISDIR(st.st_mode) else 0)
            if want == mode:
                continue
            if a.dry_run:
                print(f"  would chmod {oct(mode)} -> {oct(want)}  {p}")
            else:
                try:
                    os.chmod(p, want)
                except OSError as e:
                    print(f"  FAILED {p}: {e}", file=sys.stderr)
                    continue
            fixed += 1

verb = "would fix" if a.dry_run else "fixed"
print(f"{verb} {fixed} paths owned by uid {me}; {skipped} owned by the other uid "
      f"(run this from a node of that uid too)")
