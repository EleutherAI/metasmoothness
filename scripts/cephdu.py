#!/usr/bin/env python3
"""Map CephFS disk usage via the ceph.dir.rbytes xattr.

du walks the tree and stats every file, which on this filesystem is one network
round trip per cache miss. CephFS already maintains a recursive byte count per
directory, so one getxattr per directory replaces the whole walk.

    python cephdu.py <dir> [<dir>...]        # children of each, largest first
    python cephdu.py --depth 2 <dir>         # recurse two levels
"""
import argparse
import os
import sys


def rbytes(path):
    """Recursive byte count for a directory, or None if unavailable."""
    try:
        return int(os.getxattr(path, "ceph.dir.rbytes"))
    except OSError:
        return None


def human(n):
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.1f}{unit}" if unit not in "BK" else f"{n:.0f}{unit}"
        n /= 1024


def mtime(path):
    try:
        import datetime
        return datetime.date.fromtimestamp(os.stat(path).st_mtime).isoformat()
    except OSError:
        return "?"


def walk(path, depth, rows):
    try:
        entries = sorted(os.scandir(path), key=lambda e: e.name)
    except OSError:
        return
    for e in entries:
        if not e.is_dir(follow_symlinks=False):
            continue
        n = rbytes(e.path)
        if n is None:
            continue
        rows.append((n, e.path, mtime(e.path)))
        if depth > 1:
            walk(e.path, depth - 1, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    rows = []
    for d in a.dirs:
        total = rbytes(d)
        if total is None:
            print(f"{d}: no ceph.dir.rbytes (not CephFS?)", file=sys.stderr)
        else:
            print(f"{human(total):>8}  {d}  (total)")
        walk(d, a.depth, rows)

    rows.sort(reverse=True)
    for n, path, mt in rows[: a.top]:
        print(f"{human(n):>8}  {mt}  {path}")


if __name__ == "__main__":
    main()
