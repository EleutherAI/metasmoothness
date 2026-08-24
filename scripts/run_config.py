"""Canonical location of a run's config.

A run directory holds outputs and is disposable: it gets emptied by disk-space
sweeps, and /mnt/ssd-2 has run full more than once. The config is the
*definition* of the run, so it must not live only there. The git-tracked copy
under ``configs/`` is authoritative; the copy inside the run directory is a
convenience that may vanish at any time.

Two failures came out of ignoring that. Both times a run directory was left
holding only bergson's own ``config.yaml`` and the launch command pointed at a
``tune.yaml``/``experiment.yaml`` that no longer existed -- and because bergson
falls through to argparse for a path that is not a file, the error read
``invalid choice: '/mnt/.../tune.yaml'`` rather than "no such config".

Everything that reads or writes a run config goes through this module. Readers
get the tracked copy; a run-local copy found without a tracked counterpart is
promoted into ``configs/`` on the spot, so a config can only ever be lost once.
"""

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"

# Which configs/ subdirectory mirrors which run-local filename. The mirror is
# always named after the run directory, so tuning's _s42 suffix is preserved.
MIRROR_DIR = {
    "experiment.yaml": "experiments",
    "tune.yaml": "tuning",
}


def mirror_for(run_dir, filename="experiment.yaml") -> Path:
    """The git-tracked path for this run's config."""
    run_dir = Path(run_dir)
    try:
        sub = MIRROR_DIR[filename]
    except KeyError:
        raise ValueError(
            f"no configs/ mirror defined for {filename!r}; "
            f"add it to MIRROR_DIR in {__file__}"
        ) from None
    return CONFIGS / sub / f"{run_dir.name}.yaml"


def save(cfg: dict, run_dir, filename="experiment.yaml") -> Path:
    """Write a run config, tracked copy first.

    The tracked copy is written before the run-local one so that a crash in
    between leaves the authoritative copy on disk, never only the disposable
    one. Returns the path to launch from.
    """
    run_dir = Path(run_dir)
    mirror = mirror_for(run_dir, filename)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    with mirror.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / filename).open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return mirror


def resolve(run_dir, filename="experiment.yaml") -> Path:
    """Path to this run's config, preferring the tracked copy.

    Restores the run-local copy if it is missing, and promotes an untracked
    run-local copy into ``configs/``. Raises naming both paths if neither
    exists -- an honest error beats bergson's ``invalid choice``.
    """
    run_dir = Path(run_dir)
    mirror = mirror_for(run_dir, filename)
    local = run_dir / filename

    if mirror.is_file():
        if not local.is_file() and run_dir.is_dir():
            local.write_text(mirror.read_text())
        return mirror

    if local.is_file():
        # Untracked: promote it before anything can sweep the run directory.
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text(local.read_text())
        print(f"promoted untracked config to {mirror} -- commit it", file=sys.stderr)
        return mirror

    raise FileNotFoundError(
        f"no config for run {run_dir.name}: neither {mirror} (tracked) "
        f"nor {local} (run-local) exists. Regenerate it with the matching "
        f"scripts/gen_*.py -- do not hand-write it."
    )


def load(run_dir, filename="experiment.yaml") -> dict:
    """Parsed run config, from the authoritative copy."""
    return yaml.safe_load(resolve(run_dir, filename).read_text())


def audit(roots) -> int:
    """Report run directories whose config is not tracked. Returns a count."""
    missing = 0
    for root in roots:
        for run_dir in sorted(Path(root).iterdir()):
            if not run_dir.is_dir():
                continue
            for filename in MIRROR_DIR:
                local, mirror = run_dir / filename, mirror_for(run_dir, filename)
                if local.is_file() and not mirror.is_file():
                    print(f"UNTRACKED {run_dir.name}/{filename}")
                    missing += 1
                elif mirror.is_file() and not local.is_file():
                    print(f"run-local missing (recoverable) {run_dir.name}/{filename}")
    return missing


if __name__ == "__main__":
    roots = sys.argv[1:] or [
        "/mnt/ssd-2/lucia/paper_runs/experiments",
        "/mnt/ssd-2/lucia/paper_runs/tuning",
    ]
    sys.exit(1 if audit(roots) else 0)
