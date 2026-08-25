#!/usr/bin/env python3
"""Publish a completed retrain bank, file it in the Data Attribution collection,
and verify it landed -- so the local copy can be released with confidence.

Lucia's rule: a bank leaves the node only once everything derived from it exists
AND it is definitely public. "Definitely" is the point of the verify step: an
upload_folder that half-finishes leaves a repo that looks plausible in the web
UI, and the local copy is the only other copy.

    python publish_bank.py <run_id> [--delete-local]

Verification compares the local file list against the repo's, restricted to what
was actually meant to be uploaded, and refuses on any missing file. --delete-local
removes the retrained models only after that passes.
"""
import argparse
import fnmatch
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, add_collection_item, create_collection, get_collection
from huggingface_hub.utils import HfHubHTTPError

ORG = "EleutherAI"
COLLECTION_TITLE = "Data Attribution"
COLLECTION_DESC = (
    "Leave-1%-out retrain banks and datasets for training-data attribution: "
    "measured per-query loss changes, the ground truth for LDS."
)
IGNORE = ["checkpoints/*", "optimizer.pt", "*.log", "*.tmp", "ekfac_scores/*",
          "scores/*", "per_query/*", "slice_*.yaml", "*.premerge", "*.merged"]


def wanted(rel):
    return not any(fnmatch.fnmatch(rel, pat) for pat in IGNORE)


def find_root(run_id):
    for base in ("/mnt/ssd-2", "/mnt/ssd-1"):
        c = Path(base) / "lucia/paper_runs/experiments" / run_id
        if c.is_dir():
            return c
    sys.exit(f"run dir not found for {run_id}")


def ensure_collection(api):
    """Return the collection slug, creating it once if absent."""
    for c in api.list_collections(owner=ORG):
        if c.title == COLLECTION_TITLE:
            return c.slug
    col = create_collection(COLLECTION_TITLE, namespace=ORG, description=COLLECTION_DESC,
                            exists_ok=True)
    print(f"created collection {col.slug}")
    return col.slug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--delete-local", action="store_true")
    a = ap.parse_args()

    root = find_root(a.run_id)
    repo_id = f"{ORG}/metasmoothness-bank-{a.run_id}"
    api = HfApi()

    n_models = len(list((root / "retrained").glob("subset_*")))
    if n_models != 100:
        sys.exit(f"refusing: bank is {n_models}/100")

    probe = subprocess.run(
        ["/home/lucia/envs/paper/bin/python", "-s", "-P",
         "/mnt/ssd-2/lucia/metasmoothness/scripts/magic_lds.py", str(root)],
        capture_output=True, text=True, timeout=1800)
    if probe.returncode != 0:
        sys.exit(f"refusing: bank does not score cleanly -- "
                 f"{(probe.stderr or probe.stdout).strip().splitlines()[-1:]}")
    print(f"scoreable: {probe.stdout.strip().splitlines()[0]}")

    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(root),
                      ignore_patterns=IGNORE,
                      commit_message=f"Retrain bank for {a.run_id}: 100 leave-1%-out models "
                                     f"+ validation and tail-filter results")
    print(f"uploaded to {repo_id}")

    # --- verify -----------------------------------------------------------
    remote = set(api.list_repo_files(repo_id, repo_type="dataset"))
    local = set()
    for dirpath, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(dirpath, f), root)
            if wanted(rel):
                local.add(rel)
    missing = sorted(local - remote)
    print(f"verify: {len(local)} local files meant for upload, {len(remote)} in repo, "
          f"{len(missing)} missing")
    if missing:
        for m in missing[:10]:
            print(f"  MISSING {m}")
        sys.exit("refusing to touch the local copy: upload is incomplete")

    slug = ensure_collection(api)
    try:
        add_collection_item(slug, item_id=repo_id, item_type="dataset", exists_ok=True)
        print(f"added to collection {slug}")
    except HfHubHTTPError as e:
        print(f"collection add failed (upload is still verified): {e}")

    if a.delete_local:
        import shutil
        d = root / "retrained"
        size = int(os.getxattr(str(d), "ceph.dir.rbytes")) / 2**30
        shutil.rmtree(d)
        print(f"removed local retrained/ ({size:.0f} GiB); published copy at {repo_id}")


if __name__ == "__main__":
    main()
