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

def bank_repo_id(run_id: str) -> str:
    """Hub repo name for a run's retrain bank, e.g. LDS-retrain-bank-muon-N16k-bs256.

    Renamed 2026-08-27 from metasmoothness-bank-<run_id>; the Hub redirects the
    old names, but new repos must be created under the new scheme.
    """
    s = run_id
    for a, b in [("plan_adam_", "adamw_"), ("plan_muon_", "muon_"),
                 ("sm_adamw_", "adamw_"), ("sm_muon_", "muon_")]:
        s = s.replace(a, b)
    # Rows outside the plan_/sm_ naming scheme -- gpt2medium_16k_bs32,
    # london16k_bs256_muon -- have no <opt>_<eps>_<n>_<var> to unpack and used to
    # die here with "not enough values to unpack", so they could not be published
    # at all. Fall back to the run_id itself, which is unambiguous even if less
    # tidy, rather than refusing to archive the bank.
    bits = s.split("_", 3)
    if len(bits) < 4:
        safe = run_id.replace("_", "-")
        return f"{ORG}/LDS-retrain-bank-{safe}"
    opt, _eps, n, var = bits
    parts = [opt, "N" + n]
    if var.startswith("bs"):
        parts.append(var)
    else:
        parts.extend(["bs256", var])
    return f"{ORG}/LDS-retrain-bank-" + "-".join(parts)
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
    repo_id = bank_repo_id(a.run_id)
    api = HfApi()

    n_models = len(list((root / "retrained").glob("subset_*")))
    if n_models != 100:
        sys.exit(f"refusing: bank is {n_models}/100")

    # Counting subset_* directories only proves the directory entries exist. It
    # does not prove this process can READ them, and on this fleet that is a real
    # distinction: `lucia` is uid 1001 on three nodes and 1000 on the other seven,
    # CephFS stores the numeric uid, and the default umask writes 0600. A bank
    # built on one side lists fine from the other and fails on open. That had
    # already happened -- 42 of the 100 models in plan_adam_eps1e17_32k_bs256 were
    # unopenable from seven of ten nodes -- so a publish from one of those nodes
    # would have shipped 58 models and reported success.
    #
    # So open every file that is meant for upload. One byte is enough; the point
    # is to trigger the permission check, not to read 500 MB a hundred times.
    # This runs BEFORE create_repo, so an unreadable bank never reaches the Hub
    # even partially, and long before --delete-local can touch anything.
    unreadable = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, root)
            if not wanted(rel):
                continue
            try:
                with open(path, "rb") as fh:
                    fh.read(1)
            except OSError as e:
                unreadable.append((rel, e.strerror))
    if unreadable:
        print(f"{len(unreadable)} file(s) meant for upload cannot be opened by "
              f"uid {os.getuid()}:", file=sys.stderr)
        for rel, why in unreadable[:10]:
            print(f"  {rel}: {why}", file=sys.stderr)
        if len(unreadable) > 10:
            print(f"  ... and {len(unreadable) - 10} more", file=sys.stderr)
        sys.exit("refusing to publish a bank this process cannot fully read -- "
                 "chmod a+rX the run dir from a node whose uid owns it "
                 "(see notes/uid_split.md), then retry")
    print(f"readable: every file meant for upload opens as uid {os.getuid()}")

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
