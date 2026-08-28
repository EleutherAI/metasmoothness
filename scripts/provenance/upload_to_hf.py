"""Upload a leave-k-out bank (base + subset models) to a single HF repo.

Each N gets ONE model repo under EleutherAI, with the fully-trained model under
``base/`` and every leave-k-out model under ``subset_<i>/`` as subfolders, so
the whole bank for a training-set size lives at one URL.

Run:
    python scripts/ekfac_vs_n/upload_to_hf.py \
        --bank_dir runs/ekfac_vs_n/N4k/retrained \
        --repo EleutherAI/bergson-smollm2-lds-4k
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

DATA_REPO = "EleutherAI/bergson-smollm2-lds-chunks"


def model_card(repo: str, n_subsets: int) -> str:
    # Derive the size tag (4k/8k/16k) and matching dataset config from the repo.
    size = repo.rsplit("-", 1)[-1]
    return f"""---
datasets:
- {DATA_REPO}
base_model:
- gpt2
tags:
- influence-functions
- linear-datamodeling-score
- ekfac
---

# {repo}

Leave-k-out model bank for an EK-FAC linear-datamodeling-score (LDS) study of
how attribution quality scales with training-set size N (this repo: N={size}
512-token chunks). GPT-2 fine-tuned on the `{size}` subset of
[`{DATA_REPO}`](https://huggingface.co/datasets/{DATA_REPO}) (packed from
`EleutherAI/SmolLM2-135M-10B`).

- `base/`: GPT-2 fine-tuned on the full N-chunk training set (no leave-out).
- `subset_0/` … `subset_{n_subsets - 1}/`: each retrains from scratch with a
  random 1%% of the training chunks held out, forming the LDS leave-k-out bank.

Queries: the `query` config of the dataset repo (50 held-out chunks).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank_dir", required=True, help="<run>/retrained directory")
    ap.add_argument("--repo", required=True, help="target HF model repo id")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    bank = Path(args.bank_dir)
    if not (bank / "base").exists():
        raise FileNotFoundError(f"{bank}/base not found; is the bank complete?")

    n_subsets = len(list(bank.glob("subset_*")))
    (bank / "README.md").write_text(model_card(args.repo, n_subsets))

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"Uploading {bank} -> https://huggingface.co/{args.repo}")

    # allow_patterns keeps us from uploading stray files; base/ + subset_*/ only.
    api.upload_large_folder(
        folder_path=str(bank),
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=["base/*", "subset_*/*", "README.md"],
    )
    print(f"Done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
