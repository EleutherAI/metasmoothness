"""Upload the sized chunk datasets to one HF dataset repo with named configs.

One repo, four selectable subsets (configs): 4k / 8k / 16k training pools
(nested) and the disjoint 50-chunk query set. Linked from the model repos so
each leave-k-out bank points back at the exact data it was trained on.

Run:
    python scripts/ekfac_vs_n/upload_datasets.py \
        --data_dir runs/ekfac_vs_n/datasets \
        --repo EleutherAI/bergson-smollm2-lds-chunks
"""

import argparse

from datasets import load_from_disk

# config_name -> on-disk split dir
CONFIGS = {
    "4k": "train_4k.hf",
    "8k": "train_8k.hf",
    "16k": "train_16k.hf",
    "query": "query_50.hf",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="runs/ekfac_vs_n/datasets")
    ap.add_argument("--repo", default="EleutherAI/bergson-smollm2-lds-chunks")
    args = ap.parse_args()

    for config_name, subdir in CONFIGS.items():
        ds = load_from_disk(f"{args.data_dir}/{subdir}")
        print(f"Pushing config '{config_name}' ({ds.num_rows} rows) -> {args.repo}")
        ds.push_to_hub(args.repo, config_name=config_name, split="train")
    print(f"Done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
