#!/usr/bin/env python3
"""Write the dataset card for a published retrain bank.

The point of the card is that the models are the asset. A visitor scanning the
file list sees 500 safetensors and a CSV; what they need to know is that those
are 100 models each retrained from the same seed with a different 1% of the
corpus held out, and that validation.csv already records what that removal did
to each held-out query's loss. That pairing is what makes the bank reusable:
any new attribution method can be scored against it without retraining anything.

    python bank_card.py <run_id>
"""
import argparse
import csv
import sys

from huggingface_hub import HfApi

EXP = "/mnt/ssd-2/lucia/metasmoothness/experiments.csv"
DELTAS = "/mnt/ssd-2/lucia/metasmoothness/data/filter_deltas.csv"
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
    opt, _eps, n, var = s.split("_", 3)
    parts = [opt, "N" + n]
    if var.startswith("bs"):
        parts.append(var)
    else:
        parts.extend(["bs256", var])
    return f"{ORG}/LDS-retrain-bank-" + "-".join(parts)


def fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "not measured"


def card(rid, row, dl):
    n = int(float(row["n_docs"] or 0))
    frac = float(row["subset_fraction"] or 0.01)
    removed = int(n * frac)
    lines = [
        "---",
        "license: apache-2.0",
        "tags:",
        "  - training-data-attribution",
        "  - influence-functions",
        "  - interpretability",
        "---",
        "",
        f"# Retrain bank: `{rid}`",
        "",
        "**This repository contains 100 fully retrained language models**, not just scores.",
        "",
        f"Each model is GPT-2 ({row['model']}) fine-tuned on the same {n:,}-document corpus "
        f"with a different random **{frac:.0%} ({removed:,} documents) held out**, from the same "
        f"seed and the same data order as the base model in `retrained/base`. Retraining is "
        "deterministic within one environment, so the models differ only by the documents removed.",
        "",
        "That is the expensive part of any leave-k-out attribution study, and it is reusable: "
        "**a new attribution method can be evaluated against this bank without retraining anything.**",
        "",
        "## What is here",
        "",
        "| path | what it is |",
        "|---|---|",
        "| `retrained/base/` | the unablated fine-tuned model |",
        "| `retrained/subset_*/` | 100 models, each missing a different 1% of the corpus |",
        "| `validation.csv` | **the ground truth**: per (subset, query) change in loss caused by that removal |",
        "| `subsets.json` | which document ids each subset removed |",
        "| `config.yaml` | the exact training configuration |",
        "| `filter_proponents_*/` | tail-filter results: loss change when a scorer's top-ranked 1% is removed |",
        "",
        "## Using it",
        "",
        "```python",
        "from huggingface_hub import snapshot_download",
        "import pandas as pd",
        "",
        f'path = snapshot_download("{bank_repo_id(rid)}", repo_type="dataset")',
        "",
        "# ground truth: what removing each subset did to each query's loss",
        'truth = pd.read_csv(f"{path}/validation.csv")',
        "",
        "# score your own method, then correlate its predicted influence against `diff`",
        "# LDS = mean over queries of Spearman(predicted subset sums, measured diff)",
        "```",
        "",
        "## Measured on this bank",
        "",
        "| metric | value |",
        "|---|---|",
        f"| MAGIC LDS | {fmt(row.get('magic_lds'))} |",
        f"| EK-FAC LDS | {fmt(row.get('ekfac_lds'))} |",
        f"| metasmoothness | {fmt(row.get('metasmoothness'))} |",
    ]
    if dl:
        lines += [
            f"| tail-filter delta, MAGIC | {fmt(dl.get('magic_mean'), 5)} nats |",
            f"| tail-filter delta, EK-FAC | {fmt(dl.get('ekfac_mean'), 5)} nats |",
            f"| tail-filter delta, random control | {fmt(dl.get('random_mean'), 5)} nats |",
        ]
    lines += [
        "",
        "LDS is the mean per-query Spearman correlation between a scorer's predicted subset "
        "influence and the measured `diff`. The tail-filter delta is a different question on the "
        "same bank: remove the 1% a scorer ranks most influential, retrain once, and measure the "
        "query loss change against the bank's random removals as the matched control.",
        "",
        "## Provenance",
        "",
        f"- optimizer `{row['optimizer']}`, lr `{row['lr']}`, batch size `{row['batch_size']}`, "
        f"`{row['num_epochs']}` epochs, `{row['steps']}` steps, seed `{row['seed']}`",
        f"- corpus: {row['dataset']}, {n:,} documents",
        "- retrains for one bank all run on a single GPU type: mixing types changes the retrained "
        "models by enough to shift LDS by ~0.05, which is larger than most effects being measured.",
        "",
        "Produced by [bergson](https://github.com/EleutherAI/bergson).",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    a = ap.parse_args()

    rows = {r["run_id"]: r for r in csv.DictReader(open(EXP))}
    row = rows.get(a.run_id)
    if not row:
        sys.exit(f"{a.run_id} not in experiments.csv")
    dl = {}
    try:
        dl = {r["run"]: r for r in csv.DictReader(open(DELTAS))}.get(a.run_id, {})
    except OSError:
        pass

    repo_id = bank_repo_id(a.run_id)
    HfApi().upload_file(
        path_or_fileobj=card(a.run_id, row, dl).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Dataset card: the bank is 100 retrained models plus measured loss changes",
    )
    print(f"card written to {repo_id}")


if __name__ == "__main__":
    main()
