"""Pick training chunks for the paper figure by how their token scores spread.

Score column t weights the loss of predicting chunk token t + 1. Two kinds of
chunk are selected, both judged over the window a figure panel shows:

``spread``  scores spread over many tokens deep in the chunk, so the panel is
            informative without the inflated scores of the opening tokens.
``early``   most of the chunk's score mass sits in the first ``early_cut``
            positions, the bias the position analysis found.

Writes the heatmap JSON layout (``negative``/``positive`` lists of token rows)
so token_heatmap_paper.py can render any pair of them.

Usage:
    python runs/deep_ignorance_ms/select_coverage_chunks.py \
        --scores runs/deep_ignorance_ms/magic_tf32_pertoken_b/scores \
        --out runs/deep_ignorance_ms/token_heatmap_coverage.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from bergson.cli.commands import Magic
from bergson.config.config_io import parse_steps, read_config
from bergson.data import load_scores
from bergson.magic.cli import attach_doc_ids_if_missing, shuffled_epochs
from bergson.utils.worker_utils import setup_data_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_yaml", default="runs/deep_ignorance_ms/magic_tf32_pertoken_b.yaml"
    )
    parser.add_argument(
        "--scores", default="runs/deep_ignorance_ms/magic_tf32_pertoken_b/scores"
    )
    parser.add_argument("--out", default="runs/deep_ignorance_ms/token_heatmap_coverage.json")
    parser.add_argument("--panel_tokens", type=int, default=190,
                        help="tokens a panel shows, used to size the judged window")
    parser.add_argument("--spread_start", type=int, default=64,
                        help="first position a spread panel shows")
    parser.add_argument("--early_cut", type=int, default=64)
    parser.add_argument("--num_spread", type=int, default=8)
    parser.add_argument("--num_early", type=int, default=4)
    args = parser.parse_args()

    ((_, run_cfg),) = parse_steps(
        read_config(args.run_yaml)["steps"], {"magic": Magic}
    )
    loaded = load_scores(Path(args.scores))
    assert loaded.offsets is not None, "expected a per-token score store"
    n_cols = int(loaded.offsets[1] - loaded.offsets[0])
    scores = np.asarray(loaded[:], dtype=np.float32).reshape(-1, n_cols)
    influence = np.abs(scores)
    print(f"score grid: {scores.shape}")

    train_ds, _ = setup_data_pipeline(run_cfg)
    train_ds = attach_doc_ids_if_missing(train_ds)
    train_ds = shuffled_epochs(train_ds, run_cfg.seed, max(1, run_cfg.num_epochs))
    scores = scores[: len(train_ds)]
    influence = influence[: len(train_ds)]

    total = influence.sum(axis=1)
    keep = total > 0
    early_frac = np.where(keep, influence[:, : args.early_cut - 1].sum(axis=1) / np.maximum(total, 1e-30), 0.0)
    peak_pos = influence.argmax(axis=1) + 1

    # Window a spread panel would show.
    lo = args.spread_start - 1
    hi = lo + args.panel_tokens
    win = influence[:, lo:hi]
    win_sum = win.sum(axis=1)
    win_max = win.max(axis=1)
    top1_share = np.where(win_sum > 0, win_max / np.maximum(win_sum, 1e-30), 1.0)
    visible = (win >= 0.15 * win_max[:, None]).sum(axis=1)
    win_mean = win.mean(axis=1)
    global_mean = float(influence.mean())

    spread_ok = (top1_share < 0.10) & (visible >= 25) & (win_mean >= 3 * global_mean)
    early_ok = (early_frac >= 0.5) & (peak_pos < 20)
    print(f"spread candidates: {int(spread_ok.sum())}; early candidates: {int(early_ok.sum())}")

    spread_rows = np.argsort(np.where(spread_ok, win_mean, -np.inf))[::-1][: args.num_spread]
    early_rows = np.argsort(np.where(early_ok, influence.max(axis=1), -np.inf))[::-1][: args.num_early]

    tokenizer = AutoTokenizer.from_pretrained(run_cfg.tokenizer or run_cfg.model)

    def pack(row, kind, start):
        rec = train_ds[int(row)]
        row_scores = scores[row].tolist()
        tokens = [
            {"t": tokenizer.decode([tid]), "s": round(row_scores[i - 1], 8) if i > 0 else None}
            for i, tid in enumerate(rec["input_ids"])
        ]
        print(
            f"{kind} row={row} start={start} win_mean={win_mean[row]:.2e} "
            f"top1_share={top1_share[row]:.3f} visible={visible[row]} "
            f"early_frac={early_frac[row]:.2f} peak_pos={peak_pos[row]}"
        )
        return {
            "kind": kind,
            "row": int(row),
            "start": start,
            "min": float(scores[row].min()),
            "peak": float(scores[row].max()),
            "early_frac": float(early_frac[row]),
            "peak_position": int(peak_pos[row]),
            "window_top1_share": float(top1_share[row]),
            "window_visible_tokens": int(visible[row]),
            "tokens": tokens,
        }

    seqs = [pack(r, "spread", args.spread_start) for r in spread_rows]
    seqs += [pack(r, "early", 0) for r in early_rows]
    out = {
        "negative": seqs,
        "positive": [],
        "num_rows": int(len(scores)),
        "global_mean_abs": global_mean,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out} with {len(seqs)} chunks")


if __name__ == "__main__":
    main()
