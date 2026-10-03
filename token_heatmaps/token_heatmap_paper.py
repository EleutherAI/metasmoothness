"""Render the paper-style token attribution figure (EMNLP Figure 5).

Each paragraph is the opening of one of the top-scoring training chunks, set in
a monospace font. Token backgrounds run blue (negative) through white to red
(positive); all paragraphs share one symmetric scale set by a high percentile of
their pooled |score| so the heavy-tailed scores stay legible. Text is always black.

Usage:
    python token_heatmap_paper.py --data token_heatmap_top5_both.json \
        --out token_heatmap_negative_paper.pdf
"""

import argparse
import json
import unicodedata

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402

FONT_SIZE = 5.6
LINE_HEIGHT = 1.45 * FONT_SIZE
PARA_GAP = 1.0 * LINE_HEIGHT

BLUE = (26 / 255, 82 / 255, 208 / 255)
NEUTRAL = (1.0, 1.0, 1.0)
RED = (206 / 255, 22 / 255, 22 / 255)
CMAP = LinearSegmentedColormap.from_list("score", [BLUE, NEUTRAL, RED])


def shade(score, vmax, gamma):
    """Color for one token. ``gamma`` below 1 lifts mid-sized scores."""
    t = max(-1.0, min(1.0, score / vmax)) if vmax > 0 else 0.0
    return CMAP(0.5 + 0.5 * (abs(t) ** gamma) * (1 if t >= 0 else -1))


def drawable(text):
    """Drop zero-width formatting marks and flatten control characters.

    The PDF backend raises on a character the font maps to no glyph at all,
    which byte-level tokenizers emit for things like U+FEFF.
    """
    out = []
    for ch in text:
        category = unicodedata.category(ch)
        if category == "Cf":
            continue
        out.append(" " if category == "Cc" else ch)
    return "".join(out)


def layout_tokens(tokens, width_pt, char_w):
    """Wrap tokens onto a monospace grid; return (x, line, w, text, score).

    Each token's highlight covers its leading space too, as in the paper, except
    at the start of a line where the space is dropped.
    """
    # Group sub-word pieces so lines only break at spaces.
    words = []
    for tok in tokens:
        text = drawable(tok["t"])
        if not text:
            continue
        if not words or text.startswith(" ") or words[-1][-1][0].endswith(" "):
            words.append([])
        # A chunk's first token is never predicted, so it carries no score.
        words[-1].append((text, tok["s"] if tok["s"] is not None else 0.0))

    placed = []
    x, line = 0.0, 0
    for word in words:
        word_len = sum(len(t) for t, _ in word) - (len(word[0][0]) - len(word[0][0].lstrip(" ")))
        if x > 0 and x + len("".join(t for t, _ in word)) * char_w > width_pt:
            x, line = 0.0, line + 1
        for i, (text, score) in enumerate(word):
            if x == 0:
                text = text.lstrip(" ")
            # Words longer than a whole line still break mid-word.
            if x > 0 and word_len * char_w > width_pt and x + len(text) * char_w > width_pt:
                x, line = 0.0, line + 1
            w = len(text) * char_w
            placed.append((x, line, w, text, score))
            x += w
    return placed, line + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="token_heatmap_top5_both.json")
    parser.add_argument("--out", default="token_heatmap_negative_paper.pdf")
    parser.add_argument("--split", default="negative", choices=["positive", "negative"])
    parser.add_argument("--rows", type=int, nargs="+", default=[0, 1],
                        help="indices into the split to render")
    parser.add_argument("--lines", type=int, default=10, help="lines per paragraph")
    parser.add_argument("--start", type=int, nargs="+", default=[0],
                        help="skip this many tokens before the window starts; "
                             "one value for every row, or one per row")
    parser.add_argument("--percentile", type=float, default=99.0,
                        help="pooled |score| percentile mapped to full color")
    parser.add_argument("--gamma", type=float, default=0.6,
                        help="below 1 gives mid-sized scores more color; "
                             "1.0 is a linear ramp")
    parser.add_argument("--row_scale", action="store_true",
                        help="scale each paragraph by its own scores, for rows "
                             "whose scores differ by orders of magnitude")
    parser.add_argument("--width", type=float, default=3.6, help="figure width in inches")
    parser.add_argument("--font", default="DejaVu Sans Mono")
    parser.add_argument(
        "--sentence-start", action="store_true",
        help="start each paragraph at a sentence boundary; may drop the peak token",
    )
    args = parser.parse_args()

    plt.rcParams["pdf.fonttype"] = 42
    prop = FontProperties(family=args.font, size=FONT_SIZE)
    char_w, _, _ = TextToPath().get_text_width_height_descent("M", prop, ismath=False)

    data = json.load(open(args.data))
    width_pt = args.width * 72

    starts = args.start * len(args.rows) if len(args.start) == 1 else args.start
    assert len(starts) == len(args.rows), "--start takes one value or one per row"

    paras = []
    for idx, start in zip(args.rows, starts):
        seq = data[args.split][idx]
        tokens = seq["tokens"][start:]
        if args.sentence_start:
            # First token that follows sentence-ending punctuation and a space.
            for i in range(1, len(tokens)):
                prev, cur = tokens[i - 1]["t"], tokens[i]["t"]
                ends = prev.rstrip().endswith((".", "\n")) or prev.endswith("\n")
                if ends and (cur[:1].isupper() or cur[1:2].isupper()):
                    tokens = tokens[i:]
                    break
        placed, _ = layout_tokens(tokens, width_pt, char_w)
        placed = [p for p in placed if p[1] < args.lines]
        paras.append(placed)
        print(f"row {seq['row']}: {len(placed)} tokens, "
              f"min={min(p[4] for p in placed):.4f}")

    if args.row_scale:
        vmaxes = []
        for placed in paras:
            vmax = float(np.percentile(np.abs([p[4] for p in placed]), args.percentile))
            vmaxes.append(vmax)
            print(f"row vmax={vmax:.4f}")
    else:
        scores = np.abs([p[4] for placed in paras for p in placed])
        vmax = float(np.percentile(scores, args.percentile))
        vmaxes = [vmax] * len(paras)
        scope = "shared vmax" if len(paras) > 1 else "vmax"
        print(f"{scope}={vmax:.4f}")

    height_pt = len(paras) * args.lines * LINE_HEIGHT + (len(paras) - 1) * PARA_GAP
    fig = plt.figure(figsize=(args.width, height_pt / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_pt)
    ax.set_ylim(height_pt, 0)
    ax.axis("off")

    ascent, descent = 0.95 * FONT_SIZE, 0.33 * FONT_SIZE
    y0 = 0.0
    for placed, vmax in zip(paras, vmaxes):
        for x, line, w, text, score in placed:
            baseline = y0 + line * LINE_HEIGHT + (LINE_HEIGHT + ascent - descent) / 2
            ax.add_patch(Rectangle((x, baseline - ascent), w, ascent + descent,
                                   facecolor=shade(score, vmax, args.gamma),
                                   edgecolor="none",
                                   linewidth=0))
            ax.text(x, baseline, text, fontproperties=prop, color="black",
                    va="baseline", ha="left")
        y0 += args.lines * LINE_HEIGHT + PARA_GAP

    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {args.out} ({height_pt / 72:.2f} in tall)")


if __name__ == "__main__":
    main()
