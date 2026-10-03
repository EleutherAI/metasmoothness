# Token attribution heatmaps (paper Figure 5)

Per-token MAGIC scores for two deep-ignorance training passages (a PIAS1
caspase-cleavage passage and an orangutan translocation passage), rendered in
the paper's heatmap style, for two models.

| file | what it is |
|---|---|
| `token_heatmap_paper.py` | The renderer used for the paper figure. Blue = negative, red = positive; `--row_scale` scales each panel separately (the paper figure uses it). Needs only numpy + matplotlib. |
| `data/deep_ignorance_paper_tokens.json` | Pythia deep-ignorance scores for the two chunks in the paper figure (rows 66534 and 65750 of `token_heatmap_coverage200.json`, from `magic_tf32_pertoken_b`: `EleutherAI/deep-ignorance-e2e-strong-filter`, 1024-token chunks, batch 128, lr 1e-5). |
| `data/olmo_paper_tokens.json` | The same two passages scored by `allenai/Olmo-3.1-32B-Think` (run `runs/olmo32b/magic_full_L32b1024`: 512-token chunks, fp32, batch 1024, lr 1e-5 constant with 6 warm-up steps, LoRA r=32 on layers 32-63, 257 steps; metasmoothness 0.963 for this exact setup). |
| `figures/*.pdf` | Rendered panels (commands below). |
| `select_coverage_chunks.py` | How the deep-ignorance chunks were selected (`spread` / `early`). |
| `extract_olmo_chunks.py`, `extract_olmo_continuation.py` | How the OLMo chunks were located: rebuild the run's chunking + `shuffle(seed=42)`, assert the rebuilt `doc_ids` equal the saved `scores/doc_ids.npy` (they matched for all 262,239 rows), then read each row's scores. Run from a bergson checkout. |

JSON layout: `{"negative": [ {kind, row, start, tokens: [{t, s, ...}]} ], "positive": []}`.
`s` is the score column for that token; a chunk's first token has `s = null`
(never predicted).

## OLMo notes

* The PIAS1 passage crosses a 512-token boundary, so its entry is row 46908
  followed by the next chunk of the same document (row 69673,
  `continuation_row`); the token at the join (`by`) has no score.
* OLMo tokenises "orangutans" as `orang|ut|ans`, so its panel starts at `utans`
  rather than the paper's `angutans`.
* OLMo's raw scores are orders of magnitude smaller than Pythia's; compare the
  colour pattern, not values.

## Reproduce

```
python token_heatmap_paper.py --data data/deep_ignorance_paper_tokens.json \
    --out figures/deep_ignorance_paper_tokens.pdf --rows 0 1 --start 64 0 --lines 10 --row_scale
python token_heatmap_paper.py --data data/olmo_paper_tokens.json \
    --out figures/olmo_paper_tokens_row_scale.pdf --rows 0 1 --start 350 270 --lines 10 --row_scale
python token_heatmap_paper.py --data data/olmo_paper_tokens.json \
    --out figures/olmo_paper_tokens_shared_scale.pdf --rows 0 1 --start 350 270 --lines 10
```
