# Attribution-method LDS on GPT-2 fine-tune banks (eps_root 1e-8): adamw vs muon

Setup: GPT-2 fine-tuned on `EleutherAI/bergson-wikitext-512-chunks` (train),
`eps_root 1e-8`, 4 epochs, batch_size 64, 100 leave-1%-out subsets, full-run
retraining. Query = `test[1:51]` (50 docs). Metric = mean over the 50 queries
of the per-query LDS Spearman (subset score-sum vs subset query-loss diff).

Banks (all `bergson-damping/runs/`):
- adamw: `gpt2_wikitext_bank` (lr 8e-4, loss 2.97)
- muon lr 8e-4: `gpt2_muon_wikitext_bank` at lr 8e-4 (loss 3.61, undertrained)
- muon lr 2e-4: `gpt2_muon_wikitext_bank_lr2e-4` (loss 2.95, loss-matched to adamw)
- muon lr 8e-5: `gpt2_muon_wikitext_bank` (loss 2.87, muon's minimum)

Baselines: `bergson-damping/examples/bank_baselines/`.
Gradient methods: TrackStar run without Adam normalization; SOURCE skipped.

| Method | adamw 1e-8 (loss 2.97) | muon 1e-8 (lr 8e-4, loss 3.61) | muon 1e-8 (lr 2e-4, loss 2.95) | muon 1e-8 (lr 8e-5, loss 2.87) |
| --- | --- | --- | --- | --- |
| MAGIC (per-query, 50 backward passes) | 0.5087 | pending | pending | 0.8575 |
| EK-FAC | 0.3726 | 0.0643 | 0.3804 | 0.4960 |
| TrackStar (no Adam norm) | 0.1764 | 0.0374 | 0.1680 | 0.2395 |
| BM25 lexical | 0.16 | 0.0625 | 0.1821 | 0.2552 |
| activation similarity | 0.09 | 0.0690 | 0.0956 | 0.1220 |
| Qwen3-Embedding-8B | 0.11 | 0.0536 | 0.1036 | 0.1590 |
| Jina v3 semantic | 0.06 | FAIL | 0.0881 | 0.0945 |
| gradient cosine | 0.05 | 0.0038 | 0.0298 | 0.0505 |
