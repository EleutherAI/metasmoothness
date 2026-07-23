# Metasmoothness and LDS: Empirical Analysis

Understanding how training-algorithm design choices affect data-weight **metasmoothness**
(Chang et al. 2024, Def. 2), and in turn how metasmoothness affects the **Linear Datamodeling
Score (LDS)** of influence-function attribution methods (EK-FAC, MAGIC).

Testbeds: GPT-2 and a GPT-2-sized OLMo-2 (which has QK-norm), on the
`EleutherAI/SmolLM2-135M-10B` corpus (4k/8k/16k pre-chunked subsets on HF).

See **[LDS_RESULTS.md](LDS_RESULTS.md)** for the current grid of metasmoothness and LDS results.
