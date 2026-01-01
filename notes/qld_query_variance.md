# Quantifying query-to-query variation in QLD

Written 2026-09-05 to put numbers behind the claim that QLD varies with query, that
in-distribution queries give larger QLDs, and that queries from one set differ from
each other. Regenerate with

    python scripts/qld_query_variance.py --csv data/qld_per_query.csv
    python scripts/query_contamination.py --n 32000 64000

Scope: the GPT-2 AdamW token-scaling series, 4k-512k, both removal rules
(top 1%, top 40), all methods with a complete 20-query summary (ekfac, bm25, magic,
jina). 60 complete cells, 1200 per-query QLDs. QLD = `filter_change - random_mean`.

## 1. Why in-distribution queries score higher: it is verbatim overlap, and it
##    switches on at 64k

`query_contamination.py` probes each query with 16 spans of 32 consecutive tokens and
searches the training pool for each span verbatim:

    train_4k / 16k / 32k    query_20          0/20 queries found, probe recall 0.00
    train_64k               query_20         20/20 queries found, probe recall 1.00
    train_4k / 16k / 64k    query_20_heldout  0/20 queries found, probe recall 0.00

The train chain is a verified nested prefix, so 64k implies every larger rung. This
is the re-chunking overlap flagged in README 2026-09-04: query_20 was excluded by
chunk index, but the same source text re-enters at a different chunk boundary.

The in-distribution advantage tracks that boundary exactly:

    N < 64k  (query text absent from train)   10 cells, in-dist/held-out ratio
                                              geometric mean 1.57x, range 1.22-2.18
    N >= 64k (query text present in train)     6 cells, geometric mean 4.71x,
                                              median 4.42x, range 2.03-10.55
    Mann-Whitney on the ratios, high vs low:  p < 0.001

Per-cell, EK-FAC, top 1% / top 40 (mean +- 95% CI over 20 queries):

    N      in-dist top1%     held-out top1%   ratio   in-dist top40    held-out top40   ratio
    4k     0.0115+-0.0032    0.0083+-0.0018   1.38    --               --               --
    8k     0.0290+-0.0154    0.0196+-0.0045   1.48    0.0208+-0.0147   0.0126+-0.0033   1.65
    16k    0.0527+-0.0240    0.0292+-0.0065   1.80    0.0264+-0.0106   0.0121+-0.0030   2.18
    32k    0.0533+-0.0062    0.0394+-0.0053   1.35    0.0186+-0.0045   0.0106+-0.0022   1.76
    64k    0.1233+-0.0123    0.0410+-0.0052   3.01    0.0813+-0.0098   0.0077+-0.0026  10.55
    128k   0.2140+-0.0211    0.0752+-0.0110   2.84    0.1373+-0.0178   0.0135+-0.0091  10.21
    256k   0.2069+-0.0193    0.1018+-0.0204   2.03    0.1073+-0.0154   0.0184+-0.0181   5.83

Below 64k no cell separates at p < 0.01 except the two 32k rows; from 64k on every
cell is p < 1e-8. The effect is largest under the top-40 rule, where 40 documents is
too small a budget for anything but the duplicate to matter.

**Every method finds the same duplicate.** Spearman rank agreement between methods on
per-query QLD, same run and query set:

    top 40, N < 64k    15 pairs, median rho 0.63  [0.34, 0.89]
    top 40, N >= 64k    6 pairs, median rho 0.99  [0.97, 0.99]
    top 1%, N < 64k    17 pairs, median rho 0.65  [0.27, 0.87]
    top 1%, N >= 64k    6 pairs, median rho 0.81  [0.74, 0.96]

EK-FAC, BM25 and MAGIC agree at rho 0.99 once the query's own text is in the pool --
lexical and gradient-based attribution retrieve the same document -- against rho ~0.63
when it is not.

## 2. The spread between queries in one set, against the retrain-noise floor

Within a single (N, rule, method, query set) cell, over the 20 queries:

    in-dist   44 cells, CV median 43% [15%, 198%], p90/p10 median 2.1x
    held-out  16 cells, CV median 51% [29%, 225%], p90/p10 median 2.5x

That spread is not retrain noise. `random_sd` is the SD of the random-removal control
retrains for the same query, i.e. the noise floor on a single `filter_change`.
Restricted to the 19 cells with 100 control retrains (a 3-retrain SD is too noisy to
serve as a floor):

    SD across queries / control SD:  median 14x, range 5-56x
    query-attributable share of QLD variance (1 - noise_var/obs_var):
                                     median 99.5%, minimum 95.4%

## 3. It is not random -- it is a stable property of the query

If the between-query spread were noise it would not reproduce. It does:

    across methods, same run:            44 pairs, median rho 0.69, 100% positive
    across adjacent scale rungs,
    same rule/method/query set:          48 pairs, median rho 0.75,  98% positive

Held-out cells behave the same way (e.g. top 1% EK-FAC held-out, 8k->16k rho 0.92,
64k->128k rho 0.89), so this is not an artefact of contamination: some queries are
simply more filterable than others, consistently, at every scale and under every
attribution method.

## Caveat

Held-out control banks are 3 subsets (`random_n = 3`), in-distribution ones are mostly
100. Held-out means and the in-dist/held-out ratios are sound; held-out *noise floors*
are not, and are excluded from the section-2 floor comparison.

## 4. What the CV claim actually rests on (added 2026-09-05)

"CV 43-51%" in the first draft was the *median* of the in-dist cells (43%) and the
median of the held-out cells (51%). It is not a range: the 60 per-cell CVs run from
15% to 225%, IQR [29%, 83%], median 50%. Disaggregated:

    in-dist,  N <= 32k   26 cells, median CV 0.86  [0.17, 1.98]
    in-dist,  N >= 64k   18 cells, median CV 0.27  [0.15, 0.58]
    held-out, N <= 32k   10 cells, median CV 0.51  [0.31, 0.73]
    held-out, N >= 64k    6 cells, median CV 0.61  [0.29, 2.25]

Two things inflate the tails and must be disclosed:

**Query 17 is web boilerplate.** Every in-distribution cell with CV > 1 is one query:
q17, whose QLD is 6-8x its cell mean at 8k-32k under *every* method. Dropping the
single most extreme query takes those cells from 1.0-2.0 down to 0.35-0.65. q17 is a
WordPress post; 447 of its 16-token windows appear verbatim in train_8k, all of them
the comment-form boilerplate ("You are commenting using your account. Log Out /
Change..."). So it is not topical contamination -- the filter is removing training
chunks that share site furniture. Excluding q17 entirely: median CV 0.35 overall,
0.30 in-dist, 0.52 held-out.

**CV explodes when the mean approaches zero.** The 2.25 outlier is 256k top-40
held-out, mean 0.0184 with a 3-retrain control bank. Report p90/p10 next to CV.

**The cells are not independent.** All 60 reuse the same 20 queries; the train chain is
nested; top-1% and top-40 share the same scores; at 4k the two rules are the same
filter (40 docs is 1% of 4k), so those rows are duplicates by construction. Cell
counts are not sample sizes. The sampling uncertainty within a cell is also real:
n = 20, so a bootstrap CI on a single cell's CV is wide, e.g. 64k top-1% EK-FAC
in-dist CV 0.23 [0.16, 0.28], 8k top-1% EK-FAC in-dist CV 1.21 [0.25, 1.39].

Appendix table of all 60 cells (mean, SD, CV with bootstrap CI, min/max, p90/p10,
control SD and control n):

    python scripts/qld_query_variance.py --appendix markdown
    python scripts/qld_query_variance.py --appendix latex
