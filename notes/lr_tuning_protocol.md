# Reporting the learning-rate sweeps in the paper

Drafted 2026-09-04. Sources: `CONTROLS.md` ("Tuning protocol"), `DECISIONS.md`
("Why steps of 2x are the right spacing" / "The procedure" / "Sweep centers"),
`tuning.csv`, `qwen_tuning.csv`. The table body is generated, never hand-typed:

    python scripts/lr_sweep_table.py --latex

That script also **gates** the claim: it exits non-zero if any frozen row sits at
an lr the sweep did not select, or if a grid was left with a genuine endpoint
win. Run it before every submission build.

---

## 1. Replacement for the current main-text sentence

The current text says only that we "sweep over a range of three learning rates"
and add more "until the best-performing learning rate is an interior point". That
is accurate for the GPT-2 per-configuration sweeps but it under-reports three
things a reader needs: the grid *spacing*, where each grid was *centred*, and the
tie rule that decides near-ties. Suggested replacement:

> We use subsets from the SmolLM2-135M pre-training corpus for all experiments
> \citep{allal2025smollm2smolgoesbig}, and fine-tune models from GPT-2 small
> \citep{radford2019language} and Qwen 2.5 1.5B \citep{yang2024qwen2technicalreport}.
> Any configuration whose optimization problem differs from the reference
> configuration --- a different dataset size, batch size, epoch count, optimizer,
> or model --- receives its own peak learning-rate sweep before any attribution
> is run on it. For GPT-2 we sweep three learning rates spaced a factor of two
> apart, $\{0.5\times, 1\times, 2\times\}$ around the grid centre given in
> Table~\ref{tab:lr-centres}; if the lowest held-out loss falls on an endpoint we
> add one further point a factor of two beyond it and re-check, repeating at most
> twice. For Qwen 2.5 1.5B we instead sweep a fixed six-point grid,
> $\{1,2,5\}\times10^{-5}$ and $\{1,2,5\}\times10^{-4}$, at every dataset size,
> and use a single peak learning rate of $5\times10^{-5}$ across all Qwen sizes.
> Selection is always by mean per-token cross-entropy on a held-out set of 4000
> documents disjoint from every training set, never by training loss. Every
> selected learning rate is an interior point of its grid. The full grids,
> per-point held-out losses, and selected values are given in
> Appendix~\ref{app:lr}. Unless otherwise specified, we use the training
> hyperparameters in Table~\ref{tab:hyperparameters}.

Two edits of substance versus the current wording:

* **"three learning rates" is only the GPT-2 initial grid.** The reference sweep
  is five points and the Qwen sweeps are five or six, so the sentence should
  scope "three" to the per-configuration GPT-2 mini-sweeps.
* **"until the best is an interior point" is not quite the stopping rule.** The
  actual rule stops after two extensions, and near-ties are resolved by the tie
  rule below rather than by extending. Both are worth one clause each; a reader
  reproducing the protocol would otherwise diverge from us on exactly the cases
  where it matters.

## 2. Appendix subsection

```latex
\subsection{Learning-rate selection}
\label{app:lr}

Every experiment must run at a learning rate close enough to its optimum that no
comparison in the paper is driven by one arm being mistuned. Selection is by mean
per-token cross-entropy on \texttt{heldout\_4k}, a 4000-document set disjoint from
every training set; we never select on training loss, since on small repeated
corpora the training-loss optimum memorises and can generalise worse than the
untrained model.

\paragraph{Grid spacing.} We space grids by factors of two. The reference sweep
(GPT-2, $N=16$k, batch size 256, two epochs, both optimizers) measures how
held-out loss degrades with distance from the optimum: moving $2\times$ away costs
0.002--0.010 nats, $4\times$ costs 0.042, and $10\times$ costs 0.140. Retraining a
fixed configuration under a different seed moves held-out loss by about 0.001
nats, so two learning rates both within $2\times$ of the optimum are separated by
about seed noise. A finer grid --- steps of $1.4\times$, say --- would therefore be
resolving noise, while a coarser one would miss real mistuning. Steps of two are
the finest spacing the selection metric supports.

\paragraph{Initial grids.} For GPT-2 each group is swept at
$\{0.5\times, 1\times, 2\times\}$ its centre. Only one number in
Table~\ref{tab:lr-centres} is taken from our own measurements --- the shared
reference centre $2\times10^{-4}$, the reference configuration's optimum, applied
uniformly to every arm. Per-arm departures from it come from conventional
heuristics rather than from our data: batch-size arms shift by
$\sqrt{\mathrm{bs}/256}$ rounded to a factor of two, and model-size arms shift one
step down. If a heuristic is wrong the extension rule corrects it at the cost of
one extra run, so the prior only has to be roughly right to pay for itself.

For Qwen 2.5 1.5B we sweep a fixed grid of $\{1,2,5\}\times10^{-5}$ and
$\{1,2,5\}\times10^{-4}$ at every dataset size, rather than a three-point window
plus extensions; at $N=4$k and $8$k the $5\times10^{-4}$ point was not run, as
$2\times10^{-4}$ had already cost more than 0.18 nats against the optimum. At
$N=32$k, $5\times10^{-4}$ diverged. Every rung selects an interior point. We then
use a single peak learning rate of $5\times10^{-5}$ for the whole Qwen chain: it is
the per-rung optimum at every $N\ge16$k, and at $N=4$k and $8$k it is within
0.0021 and 0.0013 nats of the per-rung optimum ($2\times10^{-5}$) respectively,
against a seed noise of about 0.001. Holding the learning rate fixed keeps the
dataset-size axis a one-factor comparison, which a per-rung retune would break.

\paragraph{Extension.} If the lowest held-out loss falls on an endpoint of the
grid, we add one run a further factor of two out in that direction and re-check.
We stop after two extensions: an optimum more than $4\times$ from the centre
indicates something unexpected about the configuration, which warrants
investigation rather than more sweeping.

\paragraph{Ties.} If the best and second-best points differ by less than 0.002
nats --- twice the measured seed noise, i.e.\ the resolution limit of the
selection metric --- we keep the group's centre rather than the numerical winner.
This holds the learning rate fixed along an axis unless the data clearly demands
a change, so most comparisons remain one-factor. Two groups are decided this way,
both marked in Table~\ref{tab:lr-sweeps}: GPT-2 + Muon at batch size 16, where
$2.5\times10^{-5}$ beat the centre $5\times10^{-5}$ by 0.0002 nats, and the
logit-scale-0.5 group, where $4\times10^{-4}$ beat the centre $2\times10^{-4}$ by
0.0008.

\paragraph{Cost.} Sweep runs are training-only: no retrain banks and no
attribution, one held-out evaluation each. Most complete in under ten minutes;
the largest cost a few hours. The whole grid costs less than one retrain bank and
protects roughly forty of them.
```

## 3. Centres table

```latex
\begin{table}[t]
\centering
\caption{Grid centres for the GPT-2 learning-rate sweeps. Each group is swept at
$\{0.5\times, 1\times, 2\times\}$ its centre, then extended if an endpoint wins.}
\label{tab:lr-centres}
\begin{tabular}{lll}
\toprule
Axis & Centre & Reason \\
\midrule
Dataset size, 4k--64k & $2\times10^{-4}$ & no expected drift at fixed batch and epochs \\
Batch size 16--32 & $5\times10^{-5}$ & $\sqrt{\mathrm{bs}/256}$, rounded to a $2\times$ step \\
Batch size 64--128 & $1\times10^{-4}$ & same rule \\
Batch size 512 & $2\times10^{-4}$ & same rule, rounds back to the reference \\
Four epochs & $2\times10^{-4}$ & the grid's $1\times10^{-4}$ point covers a lower optimum \\
GPT-2 medium & $1\times10^{-4}$ & larger models conventionally prefer lower lr \\
Logit scale, weight decay, clipping & $2\times10^{-4}$ & these rarely move the optimum \\
\bottomrule
\end{tabular}
\end{table}
```

Note for the step-scaling arms (batch size 32 at $N\ge128$k): those centres were
set from an *established trend within this grid* --- the optimum halved as steps
doubled (1000 and 2000 steps chose $5\times10^{-5}$; 4000 chose $2.5\times10^{-5}$
at its low endpoint) --- so 128k was centred at $2.5\times10^{-5}$ and 256k at
$1.25\times10^{-5}$ rather than paying for two long extension runs. If those rows
appear in the paper, that centre choice should be stated, since it is the one
place a centre came from our own data beyond the reference value.

## 4. Sweep table

Generate the body with `python scripts/lr_sweep_table.py --latex`. Wrapper:

```latex
\begin{table}[t]
\centering\small
\caption{Learning-rate sweeps. Grid points are held-out-selected; the selected
value is in bold. $^{\dagger}$ diverged. Groups resolved by the tie rule keep the
grid centre rather than the numerical winner.}
\label{tab:lr-sweeps}
\begin{tabular}{llllr}
\toprule
Group & Optimizer & Grid & Selected & Held-out CE \\
\midrule
% <- generated body here
\bottomrule
\end{tabular}
\end{table}
```

## 5. Open items before this text is true as written

1. **`plan_adam_eps1e17_256k_bs32` and `plan_muon_eps1e17_256k_bs32` are frozen at
   $5\times10^{-5}$, which is not a point of their grid** ($6.25\times10^{-6}$,
   $1.25\times10^{-5}$, $2.5\times10^{-5}$; the third point is unmeasured). The
   adamw row already has a metasmoothness value measured at that lr. Both rows'
   notes claim the lr came from their tuning group, and it did not. Either finish
   the sweep and retrain at the winner, or drop these two points from the paper ---
   the claim "every selected learning rate is an interior point of its grid"
   is false while they are in. `scripts/lr_sweep_table.py` fails on both.
2. **The corpus sentence.** The main text says SmolLM2 is used "for all
   experiments", but the `london` arms are a different, distribution-shift corpus
   with their own sweeps, selecting $2\times10^{-4}$--$1.6\times10^{-3}$, i.e.\ up to
   $8\times$ the SmolLM2 optimum at matched $N$. If any london result appears in the paper the
   sentence needs a carve-out.
3. **Qwen 4k is 0.0001 nats outside the tie rule.** The chain runs at
   $5\times10^{-5}$ everywhere, but the $N=4$k sweep prefers $2\times10^{-5}$ by
   0.0021 nats --- just over the 0.002 threshold, where $N=8$k's 0.0013 is inside
   it. Either report the fixed-lr choice explicitly (the wording above does this,
   and is the honest and defensible option, since a per-rung retune would confound
   the dataset-size axis) or retrain the 4k rung at $2\times10^{-5}$. Do not write
   "we select the lowest held-out loss at every configuration" without one of the
   two. `scripts/lr_sweep_table.py` fails on this row.

4. **README row formatting.** The README scaling table prints `3e-05` for the
   64k/bs32 rows where `experiments.csv` holds `2.5e-05`. Cosmetic, but it will
   not match the appendix table if anyone cross-reads them.
