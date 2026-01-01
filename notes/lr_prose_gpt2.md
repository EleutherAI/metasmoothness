# Prose: learning-rate selection for the plotted GPT-2 data

Scope: only the GPT-2 rows that appear in a figure --- the corpus-size axis at
batch size 256 (AdamW $N=4$k--512k, Muon $N=4$k--256k), the batch-size axis at
$N=16$k (both optimizers, batch 16--512), and the AdamW variant panel at
$N=16$k. Qwen and the `london` corpus are not covered here.

Regenerate the underlying numbers with `python scripts/lr_sweep_table.py`.

---

## Draft

We tune the peak learning rate separately for every configuration whose
optimization problem differs from our reference configuration --- that is, for
every change of dataset size, batch size, epoch count, optimizer, or model.
Selection is always by mean per-token cross-entropy on a held-out set of 4000
documents disjoint from every training set; we never select on training loss,
since on small repeated corpora the training-loss optimum memorises and can
generalise worse than the untrained model.

We initially use a learning rate grid of $\{1\times10^{-4}, 2\times10^{-4},
4\times10^{-4}\}$ for GPT-2 fine-tuning, obtained by sweeping the reference
configuration ($N=16$k documents, batch size 256, two epochs) over the five
values $\{1\times10^{-4}, 2\times10^{-4}, 4\times10^{-4}, 8\times10^{-4},
2\times10^{-3}\}$. Both AdamW and Muon select $2\times10^{-4}$ there, and we take
that value as the centre of a three-point grid $\{0.5\times, 1\times, 2\times\}$
for every other configuration. We space grids by factors of two because the
reference sweep fixes both ends of the useful range: moving a factor of two away
from the optimum costs between 0.002 and 0.010 nats of held-out loss, a factor of
four costs 0.042, and a factor of ten costs 0.140, while retraining a fixed
configuration under a different seed moves held-out loss by about 0.001 nats. A
finer grid would therefore resolve seed noise rather than mistuning, and a
coarser one would miss real mistuning.

Only that one centre, $2\times10^{-4}$, is taken from our own measurements. Where
we shift a grid away from it we do so on conventional grounds rather than on our
own data. On the batch-size axis we scale the centre by $\sqrt{\mathrm{bs}/256}$
and round to a factor of two, giving centres of $5\times10^{-5}$ at batch sizes
16 and 32, $1\times10^{-4}$ at 64 and 128, and $2\times10^{-4}$ at 512. For GPT-2
medium we shift one step down to $1\times10^{-4}$, as larger models under standard
parameterisation conventionally prefer lower learning rates. At $N\ge128$k we
likewise centre at $1\times10^{-4}$, reflecting the drift towards lower learning
rates already visible at smaller sizes rather than letting a long and expensive
run land on a grid endpoint. The weight-decay, gradient-clipping, epoch-count and
logit-scale groups keep the reference centre of $2\times10^{-4}$; those groups
exist to verify that the optimum has not moved, not to hunt for it.

We extend a grid whenever the lowest held-out loss falls on one of its endpoints,
adding a single further point a factor of two beyond that endpoint and
re-checking. We stop after two extensions, on the grounds that an optimum more
than a factor of four from the centre indicates something unexpected about the
configuration and warrants investigation rather than further sweeping. Eight of
the plotted configurations required an extension. AdamW at $N=4$k and $N=64$k,
Muon at $N=64$k, and the four-epoch variant each extended downwards by one step
to $5\times10^{-5}$; AdamW at $N=128$k and $N=256$k each extended upwards by one
step to $4\times10^{-4}$; and Muon at $N=4$k extended upwards by one step to
$8\times10^{-4}$. The logit-scale-0.25 variant is the only configuration that
required both permitted extensions, moving upwards through $8\times10^{-4}$ to
$1.6\times10^{-3}$. The three largest configurations
were swept on a wider grid from the outset rather than by extension: AdamW at
$N=512$k over seven values from $2.5\times10^{-5}$ to $1.6\times10^{-3}$, and Muon
at $N=256$k and $N=512$k over six and five values respectively. No configuration
on the batch-size axis required an extension.

Where the best and second-best points of a grid differ by less than 0.002 nats
--- twice the measured seed noise, and so the resolution limit of the selection
metric --- we keep the grid centre rather than the numerical winner. This holds
the learning rate fixed along an axis unless the data clearly demand a change, so
that most comparisons remain one-factor. Two of the plotted configurations are
decided this way: Muon at batch size 16, where $2.5\times10^{-5}$ beats the centre
$5\times10^{-5}$ by 0.0002 nats, and the logit-scale-0.5 variant, where
$4\times10^{-4}$ beats the centre $2\times10^{-4}$ by 0.0008 nats. Both keep the
centre.

The resulting learning rates are as follows. On the corpus-size axis at batch size
256, AdamW uses $1\times10^{-4}$ at $N=4$k and $N=64$k, $2\times10^{-4}$ at $N=8$k,
$16$k, $32$k, $128$k and $256$k, and $8\times10^{-4}$ at $N=512$k; Muon uses
$2\times10^{-4}$ at $N=8$k, $16$k, $32$k and $256$k, and $1\times10^{-4}$ at $N=64$k
and $N=128$k. On the batch-size axis at $N=16$k both optimizers use
$5\times10^{-5}$ at batch sizes 16 and 32, $1\times10^{-4}$ at 64 and 128,
$2\times10^{-4}$ at 256, and $2\times10^{-4}$ at 512. Among the $N=16$k variants,
the weight-decay, gradient-clipping and logit-scale-0.5 groups keep
$2\times10^{-4}$; four epochs and GPT-2 medium select $1\times10^{-4}$; and
logit scale 0.25 selects $8\times10^{-4}$, the only variant whose optimum moves
by more than a factor of two.

---

## Two deviations that this prose currently papers over

Decide how to handle each before the paragraph above is submitted; as written it
is not yet true of the plotted data.

1. **Muon at batch size 512 was never swept.** There is no `tune_muon_16k_bs512`
   group in `tuning.csv` --- only the AdamW one. The plotted Muon bs512 point runs
   at $2\times10^{-4}$, transferred from AdamW. The transfer is reasonable (the
   two optimizers select identical learning rates at every other batch size on
   this axis: $5\times10^{-5}$ at 16 and 32, $1\times10^{-4}$ at 64 and 128,
   $2\times10^{-4}$ at 256), but it is a transfer, not a measurement, and the
   sentence "we tune separately for every change of optimizer" excludes it.
   Either run the three-point sweep --- it is minutes of compute --- or state the
   transfer explicitly.

2. **The plotted Muon $N=4$k point is not the sweep winner.** The sweep selects
   $4\times10^{-4}$ (held-out 3.3114) over $2\times10^{-4}$ (3.3138), a margin of
   0.0024 nats that just clears the 0.002 tie threshold. The figure nevertheless
   plots `plan_muon_eps1e17_4k_bs256_lr2e-4`, a re-run at $2\times10^{-4}$;
   `PREFER` in `scripts/scaling_plot_mpl.py` selects it over the $4\times10^{-4}$
   row, and that row's note in `experiments.csv` gives the reason as the
   metasmoothness value ("replacing the collapsed lr 4e-4 point"). Selecting a
   learning rate on a downstream metric is not the protocol the paragraph above
   describes, and it should not be reported as though it were.

   The defensible framing, if the point stays: $2\times10^{-4}$ is the grid centre
   and sits 0.0024 nats from the numerical winner, i.e.\ essentially at the
   resolution limit of the selection metric, so this is a borderline application
   of the tie rule. That is checkable and honest. The alternative is to plot the
   $4\times10^{-4}$ row and report the metasmoothness difference between the two
   learning rates as a finding in its own right, which is arguably the more
   interesting option --- an attribution result that moves this much under a
   within-noise learning-rate change is worth stating rather than resolving away.
