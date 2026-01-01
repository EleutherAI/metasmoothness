# EK-FAC reads 0.42 on the model where MAGIC reads zero

gpt2-medium, same bank, same 100 subsets, same 20 queries:

    MAGIC LDS   -0.0407  [-0.0980, +0.0155]
    EK-FAC LDS   0.4189  [ 0.3707,  0.4645]

MAGIC has no signal at all on this model -- its interval contains zero, and its
filter delta agrees (-0.00018 against a random control of 0.00017). EK-FAC lands
at 0.4189, which is indistinguishable from what it scores on every other row.

## The band

Every EK-FAC LDS measured so far, across models, batch sizes and corpora:

    plan_adam_eps1e17_16k_ep4         0.4730
    plan_muon_eps1e17_16k_bs128       0.4635
    plan_adam_eps1e17_16k_bs32        0.4586
    plan_adam_eps1e17_16k_bs128       0.4551
    plan_muon_eps1e17_16k_bs16        0.4276
    plan_adam_eps1e17_16k_bs64        0.4239
    plan_adam_eps1e17_16k_gpt2-medium 0.4189   <- MAGIC is ZERO here
    plan_adam_eps1e17_32k_bs256       0.4127
    plan_muon_eps1e17_32k_bs256       0.4044

Range 0.404 to 0.473, a spread of 0.069. MAGIC over the same rows ranges from
-0.04 to 0.95.

## Two readings, and they matter differently

**EK-FAC is genuinely robust where MAGIC is not.** If real, this is the more
interesting claim in the project: a scorer that keeps working at a model size
where MAGIC collapses entirely.

**Or 0.42 is EK-FAC's floor.** A number it returns whenever it is not really
tracking anything, in which case the band is an artifact and EK-FAC's LDS is not
measuring what MAGIC's LDS measures.

The gpt2-medium row is what makes this decidable, and it is why this row matters
more than another mid-range point. The test is its filter delta: if EK-FAC's 0.42
reflects genuine attribution, removing the documents EK-FAC ranks influential
should move the query loss well clear of the random control, the way MAGIC's
0.9411 row does at 400x. If the delta is at random-control level while the LDS
says 0.42, then 0.42 is a floor and the EK-FAC arm of the correlation has been
measuring noise.

That run has not been done -- gpt2-medium has a MAGIC filter delta but no EK-FAC
one. It is now the single most informative job in the queue.

## Why the EK-FAC correlation is flat

This also explains +0.186 [-0.131, +0.497] over 17 rows directly. A predictor
confined to a 0.069-wide band cannot correlate strongly with anything that varies
over a full unit interval, whatever the sample size. More rows will tighten the
interval and leave the estimate where it is.
