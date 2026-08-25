# adamw breaks on batch size. muon breaks on corpus. They are orthogonal.

Four cells measured, one pending, all at N=16k, 2 epochs, seed 42, each cell tuned
to its own lr against its own held-out set.

              bs256     bs16      drop from batch
    smollm2
      adamw   0.9930    0.9133    -0.080
      muon    0.9963    0.9939    -0.002
    london
      adamw   0.9867    0.9058    -0.081
      muon    0.8547    pending

    drop from corpus (bs256):  adamw -0.006    muon -0.142

## The dissociation

**adamw is batch-sensitive and corpus-blind.** Dropping bs256 to bs16 costs it
0.080 on smollm2 and 0.081 on london -- the same number twice, on corpora whose
starting losses differ by half a nat. Changing the corpus at fixed batch costs it
0.006.

**muon is the mirror image.** Batch costs it 0.002. Corpus costs it 0.142.

So the two optimizers fail at metasmoothness for different reasons, and neither
axis alone would have shown it. The smollm2 grid says muon is the robust one --
it holds ms above adamw at every N and barely notices batch size. That conclusion
survives only as long as the corpus stays close to pre-training.

## Why it matters for the questions

Question 5 asks whether muon and adamw differ in maintaining ms. They do, but not
as a ranking: which one is "better" depends entirely on which axis you move. Any
single-corpus comparison will get this wrong.

Question 1 asks what breaks ms. Batch size and corpus distance both do, but they
select different victims.

The pending cell (london muon bs16) decides whether the two effects compound. If
it lands near 0.85 the effects are independent -- batch does nothing to muon
regardless of corpus. If it lands well below, they interact, and the worst case
for ms is a distant corpus at small batch.

## Caveat

ms is not the same as attribution quality. On smollm2 muon has the HIGHER ms and
the LOWER MAGIC LDS (0.9963/0.8379 against adamw 0.9930/0.9411), so ms does not
rank the optimizers the way LDS does. Whether the london muon ms collapse drags
MAGIC with it is what the london muon bank is being built to answer.
