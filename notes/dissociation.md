# muon's ms failure needs BOTH a distant corpus and a large batch

Full 2x2, all at N=16k, 2 epochs, seed 42, each cell tuned to its own lr against
its own held-out set.

                  bs256     bs16      batch effect
    smollm2 adamw  0.9930    0.9133    -0.080
    smollm2 muon   0.9963    0.9939    -0.002
    london  adamw  0.9867    0.9058    -0.081
    london  muon   0.8547    0.9640    +0.109

    corpus effect   at bs256    at bs16
    adamw            -0.006      -0.008
    muon             -0.142      -0.030

## The interaction

**adamw is simple.** Batch costs it ~0.080 on both corpora; corpus costs it
~0.007 at both batch sizes. The two axes do not interact at all, and only one of
them matters.

**muon is not.** Its corpus penalty is 0.142 at bs256 and 0.030 at bs16 -- five
times smaller. Equivalently, shrinking the batch *raises* muon's ms on london by
0.109, the opposite sign to what batch does to adamw everywhere and to muon on
smollm2.

So muon only breaks in one corner: **large batch AND a distant corpus**. Neither
alone does it. I predicted before measuring that the effects would be independent
and this cell would land near 0.85; it landed at 0.9640, which refutes that.

## What it means for the questions

Question 5 (do muon and adamw differ in maintaining ms): yes, but the answer is
not a ranking and not even a pair of main effects. adamw has one failure mode
(batch). muon has one failure mode that is a *conjunction*. A grid that varies one
axis at a time reports muon as uniformly the more robust optimizer, which is what
the smollm2 grid did, and that is wrong outside the corner it never visited.

Question 1 (what breaks ms): batch size breaks adamw. Batch-and-corpus together
break muon. Nothing here breaks both.

## Caveat, unchanged

ms does not rank the optimizers the way LDS does. On smollm2 muon has the higher
ms and the LOWER MAGIC (0.9963/0.8379 against adamw's 0.9930/0.9411). Whether the
london muon ms collapse at bs256 drags MAGIC down with it is what the london muon
bank is being built to answer -- it is in MAGIC scoring now.
