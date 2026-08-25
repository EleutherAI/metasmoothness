# The corpus does not lower ms for adamw. It does for muon.

Lucia's suspicion: every row reads ms 0.98+, which does not match her WikiText
results, and held-out loss moves only ~0.1 nats over a run -- so maybe smollm2 is
simply too close to GPT-2's pre-training distribution for fine-tuning to test
anything.

Measured against a genuinely distant corpus, the answer splits by optimizer.

    corpus     opt     lr      stock gpt2   fine-tuned   drop     ms
    smollm2    adamw   2e-4    3.4981       3.2572       0.241    0.9930
    smollm2    muon    2e-4    3.4981       3.2570       0.241    0.9964
    london     adamw   8e-4    4.0181       3.8397       0.178    0.9867
    london     muon    8e-4    4.0181       3.8394       0.178    0.8547

Identical config throughout: gpt2, bs256, 2 epochs, 125 steps, seed 42, same
tokenizer and 512-token chunks. Only the text differs, and each corpus got its own
lr sweep against its own held-out set.

## What this says

**adamw is unmoved.** 0.9930 on smollm2, 0.9867 on london. GPT-2 starts half a
nat worse on the 1800s corpus, so the distribution really is further out, and
adamw's ms barely notices. The "setup is too easy" hypothesis does not explain
adamw's high ms.

**muon breaks.** 0.9964 on smollm2, 0.8547 on london -- a 0.14 drop on the same
config, from the arm that was the STEADIER of the two on smollm2. On the smollm2
grid muon holds ms higher than adamw at every N; on london the ordering reverses
and the gap is an order of magnitude larger than anything the smollm2 grid shows.

And it is not a loss story: the two optimizers reach the same london loss to
within 0.0003 (3.8397 adamw, 3.8394 muon) from the same lr. Equal loss, equal
step budget, ms 0.9867 against 0.8547.

## Why it matters

The obvious next question is whether MAGIC follows ms down on the london muon row
the way it does elsewhere. On smollm2 the ms ordering does NOT predict the LDS
ordering -- muon has the higher ms and the LOWER MAGIC (0.9964/0.8379 against
adamw's 0.9930/0.9411) -- so this is the cleanest case yet for testing whether a
big ms drop drags MAGIC with it. The london bank is building for the adamw row;
the muon row needs one too.

Second: this is a corpus x optimizer interaction, which nothing in the smollm2
grid could have revealed. Any claim about muon holding ms as things scale is now
conditional on the corpus.

## Still open

bs16 on london, both arms, is running. Batch moves ms hard on smollm2 (0.9930 at
bs256 down to 0.9133 at bs16), so bs16 on london is where corpus, optimizer and
batch all meet. muon deadlocks on london above 16k, so the N axis there is adamw
only for now -- see notes/muon32k_hang.md.
