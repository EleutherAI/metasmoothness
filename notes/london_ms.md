# The corpus was not the reason ms is high

Lucia's suspicion: every row reads ms 0.98+, which does not match her WikiText
results, and held-out loss moves only ~0.1 nats over a run -- so maybe smollm2 is
simply too close to GPT-2's pre-training distribution for fine-tuning to test
anything.

Measured against a genuinely distant corpus, that does not hold.

    corpus     lr      stock gpt2   fine-tuned   drop     ms
    smollm2    2e-4    3.4981       3.2572       0.241    0.9930
    london     8e-4    4.0181       3.8397       0.178    0.9867

Identical config otherwise: gpt2, bs256, 2 epochs, 125 steps, seed 42, same
tokenizer and 512-token chunks. Only the text differs, and each corpus got its own
lr sweep against its own held-out set (london wants 8e-4, four times smollm2's).

london IS further from pre-training -- gpt2 starts half a nat worse on it -- and
ms still comes out 0.9867. So high ms is not an artifact of the fine-tuning
corpus resembling the pre-training corpus.

## What that leaves

The WikiText discrepancy has to come from somewhere else: the step budget (125
steps is small), the batch size, the model, or something in how ms is computed
here versus there. Batch is the one axis already known to move ms hard on this
setup -- 0.9930 at bs256 down to 0.9133 at bs16 -- so a london run at small batch
is the obvious next probe, and it is cheap.

Not yet answered: whether the LDS and proponent-filter numbers shift on london.
The bank for this row is building, which is what will say.
