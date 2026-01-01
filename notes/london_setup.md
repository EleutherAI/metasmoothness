# The london row needs london queries, not smollm2 ones

Cloning the anchor experiment config gets you `query_20.hf`, which is smollm2
text. Running that against a london-trained model measures attribution for
OUT-OF-DOMAIN queries, which is a different question from the one the rest of the
grid answers and would not be comparable to any existing row.

`london_query_20.hf` is the last 20 chunks of `london_heldout_4k.hf`, so it is
disjoint from every london training set by construction (the held-out set was
packed from source rows 40000+ and verified zero-overlap against london_128k).

Same reasoning applies to lr selection: `scripts/prep_london_heldout.py` exists
because selecting on the smollm2 held-out set picks whichever lr best fits the
distribution this corpus exists to move away from. It changed the answer -- the
london adamw winner is 8e-4, four times the smollm2 winner.

## Checklist for any new corpus

    dataset      london_{16,32,64,128}k.hf   packed to 512, gpt2 tokenizer, nested
    heldout      london_heldout_4k.hf        disjoint, used for lr selection
    queries      london_query_20.hf          disjoint, used for attribution
    permissions  0777 on the dataset dirs    bergson writes a temp file INSIDE them

That last one is not cosmetic: the dirs were created 0755 and every launch died
with PermissionError on london_16k.hf/tmp*. The smollm2 dirs are 0777, which is
why this had never come up.
