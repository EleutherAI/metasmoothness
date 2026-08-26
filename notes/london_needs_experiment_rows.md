# The london ablation is blocked by an explicit design rule, not an oversight

Follow-up to the earlier note. I went to add the missing experiment rows and hit
two guards in build_experiments_csv.py. The first was mine to satisfy; the second
is a decision that is not mine to overturn.

## Guard 1 -- satisfied

    assert r["run_id"] in TUNED_LR, "tuning group complete but its selection is
    not recorded in TUNED_LR - the wrong-lr failure mode"

Correct and easily met: the winners are measured. Recorded them, all interior,
all scored against london_heldout_4k rather than the smollm2 held-out set:

    16k bs256  8e-4     32k bs256  8e-4     64k bs256  1.6e-3     16k bs16  2e-4

## Guard 2 -- STOPS HERE

    assert r["dataset"] == "smollm2",
        "non-smollm2 row admitted: ... paper runs use the SmolLM2 pipeline only
         (WikiText does not scale)"

This is an explicit design rule about what belongs in the paper grid. Adding
london experiment rows contradicts it directly, so the rows are NOT added and the
change is reverted. The standing instruction is to trust the design documents and
raise the conflict rather than route around it.

Worth noting for the decision, though: the stated reason is that WikiText does
not scale. london-llm-1800 is not WikiText. It is purpose-built by
scripts/prep_london.py, packed to 512-token chunks, and already exists at 16k,
32k, 64k and 128k with a zero-overlap held-out set of 4000 documents. It scales
to the same sizes the smollm2 arm uses. So the rule's justification may not cover
this corpus even though its wording does.

## What is waiting on the answer

Roughly 30 completed tuning runs and 8 settled lrs. Also the two london runs that
already produced ms -- 0.9867 adamw against 0.8547 muon at 16k -- which are a far
larger optimizer gap than anything in the smollm2 grid, and which currently live
only as run directories and a note.

The ablation exists to test whether ms sits at 0.98-0.99 everywhere because
smollm2 is too close to GPT-2 pre-training. That test needs london ms and LDS in
the same table as the smollm2 rows, and this rule is what prevents it.

Two ways forward, both Lucia's call:

  1. relax the assert to allow dataset in ("smollm2", "london"), and let the
     london arm into the grid as a first-class axis
  2. keep the grid smollm2-only and carry london as a separate table, in which
     case the tuning has done its job and the ablation needs its own home

scripts/gen_experiment_run.py has been taught the london corpus either way -- it
resolves london_<n>k.hf from the mirror instead of train_<n>k.hf. That change is
harmless on its own and is what stops a london row silently training on smollm2,
which is exactly what happened to the london TUNING configs earlier today.
