# The corpus is not why ms is high, and a permissions trap that will bite you

## Result

Lucia's standing worry was that every row reads ms 0.98+ because smollm2 sits too
close to GPT-2's pre-training distribution -- i.e. the setup is too easy to test
anything. Measured against a genuinely distant corpus, that does not hold.

    corpus     lr      stock gpt2   fine-tuned   drop     ms
    smollm2    2e-4    3.4981       3.2572       0.241    0.9930
    london     8e-4    4.0181       3.8397       0.178    0.9867

Same config throughout: gpt2, bs256, 2 epochs, 125 steps, seed 42, same tokenizer
and 512-token chunks. Only the text differs. GPT-2 starts half a nat worse on the
1800s corpus, so the distribution really is further out, and ms still lands at
0.9867.

Both optimizers settle at the same lr on london (8e-4) and to within 0.0003 the
same loss -- 3.8397 adamw, 3.8394 muon. That is NOT what the smollm2 grid shows,
where muon holds ms higher than adamw at every N.

Running now: london at bs16 (both optimizers), since batch is the axis known to
move ms hard here (0.9930 at bs256 down to 0.9133 at bs16 on smollm2); london 32k
for the N question; and a retrain bank for london 16k, which is what will say
whether LDS and the filter delta shift on a shifted corpus.

## If you add a corpus, it needs four things

    dataset      london_{16,32,64,128}k.hf   512-token chunks, gpt2 tokenizer, nested
    heldout      london_heldout_4k.hf        disjoint -- lr selection uses THIS
    queries      london_query_20.hf          disjoint -- attribution uses THIS
    permissions  0777 on the dataset dirs    bergson writes a temp file INSIDE them

Selecting lr on the smollm2 heldout would pick whatever fits the distribution you
are trying to leave. It changed the answer here: london wants 8e-4, four times
smollm2's. Same for queries -- cloning the anchor config inherits smollm2
query_20.hf, which measures attribution for out-of-domain queries.

## The permissions trap

`umask 022` does NOT fix the 1000/1001 uid split. safetensors saves through a
temp file and Python creates temp files 0600 regardless of umask, so
`model.safetensors` lands 0600 and the other uid gets FileNotFoundError while
`ls` shows the file. This cost four evaluations before I found it.

Two fixes are in:

* `scripts/fix_perms.py` -- run from a node of EACH uid, only the owner can chmod
* `scripts/launch_tuning_pair.sh` -- chmods a+rX after the run and drops its claim

That launcher also refuses to start a run whose model already exists. Tuning
configs carry `overwrite: true`, so relaunching a finished run rmtree's it; the
muon london 1e-4 point was saved from exactly that only by a cross-uid
PermissionError, which is luck rather than a guard.
