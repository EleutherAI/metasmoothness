# How much of an ms value is the direction it was probed along?

Every metasmoothness value in experiments.csv and london.csv was measured at
`direction_seed 0`. ms perturbs the data weights along ONE random direction v and
scores the agreement of three trainings at weights 1, 1+h*v and 1+2h*v. So a
single ms number confounds two things: how smooth the configuration actually is,
and which v happened to be drawn.

Nothing measured so far separates them. That is tolerable while ms values are
being compared coarsely -- 0.99 against 0.91 is unlikely to be a direction
artifact -- and it is not tolerable for the one comparison the london arm now
rests on.

## The number under test

    london16k_bs256_muon = 0.8547

It is the only cell in the london/smollm2 x adamw/muon x bs16/bs256 table that
breaks the pattern:

                     adamw    muon
    london  bs16     0.9058   0.9640
    london  bs256    0.9867   0.8547     <-- this one
    smollm2 bs16     0.9133   0.9939
    smollm2 bs256    0.9930   0.9964

Both corpora look alike at bs16. Both optimizers prefer the larger batch on
smollm2, and adamw does on london too. Only london muon reverses. The reading
that "the corpus was hiding an optimizer difference" is carried entirely by that
single value, drawn from a single v.

## The design

Six probes, a 2x2 of corpus x optimizer with multiple directions each:

    london16k_bs256_muon_seed1, _seed2       london muon, v=1 and v=2
    london16k_bs256_adamw_seed1              london adamw, v=1
    sm_muon_eps1e17_16k_bs256_seed1, _seed2  smollm2 muon, v=1 and v=2
    sm_adamw_eps1e17_16k_bs256_seed1         smollm2 adamw, v=1

Only `direction_seed` and `run_path` differ from the seed-0 runs. Same data, lr,
optimizer, batch, fd_step 0.1, world size 2 -- anything else moving would make
the comparison meaningless. scripts/gen_ms_seeds.py enforces that by copying the
finished run's own ms.yaml rather than regenerating one.

The smollm2 arm is not optional. Measuring london's spread without knowing the
normal spread would answer nothing, which is what the first three probes alone
would have done.

## How to read the outcome

  * smollm2 muon also swings ~0.1 -> ms is direction-noisy at this setting;
    0.8547 says nothing about the corpus, and the ms column generally should not
    be trusted to the second decimal
  * smollm2 stable, london muon not -> the corpus genuinely makes muon's
    smoothness direction-dependent, which is a sharper claim than the one it
    replaces
  * both stable -> 0.8547 stands and the 0.13 optimizer gap on london is real

Whatever comes back, this is also the first evidence about how much any single ms
value in the grid should be trusted, since all of them share seed 0.
