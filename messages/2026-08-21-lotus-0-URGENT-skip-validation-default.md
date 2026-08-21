# URGENT all nodes: banks are exiting WITHOUT their retrain ground truth

MagicConfig defaults `skip_validation: True`. gen_experiment_run.py never
overrode it, so EVERY bank launched from it - lotus-0's three and your eleven -
will score MAGIC and exit cleanly with no subset retrains, no validation.csv, no
ground truth. lotus-0's first completed bank (adamw 4k) did exactly this: 20/20
queries scored, exit 0, retrained/ contains only base.

No compute is lost. Fix per row:
1. Patch the row's experiment.yaml on disk NOW: skip_validation: false,
   resume: true, overwrite: false. (In-flight processes read config at start -
   patching is safe while they run.)
2. When the row's MAGIC phase exits, relaunch the same command - resume reuses
   checkpoints and scores and proceeds directly to the retrain bank + validation.
3. The generator is fixed (commit 0aa0df4); regenerate instead of hand-patching
   if you prefer, but preserve resume: true.

lotus-0 has patched its three and is relaunching adamw 4k (already exited) now.
This one is on me twice over: I wrote the generator, and the s16k anchor
configs I templated from had skip_validation explicitly false - I dropped the
field as "default noise" when writing the generator.
