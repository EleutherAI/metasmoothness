# lotus-0: custom GPT-2 implemented and uploaded (D10)

`EleutherAI/gpt2-custom` (private) is live with remote code. arch_mod="none" is
bit-identical to stock gpt2 (tested at 124M); variants via
from_pretrained(..., arch_mod="qk_norm"|"preact_layernorm", trust_remote_code=True).
Code on bergson branch feat/gpt2-custom (pushed; PR awaits gh auth, along with
fix/nccl-metadata and your remove-torch-upper-cap). Arch tuning groups stay blocked
until the D10 dynamic check runs in the pinned env - lotus-0 will run it (one
16k fine-tune of arch_mod=none vs the stock anchor heldout 3.2572) as soon as the
env boots. Do not claim arch rows before DECISIONS D10 marks them open.

## Update: PRs open, model public

gh auth landed on lotus-0. EleutherAI/bergson PRs: #426 (runtime NCCL version in
run metadata), #427 (gpt2-custom), #428 (your remove-torch-upper-cap - fetched
from /mnt/ssd-2/lucia/bergson-main and pushed on your behalf). EleutherAI/gpt2-custom
is now PUBLIC per Lucia (open-source-everything policy).
