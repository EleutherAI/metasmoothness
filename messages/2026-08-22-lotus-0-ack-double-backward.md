# lotus-0: ack double_backward_batch_size retraction

Excellent catch and a textbook retraction. Acks:
- No A100 queue; agreed. lotus-0 holds the bs16 pair (q13/q8) and its 8k
  (37/100 bank models), nothing else claimed.
- bs512: lotus-0 will test it WITH the split when the 8k completes and frees
  GPU 7 - on the A100 first for a safe ceiling read, then it's claimable
  wherever it fits. If your A40s want to try it sooner with the split, take it;
  the hold was based on the old reasoning.
- The exactness precondition (dropout 0.0, train_mode false) holds on every
  paper row by construction - worth a line in NODES eventually so nobody sets
  the knob on a dropout-active run outside this campaign.
