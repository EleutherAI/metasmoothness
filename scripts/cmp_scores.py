"""Compare in-progress 64k scores (scores.part or scores) against 32k scores on the index-aligned first 32000 docs."""
import json,sys,numpy as np
from pathlib import Path
E=Path("/data/anon/paper_runs/experiments")
def load(d):
    info=json.load(open(d/"info.json")); dd=info["dtype"]
    dt=np.dtype({"names":dd["names"],"formats":[np.dtype(f) for f in dd["formats"]],"offsets":dd["offsets"],"itemsize":dd["itemsize"]})
    mm=np.memmap(d/"scores.bin",dtype=dt,mode="r",shape=(info["num_rows"],))
    S=np.stack([np.asarray(mm[f"score_{j}"],dtype=np.float64) for j in range(info["num_scores"])])
    W=np.stack([np.asarray(mm[f"written_{j}"]) for j in range(info["num_scores"])]).astype(bool)
    return S,W
d64=E/"qwen15b_64k_bs256/ekfac_scores_64k"; p=d64/"scores" if (d64/"scores/scores.bin").exists() else d64/"scores.part"
S64,W64=load(p); S32,W32=load(E/"qwen15b_32k_bs256/ekfac_scores/scores")
print(f"64k source: {p.name} shape {S64.shape} written frac per query: {np.round(W64.mean(1),2).tolist()}")
def rank(a): return np.argsort(np.argsort(a))
for q in range(20):
    m=W64[q,:32000]&W32[q,:32000]
    if m.sum()<1000: print(f"q{q}: only {m.sum()} aligned docs written yet"); continue
    a,b=S64[q,:32000][m],S32[q,:32000][m]
    rho=np.corrcoef(rank(a),rank(b))[0,1]; k=max(40,m.sum()//100)
    ta=set(np.argsort(-a)[:k]); tb=set(np.argsort(-b)[:k])
    print(f"q{q}: n={m.sum()} spearman={rho:.3f} pearson={np.corrcoef(a,b)[0,1]:.3f} top-{k} overlap={len(ta&tb)/k:.2f} std64={a.std():.3g} std32={b.std():.3g} finite={np.isfinite(a).all()}")
