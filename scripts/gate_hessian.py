"""Validate a bergson EK-FAC Hessian dir (kfac or kfac.part), optionally against a reference.
Factors are ROW-SHARDED across rank files (shard_i holds rows i*d/R..(i+1)*d/R of every module);
reassemble by concatenating along dim 0. usage: gate_hessian.py <kfac_dir> [<ref_kfac_dir>]"""
import sys,os,glob,torch,numpy as np
from safetensors import safe_open
H=sys.argv[1]; R=sys.argv[2] if len(sys.argv)>2 else None
def tp(d):
    t=torch.load(f"{d}/total_processed.pt",map_location="cpu",weights_only=False); return float(t.item() if hasattr(t,"item") else t)
def modules(d,kind):
    parts={}
    for p in sorted(glob.glob(f"{d}/{kind}/shard_*.safetensors"),key=lambda s:int(s.split("_")[-1].split(".")[0])):
        with safe_open(p,"pt") as f:
            for k in f.keys(): parts.setdefault(k,[]).append(f.get_tensor(k).float())
    return parts
fail=[]
n_h=tp(H); print(f"total_processed: {n_h:.0f}" + (f" (ref {tp(R):.0f}, ratio {n_h/tp(R):.2f})" if R else ""))
for kind in ("activation_sharded","gradient_sharded","eigenvalue_sharded","eigenvalue_correction_sharded","factor_eig_a","factor_eig_g"):
    P=modules(H,kind)
    if not P: print(f"{kind}: absent"); continue
    nshard=len(next(iter(P.values()))); bad={"nonfinite":0,"allzero":0,"asym":0,"negdiag":0,"nonpos":0,"replica_mismatch":0}; stat={}
    for k,parts in P.items():
        if kind.startswith("factor_eig"):
            v=parts[0]
            if any(not torch.equal(v,q) for q in parts[1:]): bad["replica_mismatch"]+=1
        else: v=torch.cat(parts,0)
        if not torch.isfinite(v).all(): bad["nonfinite"]+=1
        if v.abs().max()==0: bad["allzero"]+=1
        if kind in ("activation_sharded","gradient_sharded"):
            if v.shape[0]!=v.shape[1]: bad["asym"]+=1
            else:
                if (v-v.T).abs().max()>1e-3*v.abs().max(): bad["asym"]+=1
                if torch.diagonal(v).min()<0: bad["negdiag"]+=1
                stat[k]=torch.diagonal(v).sum().item()/n_h
        else:
            if v.min()<=0: bad["nonpos"]+=1
            stat[k]=v.mean().item()
    print(f"{kind}: modules={len(P)} shards={nshard} "+" ".join(f"{a}={b}" for a,b in bad.items()))
    if any(bad.values()): fail.append(kind)
    if R and glob.glob(f"{R}/{kind}/shard_*.safetensors"):
        Q=modules(R,kind); n_r=tp(R); rs=[]
        for k,parts in Q.items():
            v=parts[0] if kind.startswith("factor_eig") else torch.cat(parts,0)
            r=torch.diagonal(v).sum().item()/n_r if kind in ("activation_sharded","gradient_sharded") else v.mean().item()
            if k in stat and r: rs.append(stat[k]/r)
        rs=np.array(rs); print(f"   vs ref (per-module, count-normalised): common={len(rs)}/{len(Q)} ratio med={np.median(rs):.2f} p5={np.percentile(rs,5):.2f} p95={np.percentile(rs,95):.2f} min={rs.min():.2f} max={rs.max():.2f}")
        if len(rs)!=len(Q) or rs.max()>20 or rs.min()<0.05: fail.append(kind+":ref")
print("HESSIAN GATE:", ("FAIL "+str(fail)) if fail else "PASS")
