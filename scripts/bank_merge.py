"""Offline consume-step for bergson filter shards run with num_subsets: 0.
Evaluates a bank's retrained/{base,subset_i} on the run's query set as bergson's
bank branch does (fp32, eager, mean CE over the row's label tokens; one query per row), then writes random_filter.csv + filter_summary.csv for the run.
usage: bank_merge.py --run <filter run dir> --bank <bank dir> --query <hf dataset> [--out <dir>] [--force]"""
import argparse,csv,json,os,statistics as st,torch,torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM
ap=argparse.ArgumentParser(); ap.add_argument("--run",required=True); ap.add_argument("--bank",required=True); ap.add_argument("--query",required=True)
ap.add_argument("--out"); ap.add_argument("--force",action="store_true"); ap.add_argument("--batch",type=int,default=8); a=ap.parse_args()
out=a.out or a.run; os.makedirs(out,exist_ok=True)
if os.path.exists(f"{out}/filter_summary.csv") and not a.force: raise SystemExit(f"{out}/filter_summary.csv exists; use --force")
subsets=json.load(open(f"{a.bank}/subsets.json")); q=load_from_disk(a.query); dev="cuda" if torch.cuda.is_available() else "cpu"
num_docs=len(q)  # bergson's query unit is the ROW (one query per row); doc_ids are corpus ids, not query indices
@torch.no_grad()
def per_doc_losses(mdir):
    m=AutoModelForCausalLM.from_pretrained(mdir,dtype=torch.float32,attn_implementation="eager").to(dev).eval(); out=[]
    for i in range(len(q)):
        ids=torch.tensor(q[i]["input_ids"])[None].to(dev); lg=m(input_ids=ids).logits; sl=ids[:,1:]
        out.append(F.cross_entropy(lg[:,:-1].flatten(0,1).float(),sl.flatten(),reduction="none").mean().item())
    del m; torch.cuda.empty_cache(); return torch.tensor(out)
base=per_doc_losses(f"{a.bank}/retrained/base"); subs=[per_doc_losses(f"{a.bank}/retrained/subset_{i}") for i in range(len(subsets))]
prop=list(csv.DictReader(open(f"{a.run}/filter_proponents.csv")))
nq=len(prop); assert nq==len(q), f"{nq} proponent rows but the query dataset has {len(q)} rows: wrong --query slice for this run?"
assert sorted(int(r["query"]) for r in prop)==list(range(nq)), "proponent rows are not local query indices 0..n-1"
print(f"queries={nq} bank subsets={len(subsets)} sizes={[len(s) for s in subsets]} bank-base loss mean={base[:nq].mean():.4f} recorded in-job baseline mean={st.mean(float(r['baseline_loss']) for r in prop):.4f}")
with open(f"{out}/random_filter.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["subset","query","n_removed","baseline_loss","filtered_loss","loss_change"])
    for i,s in enumerate(subs):
        for qi in range(nq): w.writerow([i,qi,len(subsets[i]),float(base[qi]),float(s[qi]),float(s[qi]-base[qi])])
with open(f"{out}/filter_summary.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["query","n_removed","filter_change","random_mean","random_sd","random_n","rank"])
    for r in prop:
        qi=int(r["query"]); fc=float(r["loss_change"]); rc=[float(s[qi]-base[qi]) for s in subs]
        w.writerow([qi,r["n_removed"],fc,st.mean(rc),st.stdev(rc) if len(rc)>1 else 0.0,len(rc),1+sum(x>fc for x in rc)])
qld=[float(r["loss_change"])-st.mean(float(s[int(r["query"])]-base[int(r["query"])]) for s in subs) for r in prop]
print(f"wrote {out}/filter_summary.csv: QLD mean={st.mean(qld):+.4f} pos={sum(x>0 for x in qld)}/{nq}")
