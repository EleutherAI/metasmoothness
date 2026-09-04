"""Qwen2.5-1.5B counterpart of figures/filter_scaling.png: (a) top-1% / 5% / 10% proponent filters, (b) top-40,
QLD = filter_change - random_mean, 95% CI over 20 queries, x = training tokens (2 epochs x N docs x 512 tokens)."""
import csv,os,math,statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
E="/mnt/ssd-2/lucia/paper_runs/experiments"; NS=[4000,8000,16000,32000,64000]
SER=[("Top 1%","filter_proponents_ekfac","#2a78d6"),("Top 5%","filter_prop5pct_ekfac","#1baf7a"),("Top 10%","filter_prop10pct_ekfac","#eb6834")]
tokens=lambda n: 2*n*512
def qld(n,d):
    p=f"{E}/qwen15b_{n//1000}k_bs256/{d}/filter_summary.csv"
    if not os.path.exists(p): return None
    rows=list(csv.DictReader(open(p)))
    if len(rows)<20: return None
    q=[float(r["filter_change"])-float(r["random_mean"]) for r in rows]
    return st.mean(q), 1.96*st.stdev(q)/math.sqrt(len(q))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,3.8),dpi=200,sharey=True)
for lab,d,col in SER:
    pts=[(tokens(n),)+qld(n,d) for n in NS if qld(n,d)]
    if pts: ax1.errorbar([p[0] for p in pts],[p[1] for p in pts],yerr=[p[2] for p in pts],fmt="o-",color=col,capsize=3,ms=4,lw=1.5,label=lab)
pts=[(tokens(n),)+qld(n,"filter_top40_ekfac") for n in NS if qld(n,"filter_top40_ekfac")]
ax2.errorbar([p[0] for p in pts],[p[1] for p in pts],yerr=[p[2] for p in pts],fmt="o-",color="#2a78d6",capsize=3,ms=4,lw=1.5)
for ax,t in ((ax1,"(a) Top 1% / 5% / 10% of documents removed"),(ax2,"(b) Top 40 documents removed")):
    ax.set_xscale("log"); ax.set_xticks([tokens(n) for n in NS]); ax.set_xticklabels([f"{tokens(n)/1e6:.0f}M" for n in NS]); ax.minorticks_off()
    ax.grid(alpha=0.25); ax.set_title(t,fontsize=10); ax.set_xlabel("Number of training tokens"); ax.axhline(0,color="k",lw=0.6,alpha=0.5)
ax1.set_ylabel("Query loss difference (filter - random)"); ax1.legend(frameon=False,fontsize=9,loc="upper left")
fig.suptitle("Qwen2.5-1.5B, EK-FAC proponent filter (partial: 5%/10% at 16k-32k and all 64k pending)",fontsize=9)
fig.tight_layout(); out="/mnt/ssd-2/lucia/metasmoothness/figures/filter_scaling_qwen_partial.png"; fig.savefig(out); print("wrote",out)
for n in NS:
    print(f"{n//1000:>3}k:", "  ".join(f"{lab}={qld(n,d)[0]:+.4f}" for lab,d,_ in SER+[("top40","filter_top40_ekfac","")] if qld(n,d)))
