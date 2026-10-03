import json, re
import numpy as np
from transformers import AutoTokenizer
from bergson.data import load_data_string, tokenize_and_chunk
RUN="runs/olmo32b/magic_full_L32b1024"; P="runs/olmo32b/paper_tokens/olmo_paper_tokens.json"
tok=AutoTokenizer.from_pretrained("allenai/Olmo-3.1-32B-Think")
ds=load_data_string("runs/deep_ignorance_ms/mix_85_15","train",None,"")
ch=tokenize_and_chunk(ds,tok,chunk_size=512).shuffle(seed=42)
saved=np.load(f"{RUN}/scores/doc_ids.npy",mmap_mode="r")
info=json.load(open(f"{RUN}/scores/info.json"))
dt=np.dtype({k:info["dtype"][k] for k in ("names","formats","offsets","itemsize")})
scores=np.memmap(f"{RUN}/scores/scores.bin",dtype=dt,mode="r"); off=np.load(f"{RUN}/scores/offsets.npy")
d=json.load(open(P)); s=[x for x in d["negative"] if x["kind"]=="pias1"][0]
norm=lambda t: re.sub(r"\s+"," ",t)
for r in np.where((saved==48295).any(axis=1))[0]:
    ids=ch[int(r)]["input_ids"]; text=norm(tok.decode(ids))
    if "by caspase-8 leads to two N-terminal" in text and text.lstrip().startswith(norm(tok.decode(ids[:5])).lstrip()):
        sc=[None]+[float(x) for x in scores[off[r]:off[r+1]]["score_0"]]
        nxt=[{"t":tok.decode([i]),"s":sc[j],"id":int(i),"doc":int(saved[r,j]),"row":int(r)} for j,i in enumerate(ids)]
        print("continuation row",int(r),repr(text[:80]))
        break
s["continuation_row"]=int(r)
s["tokens"]=s["tokens"]+nxt
json.dump(d,open(P,"w"))
