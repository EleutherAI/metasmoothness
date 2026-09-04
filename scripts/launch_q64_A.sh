#!/bin/bash
# Gate-time launcher for qwen 64k plan A, run from the Mac (needs kubectl). Re-runnable: launches whatever fits on
# currently free pairs and records launched job names in /tmp/q64_launched.txt; run again as pairs free.
# Bank shards (save models) and proponent shards (save_models: true) only go to uid-1000 nodes (not iris/secret-ord).
set -u
E=/mnt/ssd-2/lucia/paper_runs/experiments/qwen15b_64k_bs256
L=/mnt/ssd-2/lucia/paper_runs/_launch
NODES="lucia-ord-0 poppy-0 daisy-0 wisteria-0 jasmine-0 allium-0 lily-0 clover-0 heather-0 orchid-0 bellflower-0 lotus-0 marisa-0 shivam2-0 louis-ord-0 shared-ord-0 violet-0 yarrow-0"
BAD_PAIRS="poppy-0:4,5"   # quarantined (NaN baseline eval)
LAUNCHED=${LAUNCHED:-/tmp/q64_launched.txt}; TAGS=${TAGS:-1pct:filter_proponents_ekfac,top40:filter_top40_ekfac}; PORTBASE=${PORTBASE:-31300}
touch $LAUNCHED
echo "== scanning free pairs"
: > /tmp/q64_free.txt
for p in $NODES; do
  timeout 30 kubectl exec $p -- sh -c 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits' 2>/dev/null \
   | awk -F, -v n=$p '{gsub(/ /,"",$2); u[$1]=$2} END{for(i=0;i<8;i+=2){ if((i in u) && u[i]<1000 && u[i+1]<1000) print n":"i","i+1 }}' >> /tmp/q64_free.txt &
done; wait
grep -v -F "$BAD_PAIRS" /tmp/q64_free.txt | sort > /tmp/q64_free_sorted.txt
echo "free pairs: $(wc -l < /tmp/q64_free_sorted.txt); already launched: $(wc -l < $LAUNCHED)"
rm -f /tmp/q64_launch_*.sh /tmp/q64_nodes.txt
python3 - "$E" "$LAUNCHED" "$TAGS" "$PORTBASE" <<'PY'
import sys
E,LAUNCHED,TAGS,PORTBASE=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
tags=[t.split(":") for t in TAGS.split(",")]
free=[l.strip() for l in open("/tmp/q64_free_sorted.txt") if l.strip()]
done={l.strip() for l in open(LAUNCHED) if l.strip()}
jobs=[(f"{E}/bank_{t}_s{i}.yaml",f"q15b_64k_bank_{t}_s{i}") for t,_ in tags for i in range(3)]
jobs+=[(f"{E}/{p}_q{a}_{a+1}.yaml",f"q15b_64k_{t}_q{a}_{a+1}") for t,p in tags for a in range(20)]
todo=[j for j in jobs if j[1] not in done]
per={}; port=PORTBASE+len(done)
for (cfg,name),slot in zip(todo,free):
    node,pair=slot.split(":"); per.setdefault(node,[]).append((cfg,pair,port,name)); port+=1
launched=[]
for node,items in per.items():
    with open(f"/tmp/q64_launch_{node}.sh","w") as f:
        f.write("cd /mnt/ssd-2/lucia/metasmoothness\nfree(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v a=$1 -v b=$2 '($1==a||$1==b)&&$2>1000{bad=1}END{exit bad}'; }\n")
        for cfg,pair,pt,name in items:
            a,b=pair.split(","); f.write(f"free {a} {b} && bash scripts/launch_one.sh {cfg} {pair} {pt} {name} | tail -1 || echo '  SKIP {name}: {node} {pair} busy'\n"); launched.append(name)
    print(f"  {node}: {len(items)} jobs")
open("/tmp/q64_nodes.txt","w").write("\n".join(per))
open(LAUNCHED,"a").write("".join(n+"\n" for n in launched))
print(f"launching {len(launched)} of {len(todo)} pending ({len(todo)-len(launched)} wait for free pairs)")
PY
[ -s /tmp/q64_nodes.txt ] || { echo "nothing to launch"; exit 0; }
echo "== copying launch scripts + firing"
for node in $(cat /tmp/q64_nodes.txt); do
  kubectl cp /tmp/q64_launch_$node.sh lucia-ord-0:$L/q64_launch_$node.sh >/dev/null 2>&1
done
for node in $(cat /tmp/q64_nodes.txt); do
  timeout 300 kubectl exec $node -- su - lucia -c "bash $L/q64_launch_$node.sh" 2>&1 | sed "s/^/[$node] /" &
done; wait
echo "== done; re-run this script later to launch the remainder (if any). SKIPped jobs must be removed from /tmp/q64_launched.txt to retry."
