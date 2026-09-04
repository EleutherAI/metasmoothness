#!/bin/bash
# Post-run step for qwen 64k plan A. Run on lucia-ord as ROOT via: kubectl exec lucia-ord-0 -- bash /tmp/assemble_and_merge_q64.sh <1pct|top40>
# 1) chmod bank shard models readable, 2) assemble bank_<tag>/ (base + subset_0..2 + subsets.json), 3) bank_merge every finished
# one-query shard that lacks a summary, 4) merge the 20 shard summaries into filter_<prefix>_ekfac/filter_summary.csv.
set -u
tag=${1:?1pct|top40|5pct|10pct}; case $tag in 1pct) prefix=filter_proponents;; top40) prefix=filter_top40;; 5pct) prefix=filter_prop5pct;; 10pct) prefix=filter_prop10pct;; *) echo bad tag; exit 1;; esac
E=/mnt/ssd-2/lucia/paper_runs/experiments/qwen15b_64k_bs256; PY=/mnt/ssd-2/lucia/envs/paper/bin/python; M=/mnt/ssd-2/lucia/metasmoothness/scripts; D=/mnt/ssd-2/lucia/datasets_local
echo "== 1. bank shards"
for i in 0 1 2; do
  f=$E/bank_${tag}_s$i/retrained/subset_$i/model.safetensors
  [ -f "$f" ] && [ "$(stat -c %s "$f")" = "3087467832" ] || { echo "bank shard s$i model missing/incomplete: $f"; exit 1; }
  chmod -R a+rX $E/bank_${tag}_s$i/retrained
done
[ -f $E/bank_${tag}_s0/subsets.json ] || { echo "no subsets.json in bank_${tag}_s0"; exit 1; }
B=$E/bank_$tag; mkdir -p $B/retrained; [ -e $B/retrained/base ] || ln -s $E/base/model $B/retrained/base
for i in 0 1 2; do [ -e $B/retrained/subset_$i ] || ln -s $E/bank_${tag}_s$i/retrained/subset_$i $B/retrained/subset_$i; done
cp -f $E/bank_${tag}_s0/subsets.json $B/subsets.json; chown -R lucia:lucia $B
echo "bank_$tag: $(ls $B/retrained | tr '\n' ' ') sizes=$($PY -c "import json;print([len(x) for x in json.load(open('$B/subsets.json'))])")"
echo "== 2. bank_merge per finished shard"
merged=0; pending=0
for a in $(seq 0 19); do
  R=$E/${prefix}_ekfac_q${a}_$((a+1))
  if [ ! -s $R/filter_proponents.csv ] || [ "$(wc -l < $R/filter_proponents.csv)" -lt 2 ]; then pending=$((pending+1)); continue; fi
  if [ -f $R/filter_summary.csv ]; then merged=$((merged+1)); continue; fi
  su - lucia -c "cd /tmp && CUDA_VISIBLE_DEVICES=0 $PY $M/bank_merge.py --run $R --bank $B --query $D/query_20_qwen_q${a}_$((a+1)).hf" 2>&1 | grep -E "^wrote|Error" | sed "s#$E/##" && merged=$((merged+1))
done
echo "shards merged=$merged pending=$pending"
if [ $merged -eq 20 ]; then
  echo "== 3. merge 20 shards"
  su - lucia -c "cd /tmp && $PY -P $M/merge_filter_shards.py qwen15b_64k_bs256 --source ekfac --prefix $prefix --force" 2>&1 | tail -2
fi
