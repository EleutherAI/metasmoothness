#!/bin/bash
# csv-based consume for Qwen 1.5B rows whose banks kept no retrained models: for each series, pool the 3 bank shards'
# validation csvs into <run>/validation_merged.csv, rebuild every shard's filter_summary.csv from it, then merge the
# 20 shards into filter_<prefix>_ekfac/filter_summary.csv. Idempotent (--force). Usage: bash merge_q15b_csv.sh 128k [tags...]
N=${1:?}; shift; TAGS=${@:-"1pct 5pct 10pct top40 top200 top400"}
E=/mnt/ssd-2/lucia/paper_runs/experiments; RUN=qwen15b_${N}_bs256; R=$E/$RUN; M=/mnt/ssd-2/lucia/metasmoothness/scripts; PY=/mnt/ssd-2/lucia/envs/paper/bin/python
cd /tmp
for tag in $TAGS; do
  case $tag in 1pct) pfx=filter_proponents;; 5pct) pfx=filter_prop5pct;; 10pct) pfx=filter_prop10pct;; *) pfx=filter_$tag;; esac
  done_n=$(for a in $(seq 0 19); do c=$R/${pfx}_ekfac_q${a}_$((a+1))/filter_proponents.csv; [ -s "$c" ] && [ $(wc -l < $c) -ge 2 ] && echo x; done | wc -l)
  bank_n=$(for i in 0 1 2; do c=$(ls $R/bank_${tag}_s$i/validation_*.csv 2>/dev/null | head -1); [ -n "$c" ] && [ $(wc -l < $c) -ge 21 ] && echo x; done | wc -l)
  echo "== $N $tag ($pfx): shards done $done_n/20, bank shards $bank_n/3"
  [ $done_n -eq 20 ] && [ $bank_n -eq 3 ] || { echo "   skip (incomplete)"; continue; }
  { head -1 $(ls $R/bank_${tag}_s0/validation_*.csv | head -1); for i in 0 1 2; do tail -n +2 $(ls $R/bank_${tag}_s$i/validation_*.csv | head -1); done; } > $R/validation_merged.csv
  echo "   bank rows: $(tail -n +2 $R/validation_merged.csv | wc -l) (subsets: $(tail -n +2 $R/validation_merged.csv | cut -d, -f1 | sort -u | tr '\n' ' '))"
  for a in $(seq 0 19); do $PY -P $M/recover_shard_summary.py $RUN --source ekfac --prefix $pfx --shard q${a}_$((a+1)) --force 2>&1 | grep -E "wrote|WARN|absent|Error" | sed "s#$R/##" | tail -1; done | grep -v "^  wrote" ; 
  echo "   summaries: $(ls $R/${pfx}_ekfac_q*_*/filter_summary.csv | wc -l)/20"
  $PY -P $M/merge_filter_shards.py $RUN --source ekfac --prefix $pfx --force 2>&1 | tail -2 | sed "s#$R/##"
  rm -f $R/validation_merged.csv
  $PY - "$R/${pfx}_ekfac/filter_summary.csv" <<'PY'
import csv,sys,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
d=[float(r["filter_change"])-float(r["random_mean"]) for r in rows]
print(f"   QLD mean {st.fmean(d):+.4f}  median {st.median(d):+.4f}  min {min(d):+.4f} max {max(d):+.4f}  n={len(d)}  filter_mean {st.fmean(float(r['filter_change']) for r in rows):+.4f} random_mean {st.fmean(float(r['random_mean']) for r in rows):+.4f}")
PY
  chmod -R a+rwX $R/${pfx}_ekfac 2>/dev/null
done
