#!/bin/bash
# Gate-time chain for qwen 64k plan A. Run as lucia on lucia-ord. Stops at the first failure.
set -e; PY=/mnt/ssd-2/lucia/envs/paper/bin/python; M=/mnt/ssd-2/lucia/metasmoothness/scripts; E=/mnt/ssd-2/lucia/paper_runs/experiments/qwen15b_64k_bs256; cd /tmp
echo "== 1. score gate"; $PY -P $M/gate_ekfac.py scores $E/ekfac_scores_64k | tail -1 | grep -q PASS || { echo "SCORE GATE FAIL"; exit 1; }
echo "== 2. 1-query score slices"; $PY -P $M/slicefix.py $E/ekfac_scores_64k/scores $E scores64 1 20 2>&1 | tail -1
grep -q "higher_is_better: true" $E/scores64_q0_1/config.yaml || { echo "slice config lacks higher_is_better true"; exit 1; }
echo "== 3. selection audit"; $PY /tmp/audit_selection.py | tail -5
echo "== 4. full 64k-vs-32k score comparison (all 20 queries)"; $PY /tmp/cmp_scores.py | grep -v Warning | tail -21
echo "GATE CHAIN PASSED -- launch banks + 40 proponent shards next"
