#!/bin/bash
set -e
cd /workspace/DroneStudio/autoresearch
source /workspace/.dashboard_env
PY=/workspace/venv/bin/python
[ -f /workspace/t4_best_pre_omegafix.json ] || cp /workspace/t4_best.json /workspace/t4_best_pre_omegafix.json
echo "CHAIN_START $(date -u +%FT%TZ)"
for s in t4_pilot.py t4_pilot2.py t4_bc.py t4_bc_eval.py t4_dagger.py t4_dagger2.py; do
  echo "STAGE_START $s $(date -u +%FT%TZ)"
  $PY $s
  echo "STAGE_DONE $s $(date -u +%FT%TZ)"
done
echo "CHAIN_DONE $(date -u +%FT%TZ)"
