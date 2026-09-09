#!/bin/sh
# vis_scenario_streamer supervisor (rung-2): DR visuals + v2 ckpt, restart on crash
. /workspace/.dashboard_env
export VIS_CKPT=/workspace/vision_model/visnet_v2_best.pt
export STREAM_POLICY_FLAT=/workspace/t4_live.json
while true; do
  /workspace/venv-vision/bin/python /workspace/DroneStudio/autoresearch/vis_scenario_streamer.py >> /workspace/vision_model/scenario_streamer.log 2>&1
  echo "$(date -u +%FT%TZ) scenario streamer exited rc=$? - restarting in 10s" >> /workspace/vision_model/scenario_streamer.log
  sleep 10
done
