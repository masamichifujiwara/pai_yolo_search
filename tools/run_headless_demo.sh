#!/usr/bin/env bash
# Run the whole system without a GUI and record the composed demo video.
# Used to produce the submission video on a GPU-less machine (software rendering via EGL).
#   usage: tools/run_headless_demo.sh <output.mp4> <duration_sec> [model]

OUT=${1:-$HOME/yolo_search_demo.mp4}
DURATION=${2:-300}
MODEL=${3:-yolo11n.pt}
WS=${WS:-$HOME/ws}

source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
export LIBGL_ALWAYS_SOFTWARE=1

timeout --signal=INT "$DURATION" ros2 launch yolo_search_robot yolo_search.launch.py \
  headless:=true record:=true video:="$OUT" model:="$MODEL"
# make sure nothing is left behind
pkill -f "gz sim" 2>/dev/null
pkill -f parameter_bridge 2>/dev/null
exit 0
