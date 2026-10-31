#!/usr/bin/env bash
# Record the submission demo without a GUI:
#   1) the mission in params.yaml (fire hydrant -> chair)
#   2) after it is complete, a natural-language command is sent (default: 人を探して)
#   usage: WS=~/ws tools/record_final.sh <out.mp4> [command] [yolo weights]
OUT=$1
CMD=${2:-人を探して}
MODEL=${3:-yolo11n.pt}
WS=${WS:-$HOME/ws}

source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
export LIBGL_ALWAYS_SOFTWARE=1

LOG=${OUT%.mp4}.log
ros2 launch yolo_search_robot yolo_search.launch.py headless:=true record:=true \
  video:="$OUT" model:="$MODEL" > "$LOG" 2>&1 &
LPID=$!

timeout 1200 bash -c "until grep -q -- '-> DONE' '$LOG'; do sleep 2; done"
sleep 8
ros2 topic pub --once /search/command std_msgs/msg/String "{data: '$CMD'}" > /dev/null
timeout 900 bash -c "until [ \$(grep -c -- '-> DONE' '$LOG') -ge 2 ]; do sleep 2; done"
sleep 12

# Stop the recorder first so that ffmpeg can finish the file, then everything else.
# (SIGTERM, because background jobs of a non-interactive script ignore SIGINT.)
pkill -TERM -f "lib/yolo_search_robot/demo_recorder"
for _ in $(seq 1 30); do pgrep -f "ffmpeg -y -loglevel error -f rawvideo" > /dev/null || break; sleep 1; done
kill -TERM $LPID
pkill -TERM -f "lib/yolo_search_robot/" 2>/dev/null
pkill -f "gz sim" 2>/dev/null
pkill -f parameter_bridge 2>/dev/null
wait $LPID
echo "finished: $OUT"
