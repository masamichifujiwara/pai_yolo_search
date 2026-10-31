#!/usr/bin/env bash
# Teleport the robot around an object, grab camera frames, used to evaluate YOLO detection
# usage: detection_test.sh <out_dir> <obj_x> <obj_y>
OUT=$1; OX=$2; OY=$3
HERE=$(cd "$(dirname "$0")" && pwd)
source /opt/ros/jazzy/setup.bash; source "${WS:-$HOME/ws}/install/setup.bash"
mkdir -p $OUT
for spec in "1.5 -30" "1.5 -60" "3.0 -20" "3.0 -45" "5.0 -10" "5.0 -30"; do
  set -- $spec; D=$1; TH=$2
  read X Y YAW QZ QW < <(/usr/bin/python3 -c "
import math; d=$D; th=math.radians($TH)
x=$OX+d*math.cos(th); y=$OY+d*math.sin(th); yaw=th+math.pi
print(x, y, yaw, math.sin(yaw/2), math.cos(yaw/2))")
  gz service -s /world/search_arena/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 3000 \
    --req "name: \"search_robot\", position: {x: $X, y: $Y, z: 0.02}, orientation: {z: $QZ, w: $QW}" > /dev/null
  sleep 2
  /usr/bin/python3 "$HERE/snap_images.py" $OUT /camera/image_raw > /dev/null
  mv $OUT/camera_image_raw.png $OUT/d${D}_th${TH}.png
done
ls $OUT
