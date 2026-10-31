"""Save one frame of the given ROS image topics as PNG files (debug helper)."""
import sys
import time

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

out_dir = sys.argv[1]
topics = sys.argv[2:]
rclpy.init()
node = rclpy.create_node('snap_images')
got = {}


def make_cb(topic):
    def cb(msg):
        if topic in got:
            return
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        arr = arr.reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            arr = arr[:, :, ::-1]
        name = topic.strip('/').replace('/', '_')
        PILImage.fromarray(arr).save(f'{out_dir}/{name}.png')
        got[topic] = True
    return cb


for t in topics:
    node.create_subscription(Image, t, make_cb(t), qos_profile_sensor_data)
t0 = time.time()
while len(got) < len(topics) and time.time() - t0 < 60:
    rclpy.spin_once(node, timeout_sec=0.5)
print('saved', list(got))
