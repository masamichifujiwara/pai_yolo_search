"""YOLO object detector node.

Subscribes to the robot's onboard camera, runs YOLO (Ultralytics) inference and publishes
  * /yolo/detections       vision_msgs/Detection2DArray  (class name, score, bbox in pixels)
  * /yolo/annotated_image  sensor_msgs/Image (bgr8)      (camera image with boxes drawn)

Only the newest frame is processed (queue depth 1, best effort) so that inference latency on a
CPU never builds up a backlog of stale images.
"""
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .image_utils import image_msg_to_bgr, bgr_to_image_msg


class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_detector')
        self.declare_parameter('model', 'yolo11n.pt')
        self.declare_parameter('confidence', 0.30)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('image_topic', '/camera/image_raw')

        model_path = self.get_parameter('model').value
        self.conf = float(self.get_parameter('confidence').value)
        self.imgsz = int(self.get_parameter('image_size').value)
        self.device = self.get_parameter('device').value
        self.publish_annotated = bool(self.get_parameter('publish_annotated').value)

        # Imported lazily so that `ros2 run ... --help` works without torch installed.
        from ultralytics import YOLO
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.names = self.model.names
        # Warm up once so the first real frame is not slow.
        self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8), imgsz=self.imgsz,
                           conf=self.conf, device=self.device, verbose=False)

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value, self.on_image, sensor_qos)
        self.det_pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.img_pub = self.create_publisher(Image, '/yolo/annotated_image', 2)

        self._n = 0
        self._t_sum = 0.0
        self.get_logger().info('YOLO detector ready')

    def on_image(self, msg: Image):
        frame = image_msg_to_bgr(msg)
        if frame is None:
            return
        t0 = time.perf_counter()
        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                    device=self.device, verbose=False)[0]
        dt = time.perf_counter() - t0

        out = Detection2DArray()
        out.header = msg.header
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                det = Detection2D()
                det.header = msg.header
                det.bbox.center.position.x = float((x1 + x2) / 2.0)
                det.bbox.center.position.y = float((y1 + y2) / 2.0)
                det.bbox.size_x = float(x2 - x1)
                det.bbox.size_y = float(y2 - y1)
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(self.names[int(k)])
                hyp.hypothesis.score = float(c)
                det.results.append(hyp)
                out.detections.append(det)
        self.det_pub.publish(out)

        if self.publish_annotated:
            annotated = result.plot(line_width=2)
            self.img_pub.publish(bgr_to_image_msg(annotated, msg.header))

        self._n += 1
        self._t_sum += dt
        if self._n % 50 == 0:
            self.get_logger().info(
                f'avg inference {1000.0 * self._t_sum / self._n:.1f} ms over {self._n} frames')


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
