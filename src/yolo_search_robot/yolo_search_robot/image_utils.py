"""Minimal sensor_msgs/Image <-> numpy conversion (avoids a cv_bridge / NumPy ABI dependency)."""
import numpy as np
from sensor_msgs.msg import Image


def image_msg_to_bgr(msg: Image):
    """Return an HxWx3 uint8 BGR array, or None for unsupported encodings."""
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ('rgb8', 'bgr8'):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if enc == 'rgb8' else img.copy()
    if enc in ('rgba8', 'bgra8'):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 4].reshape(msg.height, msg.width, 4)
        img = img[:, :, :3]
        return img[:, :, ::-1].copy() if enc == 'rgba8' else img.copy()
    if enc in ('mono8', '8uc1'):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width]
        return np.repeat(img[:, :, None], 3, axis=2)
    return None


def bgr_to_image_msg(img, header) -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = img.shape[:2]
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(img, dtype=np.uint8).tobytes()
    return msg
