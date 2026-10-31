"""Demo video recorder.

Composes one 1280x720 frame per tick (simulation time) from
  * left  : overhead camera of the arena + robot trajectory + positions of the found objects
  * right : robot camera with YOLO boxes (from /yolo/annotated_image)
  * panel : mission status in Japanese (state, current target, checklist, latest event)
and streams it to ffmpeg (H.264, fragmented MP4 so the file stays valid even if the process is
killed). Falls back to OpenCV's VideoWriter when ffmpeg is not available.
"""
import json
import math
import os
import shutil
import subprocess

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .image_utils import image_msg_to_bgr
from .labels import ja

try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    PILImage = None

STATE_JA = {
    'IDLE': '待機中',
    'SPIN': 'その場で回転しながら探索中',
    'TURN_TO': '未探索エリアへ方向転換',
    'MOVE': '未探索エリアへ移動中',
    'APPROACH': '目標を発見 → 接近中',
    'ARRIVED': '目標に到着・位置を記録',
    'DONE': 'ミッション完了',
    'MANUAL': '手動操作中 (キーボード)',
}
STATE_COLOR = {  # RGB for PIL
    'IDLE': (180, 180, 180), 'SPIN': (120, 200, 255), 'TURN_TO': (120, 200, 255),
    'MOVE': (120, 200, 255), 'APPROACH': (255, 200, 60), 'ARRIVED': (90, 230, 120),
    'DONE': (90, 230, 120), 'MANUAL': (255, 120, 200),
}
FONT_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
    '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
    '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf',
]


class DemoRecorder(Node):
    W, H = 1280, 720
    OV = 720                 # overhead view size (left square)
    CAM_W, CAM_H = 560, 420  # robot camera view (top right)

    def __init__(self):
        super().__init__('demo_recorder')
        p = self.declare_parameter
        p('output', os.path.expanduser('~/yolo_search_demo.mp4'))
        p('fps', 10.0)
        p('overhead_height', 12.0)
        p('overhead_hfov', 0.9)
        p('overhead_x', 0.0)
        p('overhead_y', 0.0)
        p('title', 'AI物体探索ロボット  YOLO × ROS 2 × Gazebo')
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.output = g('output')
        self.fps = float(g('fps'))
        self.ov_h = float(g('overhead_height'))
        self.ov_f = (self.OV / 2.0) / math.tan(float(g('overhead_hfov')) / 2.0)
        self.ov_c = (float(g('overhead_x')), float(g('overhead_y')))
        self.title = g('title')

        self.overhead = None
        self.camera = None
        self.annotated = None
        self.status = {}
        self.trail = []
        self.frames = 0
        self.t0 = None
        self._texts = []

        self.font = {}
        self.font_path = next((p for p in FONT_CANDIDATES
                               if PILImage is not None and os.path.exists(p)), None)
        if self.font_path is None:
            self.get_logger().warn('Japanese font not found; panel text may not render')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/overhead/image_raw',
                                 lambda m: setattr(self, 'overhead', m), qos)
        self.create_subscription(Image, '/camera/image_raw',
                                 lambda m: setattr(self, 'camera', m), qos)
        self.create_subscription(Image, '/yolo/annotated_image',
                                 lambda m: setattr(self, 'annotated', m), qos)
        self.create_subscription(String, '/search/status', self.on_status, 10)
        self.create_subscription(OccupancyGrid, '/search/coverage', self.on_coverage, 1)
        self.cov_mask = None

        os.makedirs(os.path.dirname(os.path.abspath(self.output)) or '.', exist_ok=True)
        self.proc = None
        self.writer = None
        if shutil.which('ffmpeg'):
            self.proc = subprocess.Popen(
                ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                 '-s', f'{self.W}x{self.H}', '-r', str(self.fps), '-i', '-',
                 '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
                 '-movflags', 'frag_keyframe+empty_moov', self.output],
                stdin=subprocess.PIPE)
        else:
            self.writer = cv2.VideoWriter(self.output, cv2.VideoWriter_fourcc(*'mp4v'),
                                          self.fps, (self.W, self.H))
        self.create_timer(1.0 / self.fps, self.tick)
        self.get_logger().info(f'Recording demo video to {self.output}')

    # ------------------------------------------------------------------
    def on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        r = self.status.get('robot')
        if r and (not self.trail or math.dist(self.trail[-1], r[:2]) > 0.03):
            self.trail.append((r[0], r[1]))

    def on_coverage(self, msg):
        """Rasterise the cells the camera has already looked at into the overhead view."""
        w, h, res = msg.info.width, msg.info.height, msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        data = np.asarray(msg.data, dtype=np.int8).reshape(h, w)
        iy, ix = np.nonzero(data == 0)
        mask = np.zeros((self.OV, self.OV), np.uint8)
        # the ground plane maps to the overhead image by a pure scale + offset, so every cell
        # is an axis-aligned rectangle in pixels
        d = self.ov_h
        u0 = np.floor(self.OV / 2.0 + self.ov_f * (ox + ix * res - self.ov_c[0]) / d).astype(int)
        u1 = np.ceil(self.OV / 2.0 + self.ov_f * (ox + (ix + 1) * res - self.ov_c[0]) / d).astype(int)
        v0 = np.floor(self.OV / 2.0 - self.ov_f * (oy + (iy + 1) * res - self.ov_c[1]) / d).astype(int)
        v1 = np.ceil(self.OV / 2.0 - self.ov_f * (oy + iy * res - self.ov_c[1]) / d).astype(int)
        for a, b, c, e in zip(np.clip(u0, 0, self.OV), np.clip(u1, 0, self.OV),
                              np.clip(v0, 0, self.OV), np.clip(v1, 0, self.OV)):
            mask[c:e, a:b] = 255
        self.cov_mask = mask

    def world_to_px(self, x, y, z=0.1):
        d = self.ov_h - z
        u = self.OV / 2.0 + self.ov_f * (x - self.ov_c[0]) / d
        v = self.OV / 2.0 - self.ov_f * (y - self.ov_c[1]) / d
        return int(round(u)), int(round(v))

    def text(self, img, xy, s, size=20, color=(255, 255, 255)):
        """Queue (possibly Japanese) text; drawn in one PIL pass by flush_text(). RGB color."""
        self._texts.append((xy, s, size, color))
        return img

    def flush_text(self, img):
        texts, self._texts = self._texts, []
        if self.font_path is None:
            for (x, y), s, size, color in texts:
                cv2.putText(img, s.encode('ascii', 'replace').decode(), (x, y + size),
                            cv2.FONT_HERSHEY_SIMPLEX, size / 32.0, color[::-1], 1, cv2.LINE_AA)
            return img
        pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for xy, s, size, color in texts:
            draw.text(xy, s, font=self.get_font(size), fill=color,
                      stroke_width=2 if size >= 40 else 0, stroke_fill=(0, 0, 0))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    def get_font(self, size):
        if size not in self.font:
            self.font[size] = ImageFont.truetype(self.font_path, size)
        return self.font[size]

    @staticmethod
    def shade(img, x0, y0, x1, y1, alpha=0.55):
        roi = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = (roi * (1.0 - alpha)).astype(np.uint8)

    def tick(self):
        try:
            self._tick()
        except Exception as e:  # never lose the recording because of a drawing problem
            self._texts = []
            self.get_logger().error(f'frame composition failed: {e!r}')

    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.overhead is None and self.camera is None:
            return
        if self.t0 is None:
            self.t0 = now
        frame = np.full((self.H, self.W, 3), 28, np.uint8)

        # ---------------- overhead view ----------------
        ov = image_msg_to_bgr(self.overhead) if self.overhead is not None else None
        if ov is not None:
            ov = cv2.resize(ov, (self.OV, self.OV))
            if self.cov_mask is not None:
                m = self.cov_mask > 0
                tint = np.array([255, 170, 60], np.float32)       # BGR light blue
                ov[m] = (ov[m] * 0.62 + tint * 0.38).astype(np.uint8)
            path = self.status.get('path') or []
            r0 = self.status.get('robot')
            if path and r0:
                pts = [self.world_to_px(r0[0], r0[1])] + [self.world_to_px(x, y) for x, y in path]
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(ov, a, b, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(ov, pts[-1], 7, (255, 255, 255), 2, cv2.LINE_AA)
            if len(self.trail) > 1:
                pts = np.array([self.world_to_px(x, y) for x, y in self.trail], np.int32)
                cv2.polylines(ov, [pts], False, (0, 140, 255), 3, cv2.LINE_AA)
            r = self.status.get('robot')
            if r:
                u, v = self.world_to_px(r[0], r[1], 0.2)
                cv2.circle(ov, (u, v), 11, (0, 220, 255), 2, cv2.LINE_AA)
                tip = self.world_to_px(r[0] + 0.6 * math.cos(r[2]), r[1] + 0.6 * math.sin(r[2]))
                cv2.arrowedLine(ov, (u, v), tip, (0, 220, 255), 2, cv2.LINE_AA, tipLength=0.35)
            frame[:, :self.OV] = ov
            found_pos = [(f['x'], f['y']) for f in self.status.get('found', [])]
            for mobj in self.status.get('memory', []):
                if any(math.dist((mobj['x'], mobj['y']), p) < 1.0 for p in found_pos):
                    continue
                u, v = self.world_to_px(mobj['x'], mobj['y'], 0.3)
                cv2.circle(frame, (u, v), 6, (230, 230, 230), 2, cv2.LINE_AA)
                self.shade(frame, max(u + 8, 0), max(v - 11, 0), min(u + 12 + 16 * len(mobj['name_ja']),
                           self.OV), v + 9, 0.5)
                frame = self.text(frame, (u + 10, v - 11), mobj['name_ja'], 16, (235, 235, 235))
            for i, f in enumerate(self.status.get('found', [])):
                u, v = self.world_to_px(f['x'], f['y'], 0.5)
                cv2.drawMarker(frame, (u, v), (60, 230, 90), cv2.MARKER_STAR, 26, 3, cv2.LINE_AA)
                label = f'{i + 1}. {f["name_ja"]} ({f["x"]:.1f}, {f["y"]:.1f})'
                lx = min(u + 14, self.OV - 20 * len(label) // 2 - 10)
                self.shade(frame, max(lx - 4, 0), max(v - 16, 0), min(lx + 11 * len(label), self.OV),
                           v + 12, 0.6)
                frame = self.text(frame, (lx, v - 15), label, 20, (120, 255, 140))
        self.shade(frame, 0, 0, 720, 30)
        frame = self.text(frame, (10, 5),
                          '俯瞰  青: カメラで確認済み  白線: 移動経路  オレンジ: 軌跡  ★: 発見  ○: 記憶した物体',
                          16,
                          (255, 255, 255))

        # ---------------- robot camera + YOLO ----------------
        cam = None
        if self.annotated is not None:
            cam = image_msg_to_bgr(self.annotated)
        elif self.camera is not None:
            cam = image_msg_to_bgr(self.camera)
        if cam is not None:
            frame[:self.CAM_H, self.OV:self.OV + self.CAM_W] = cv2.resize(
                cam, (self.CAM_W, self.CAM_H))
        self.shade(frame, self.OV, self.CAM_H - 28, self.OV + 300, self.CAM_H)
        frame = self.text(frame, (self.OV + 8, self.CAM_H - 25), 'ロボット視点カメラ + YOLO 物体検出',
                          16, (255, 255, 255))
        st = self.status.get('state', 'IDLE')
        if st == 'ARRIVED' and self.status.get('found'):
            name = self.status['found'][-1]['name_ja']
            self.shade(frame, self.OV, self.CAM_H - 100, self.W, self.CAM_H - 40, 0.6)
            frame = self.text(frame, (self.OV + 20, self.CAM_H - 96), f'{name} を発見!', 40,
                              (90, 255, 120))
        if st == 'DONE':
            self.shade(frame, 110, 320, 610, 400, 0.6)
            frame = self.text(frame, (150, 330), 'ミッション完了!', 48, (90, 255, 120))

        # ---------------- status panel ----------------
        x0, y0 = self.OV + 14, self.CAM_H + 8
        st = self.status.get('state', 'IDLE')
        frame = self.text(frame, (x0, y0), self.title, 20, (255, 255, 255))
        frame = self.text(frame, (x0, y0 + 34), f'状態: {STATE_JA.get(st, st)}', 24,
                          STATE_COLOR.get(st, (255, 255, 255)))
        tgt = self.status.get('target')
        frame = self.text(frame, (x0, y0 + 70),
                          f'探索目標: {ja(tgt)} ({tgt})' if tgt else '探索目標: なし', 20,
                          (255, 230, 120))
        found = [f['name_ja'] for f in self.status.get('found', [])]
        remaining = [ja(t) for t in self.status.get('remaining', [])]
        checklist = '  '.join([f'●{n}' for n in found] + [f'○{n}' for n in remaining])
        frame = self.text(frame, (x0, y0 + 102), f'ミッション: {checklist}', 20, (220, 220, 220))
        ev = self.status.get('event', '')
        frame = self.text(frame, (x0, y0 + 136), f'▶ {ev[:26]}', 20, (150, 220, 255))
        if len(ev) > 26:
            frame = self.text(frame, (x0 + 22, y0 + 164), ev[26:52], 20, (150, 220, 255))
        cmd_text = self.status.get('command')
        if cmd_text:
            frame = self.text(frame, (x0, y0 + 204), f'ユーザーの指示: 「{cmd_text}」', 18,
                              (255, 190, 120))
        el = now - self.t0
        frame = self.text(frame, (x0, y0 + 250),
                          f'シミュレーション時間 {int(el // 60):02d}:{int(el % 60):02d}', 16,
                          (170, 170, 170))
        cv2.line(frame, (self.OV, self.CAM_H), (self.W, self.CAM_H), (90, 90, 90), 1)

        frame = self.flush_text(frame)
        # keep the video time equal to simulation time: if ticks were skipped because the CPU
        # was busy, repeat the frame (at most 1 s worth) instead of silently speeding up
        expected = int((now - self.t0) * self.fps) + 1
        for _ in range(max(1, min(expected - self.frames, int(self.fps)))):
            self.write(frame)

    def write(self, frame):
        self.frames += 1
        if self.proc is not None:
            try:
                self.proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                self.get_logger().error('ffmpeg pipe closed')
                self.proc = None
        elif self.writer is not None:
            self.writer.write(frame)

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        if self.writer is not None:
            self.writer.release()
        self.get_logger().info(f'Wrote {self.frames} frames to {self.output}')


def main(args=None):
    rclpy.init(args=args)
    node = DemoRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # let ffmpeg finish the file
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
