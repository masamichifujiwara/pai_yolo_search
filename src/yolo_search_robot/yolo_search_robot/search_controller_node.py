"""Object search controller (state machine).

Given a list of target classes (COCO names, e.g. ["person", "fire hydrant", "chair"]) the robot
finds and visits them one after another:

  SPIN      rotate 360 deg in place while YOLO looks for the current target
  MOVE      frontier exploration: follow a grid path (BFS) to the nearest boundary between
            areas the camera has already seen and areas it has not, then SPIN again
            (fallback when no frontier is reachable: TURN_TO + straight MOVE towards the most
            open, least visited direction)
  APPROACH  target confirmed -> visual servoing on the bounding box, LiDAR gives the distance
  ARRIVED   stopped in front of the object, its map position is recorded, then next target
  DONE      every target has been found
  MANUAL    keyboard teleop (same key layout as the baseline sample) overrides everything

Latency compensation: the bearing of a detection is converted into a *world* heading using the
robot yaw at the image capture time (looked up in an odometry history buffer), so slow CPU
inference does not make the robot overshoot while it is turning.

Semantic memory: every confident YOLO detection (any class) is turned into a map position
(bearing + LiDAR / ground-plane distance) and remembered. When asked for an object that was
already seen, the robot plans a path straight to it instead of exploring again.

Commands can be sent at runtime on /search/command (std_msgs/String), in English or Japanese:
  "消火栓と椅子を探して", "find person", "止まって", "手動", "自動"
"""
import json
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

from nav_msgs.msg import OccupancyGrid

from .exploration_map import ExplorationMap
from .labels import ja, parse_targets, classify_command


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SearchController(Node):
    IDLE, SPIN, TURN_TO, MOVE, APPROACH, ARRIVED, DONE, MANUAL = (
        'IDLE', 'SPIN', 'TURN_TO', 'MOVE', 'APPROACH', 'ARRIVED', 'DONE', 'MANUAL')

    def __init__(self):
        super().__init__('search_controller')
        p = self.declare_parameter
        p('targets', ['fire hydrant', 'chair'])
        p('confidence', 0.45)
        p('confirm_frames', 3)
        p('lost_timeout', 2.5)
        p('teleop_timeout',0.5)
        p('stop_distance', 0.9)
        p('max_linear', 0.30)
        p('max_angular', 0.8)
        p('spin_speed', 0.55)
        p('k_bearing', 1.6)
        p('camera_hfov', 1.3)
        p('camera_height', 0.14)           # [m] optical centre above the floor
        p('image_width', 640)
        p('image_height', 480)
        p('obstacle_distance', 0.45)
        p('explore_step', 2.5)
        p('arrived_pause', 3.0)
        p('spawn_x', 0.0)
        p('spawn_y', 0.0)
        p('spawn_yaw', 0.0)
        p('pose_source', 'ground_truth')   # 'ground_truth' (world frame) or 'odom' (wheel odom)
        p('use_frontier', True)
        p('map_resolution', 0.25)
        p('map_size', 16.0)
        p('map_origin_x', -8.0)
        p('map_origin_y', -8.0)
        p('view_range', 5.0)               # [m] camera range assumed for "seen" cells
        p('exclusion_radius', 1.0)         # [m] ignore detections close to objects already found
        p('memory_confidence', 0.5)        # min score for remembering a detection
        p('memory_max_distance', 6.0)      # [m] only remember objects closer than this
        p('auto_start', True)
        p('start_delay', 3.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.targets = list(g('targets'))
        self.conf = float(g('confidence'))
        self.confirm_frames = int(g('confirm_frames'))
        self.lost_timeout = float(g('lost_timeout'))
        self.teleop_timeout = float(g('teleop_timeout'))
        self.stop_distance = float(g('stop_distance'))
        self.max_lin = float(g('max_linear'))
        self.max_ang = float(g('max_angular'))
        self.spin_speed = float(g('spin_speed'))
        self.k_bearing = float(g('k_bearing'))
        self.img_w = int(g('image_width'))
        self.img_h = int(g('image_height'))
        self.fx = (self.img_w / 2.0) / math.tan(float(g('camera_hfov')) / 2.0)
        self.cam_h = float(g('camera_height'))
        self.obstacle_distance = float(g('obstacle_distance'))
        self.explore_step = float(g('explore_step'))
        self.arrived_pause = float(g('arrived_pause'))
        self.spawn = (float(g('spawn_x')), float(g('spawn_y')), float(g('spawn_yaw')))
        self.start_delay = float(g('start_delay'))
        self.pose_source = g('pose_source')
        self.use_frontier = bool(g('use_frontier'))
        self.exclusion_radius = float(g('exclusion_radius'))
        self.memory_conf = float(g('memory_confidence'))
        self.memory_max_dist = float(g('memory_max_distance'))
        self.memory = {}                      # class -> [{'x','y','n','score','checked'}]
        self.last_command = ''                # last natural-language command (for the video)
        self.map = ExplorationMap(float(g('map_resolution')), float(g('map_size')),
                                  (float(g('map_origin_x')), float(g('map_origin_y'))),
                                  float(g('view_range')), float(g('camera_hfov')))
        self.waypoints = []                   # world-frame path to the current frontier
        self.leg_t0 = 0.0
        self.last_map_update = 0.0
        self.odom_world = None                # wheel odometry converted to the world frame
        self.gt_pose = None                   # ground truth (world frame)

        # ---- runtime state ----
        self.state = self.IDLE
        self.auto_start = bool(g('auto_start'))
        self.prev_auto_state = self.SPIN
        self.pose = None                      # (x, y, yaw) in odom frame
        self.pose_hist = deque(maxlen=400)    # (t, x, y, yaw)
        self.scan = None
        self.hits = 0
        self.target_world_yaw = None          # heading (odom frame) towards the target
        self.last_seen_t = None
        self.last_bbox = None                 # (cx, w, h, score)
        self.spin_accum = 0.0
        self.spin_dir = 1.0
        self.prev_yaw = None
        self.goal_yaw = 0.0
        self.leg_start = None
        self.leg_len = 0.0
        self.visited = []                     # odom-frame points already explored
        self.found = []                       # dicts: class, x, y (world)
        self.arrived_t = None
        self.teleop_cmd = Twist()
        self.teleop_last_t = None
        self.event = ''                       # last human-readable event (for the video)
        self.t_start = None
        self.yolo_ready = False               # set on the first /yolo/detections message
        self.avoid_until = 0.0                # obstacle side-step manoeuvre (APPROACH)
        self.avoid_dir = 1.0

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/search/status', 10)
        self.found_pub = self.create_publisher(String, '/search/found', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/search/coverage', 1)
        self.create_timer(1.0, self.publish_map)

        self.create_subscription(Detection2DArray, '/yolo/detections', self.on_detections, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 20)
        self.create_subscription(Odometry, '/ground_truth', self.on_ground_truth, 20)
        self.create_subscription(String, '/search/command', self.on_command, 10)
        self.create_subscription(Twist, '/teleop/cmd_vel', self.on_teleop, 10)
        self.create_subscription(String, '/teleop/mode', self.on_mode, 10)

        self.create_timer(0.1, self.tick)
        self.get_logger().info(f'Search targets: {self.targets}')

    # ------------------------------------------------------------------ helpers
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @property
    def target(self):
        return self.targets[0] if self.targets else None

    def to_world(self, x, y, yaw=0.0):
        """Control frame -> world frame (identity when localising with ground truth)."""
        if self.pose_source == 'ground_truth':
            return x, y, yaw
        return self.odom_to_world(x, y, yaw)

    def odom_to_world(self, x, y, yaw=0.0):
        sx, sy, syaw = self.spawn
        c, s = math.cos(syaw), math.sin(syaw)
        return sx + c * x - s * y, sy + s * x + c * y, wrap(syaw + yaw)

    def yaw_at(self, t):
        """Robot yaw at (sim) time t, from the odometry history."""
        if not self.pose_hist:
            return self.pose[2] if self.pose else 0.0
        best = min(self.pose_hist, key=lambda h: abs(h[0] - t))
        return best[3]

    def range_at(self, rel_angle, half_width=math.radians(4)):
        """Minimum LiDAR range around a robot-relative angle."""
        s = self.scan
        if s is None:
            return float('inf')
        vals = []
        a = rel_angle - half_width
        while a <= rel_angle + half_width + 1e-9:
            i = int(round((wrap(a) - s.angle_min) / s.angle_increment))
            if 0 <= i < len(s.ranges):
                r = s.ranges[i]
                if math.isfinite(r) and r >= s.range_min:
                    vals.append(r)
                elif math.isinf(r) and r > 0:
                    vals.append(s.range_max)
            a += s.angle_increment
        return min(vals) if vals else s.range_max

    def set_state(self, st, event=None):
        if st != self.state:
            self.get_logger().info(f'{self.state} -> {st}' + (f' ({event})' if event else ''))
        self.state = st
        if st != self.MOVE:
            self.waypoints = []
        if event:
            self.event = event
        if st == self.SPIN:
            self.spin_accum = 0.0
            self.prev_yaw = self.pose[2] if self.pose else None
            if self.pose:
                self.visited.append(self.pose[:2])

    # ------------------------------------------------------------------ callbacks
    def on_odom(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.odom_world = self.odom_to_world(x, y, yaw)
        if self.pose_source == 'odom':
            self.set_pose(msg.header.stamp, x, y, yaw)

    def on_ground_truth(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.gt_pose = (x, y, yaw)
        if self.pose_source == 'ground_truth':
            self.set_pose(msg.header.stamp, x, y, yaw)

    def set_pose(self, stamp, x, y, yaw):
        self.pose = (x, y, yaw)
        t = stamp.sec + stamp.nanosec * 1e-9
        self.pose_hist.append((t, x, y, yaw))

    def on_scan(self, msg):
        self.scan = msg
        now = self.now()
        if self.pose is not None and now - self.last_map_update > 0.2:
            self.last_map_update = now
            self.map.update(self.to_world(*self.pose), msg)

    def on_teleop(self, msg):
        self.teleop_cmd = msg
        self.teleop_last_t = self.now()

    def on_mode(self, msg):
        self.handle_mode(msg.data.strip().lower())

    def handle_mode(self, mode):
        if mode == 'manual' and self.state != self.MANUAL:
            self.prev_auto_state = self.state if self.state not in (self.IDLE, self.DONE) \
                else self.SPIN
            self.set_state(self.MANUAL, '手動操作モード')
        elif mode == 'auto' and self.state == self.MANUAL:
            self.set_state(self.SPIN if self.targets else self.DONE, '自律探索モード')

    def on_command(self, msg):
        text = msg.data
        self.last_command = text
        kind = classify_command(text)
        self.get_logger().info(f'command "{text}" -> {kind}')
        if kind == 'stop':
            self.set_state(self.IDLE, '停止指令')
        elif kind == 'manual':
            self.handle_mode('manual')
        elif kind == 'auto':
            if self.state == self.MANUAL:
                self.handle_mode('auto')
            elif self.targets:
                self.set_state(self.SPIN, '探索再開')
        elif kind == 'targets':
            self.targets = parse_targets(text)
            self.hits = 0
            names = '、'.join(ja(t) for t in self.targets)
            if self.recall(self.target) is None:
                # never seen it: the earlier coverage was for other objects, explore again
                self.map.reset_seen()
            self.start_search(f'指示「{text}」→ {names} を探す')
        else:
            self.event = f'理解できない指令: {text}'

    def on_detections(self, msg):
        if not self.yolo_ready:
            self.yolo_ready = True
            self.t_start = self.now()      # start the mission clock once the detector is up
        if self.pose is None:
            return
        self.remember(msg)
        if self.target is None:
            return
        best = None
        for d in msg.detections:
            if not d.results:
                continue
            h = d.results[0].hypothesis
            if h.class_id != self.target or h.score < self.conf:
                continue
            area = d.bbox.size_x * d.bbox.size_y
            key = h.score * math.sqrt(area)
            if best is None or key > best[0]:
                best = (key, d, h.score)
        if best is None:
            self.hits = 0
            return
        _, d, score = best
        cx = d.bbox.center.position.x
        bearing_cam = math.atan2(-(cx - self.img_w / 2.0), self.fx)   # left positive
        t_img = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        world_yaw = wrap(self.yaw_at(t_img) + bearing_cam)
        if self.near_found_object(world_yaw):
            # one physical object cannot be two classes: e.g. a fire hydrant seen from very
            # close and mis-classified as "chair" right after it was found
            self.hits = 0
            return
        self.target_world_yaw = world_yaw
        self.last_seen_t = self.now()
        self.last_bbox = (cx, d.bbox.size_x, d.bbox.size_y, score,
                          d.bbox.center.position.y + d.bbox.size_y / 2.0)
        self.hits += 1

    # ------------------------------------------------------------------ main loop
    def tick(self):
        cmd = Twist()
        now = self.now()
        if self.t_start is None:
            self.t_start = now

        if self.pose is None or self.scan is None:
            self.publish(cmd)
            return

        if (self.state == self.IDLE and self.auto_start and self.yolo_ready
                and now - self.t_start > self.start_delay):
            self.auto_start = False
            names = '、'.join(ja(t) for t in self.targets)
            self.start_search(f'ミッション開始: {names} を探す')

        x, y, yaw = self.pose
        confirmed = self.hits >= self.confirm_frames and self.target is not None

        if self.state == self.MANUAL:
            if self.teleop_last_t is None or now - self.teleop_last_t > self.teleop_timeout:
                pass                    # 何もしない → cmd は0のまま = 停止
            else:
                cmd = self.teleop_cmd

        elif self.state in (self.SPIN, self.TURN_TO, self.MOVE) and confirmed:
            self.set_state(self.APPROACH, f'{ja(self.target)} を発見! 接近します')

        elif self.state == self.SPIN:
            if self.prev_yaw is None:
                self.prev_yaw = yaw
            self.spin_accum += abs(wrap(yaw - self.prev_yaw))
            self.prev_yaw = yaw
            # slow down while a candidate is being confirmed so YOLO gets a few frames of it
            speed = 0.2 if self.hits > 0 else self.spin_speed
            cmd.angular.z = self.spin_dir * speed
            if self.spin_accum > 2 * math.pi + 0.2:
                self.choose_exploration_goal()

        elif self.state == self.TURN_TO:
            err = wrap(self.goal_yaw - yaw)
            cmd.angular.z = clamp(2.0 * err, -self.max_ang, self.max_ang)
            if abs(err) < 0.08:
                self.leg_start = (x, y)
                self.set_state(self.MOVE)

        elif self.state == self.MOVE and self.waypoints:
            cmd = self.follow_path(now)

        elif self.state == self.MOVE:
            err = wrap(self.goal_yaw - yaw)
            front = self.range_at(0.0, math.radians(20))
            travelled = math.hypot(x - self.leg_start[0], y - self.leg_start[1])
            if front < self.obstacle_distance + 0.1 or travelled >= self.leg_len:
                self.set_state(self.SPIN)
            else:
                cmd.linear.x = self.max_lin * clamp((front - self.obstacle_distance) / 0.6, 0.3, 1.0)
                cmd.angular.z = clamp(2.0 * err, -self.max_ang, self.max_ang)
                if travelled > 0.5 and (not self.visited or
                                        math.dist(self.visited[-1], (x, y)) > 0.7):
                    self.visited.append((x, y))

        elif self.state == self.APPROACH:
            cmd = self.approach(now, yaw)

        elif self.state == self.ARRIVED:
            if now - self.arrived_t > self.arrived_pause:
                self.hits = 0
                if self.targets:
                    self.start_search(f'次は {ja(self.target)} を探します')
                else:
                    self.set_state(self.DONE, 'すべての物体を発見しました!')

        self.publish(cmd)

    def approach(self, now, yaw):
        cmd = Twist()
        if self.last_seen_t is None or now - self.last_seen_t > self.lost_timeout:
            # lost it: spin towards where it was last seen
            if self.target_world_yaw is not None:
                self.spin_dir = 1.0 if wrap(self.target_world_yaw - yaw) >= 0 else -1.0
            self.hits = 0
            self.set_state(self.SPIN, f'{ja(self.target)} を見失いました。再探索')
            return cmd

        err = wrap(self.target_world_yaw - yaw)
        dist = self.fuse_distance(self.range_at(err), self.ground_distance())
        big_bbox = self.last_bbox is not None and self.last_bbox[2] > 0.85 * self.img_h
        if dist <= self.stop_distance or big_bbox:
            self.record_found(dist, err)
            return cmd

        cmd.angular.z = clamp(self.k_bearing * err, -self.max_ang, self.max_ang)
        heading_factor = clamp(1.0 - abs(err) / 0.6, 0.0, 1.0)
        dist_factor = clamp((dist - self.stop_distance) / 0.8, 0.25, 1.0)
        cmd.linear.x = self.max_lin * heading_factor * dist_factor
        if now - self.last_seen_t > 0.6:
            # target temporarily not detected: keep heading for where it was, but slower
            cmd.linear.x *= 0.6

        # obstacle between the robot and the target (e.g. a suitcase): side-step around it.
        # Once started, the manoeuvre is kept for a short time so it does not oscillate
        # between "turn away" and "turn back to the target".
        front = self.range_at(0.0, math.radians(20))
        if now >= self.avoid_until and front < self.obstacle_distance + 0.15 and dist > front + 0.3:
            left = self.range_at(math.radians(60), math.radians(25))
            right = self.range_at(math.radians(-60), math.radians(25))
            self.avoid_dir = 1.0 if left >= right else -1.0
            self.avoid_until = now + 2.5
            self.event = '目標の手前に障害物 → 回避して接近'
        if now < self.avoid_until:
            if front < self.obstacle_distance + 0.25:
                cmd.linear.x, cmd.angular.z = 0.0, self.avoid_dir * self.max_ang
            else:
                cmd.linear.x, cmd.angular.z = 0.25, 0.0
        return cmd

    def remember(self, msg):
        """Add every confident detection to the semantic memory (world-frame positions)."""
        t_img = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        yaw_img = self.yaw_at(t_img)
        x, y, yaw = self.pose
        for d in msg.detections:
            if not d.results:
                continue
            h = d.results[0].hypothesis
            if h.score < self.memory_conf:
                continue
            cx = d.bbox.center.position.x
            v_bottom = d.bbox.center.position.y + d.bbox.size_y / 2.0
            world_yaw = wrap(yaw_img + math.atan2(-(cx - self.img_w / 2.0), self.fx))
            d_lidar = self.range_at(wrap(world_yaw - yaw))
            dv = v_bottom - self.img_h / 2.0
            d_ground = (self.cam_h * self.fx / dv if dv > 4.0 and v_bottom < self.img_h - 2
                        else float('inf'))
            dist = self.fuse_distance(d_lidar, d_ground)
            if dist > self.memory_max_dist:
                continue
            px, py, _ = self.to_world(x + (dist + 0.15) * math.cos(world_yaw),
                                      y + (dist + 0.15) * math.sin(world_yaw))
            # a found object of another class at the same place -> misclassification
            if any(f['class'] != h.class_id and math.dist((px, py), (f['x'], f['y'])) <
                   self.exclusion_radius for f in self.found):
                continue
            entries = self.memory.setdefault(h.class_id, [])
            for e in entries:
                if math.dist((px, py), (e['x'], e['y'])) < 1.0:
                    k = min(e['n'], 20)
                    e['x'] = (e['x'] * k + px) / (k + 1)
                    e['y'] = (e['y'] * k + py) / (k + 1)
                    e['n'] += 1
                    e['score'] = max(e['score'], h.score)
                    break
            else:
                entries.append({'x': px, 'y': py, 'n': 1, 'score': h.score, 'checked': False})

    def recall(self, cls):
        """Best remembered, not yet checked, position of `cls` (world frame), or None."""
        cands = [e for e in self.memory.get(cls, []) if e['n'] >= 3 and not e['checked']
                 and not any(math.dist((e['x'], e['y']), (f['x'], f['y'])) < self.exclusion_radius
                             for f in self.found)]
        return max(cands, key=lambda e: e['n']) if cands else None

    def start_search(self, event=None):
        """Begin looking for the current target: go to its remembered position if known."""
        if not self.targets:
            self.set_state(self.DONE, event or 'すべての物体を発見しました!')
            return
        if self.goto_memory(event):
            return
        self.set_state(self.SPIN, event)

    def goto_memory(self, prefix=None):
        e = self.recall(self.target)
        if e is None or self.pose is None:
            return False
        wx, wy, _ = self.to_world(*self.pose)
        if math.dist((wx, wy), (e['x'], e['y'])) < 2.0:
            e['checked'] = True
            self.set_state(self.SPIN, (prefix + ' / ' if prefix else '') +
                           f'{ja(self.target)} はこの近くで見かけたはず → 周囲を確認')
            return True
        wp = self.map.plan_to(wx, wy, e['x'], e['y'], radius=1.6)
        if not wp:
            return False
        e['checked'] = True
        self.waypoints = wp
        self.leg_t0 = self.now()
        self.set_state(self.MOVE, f'記憶: {ja(self.target)} は ({e["x"]:.1f}, {e["y"]:.1f}) 付近で'
                                  '見かけた → 直行')
        return True

    @staticmethod
    def fuse_distance(d_lidar, d_ground):
        """Combine the LiDAR range and the monocular ground-plane estimate.

        * both agree (or no image estimate)  -> LiDAR (more precise)
        * LiDAR much shorter than the image  -> the beam hit something in front of the object
                                                (occlusion), use the image estimate
        * LiDAR much longer                  -> the beam passed e.g. between chair legs,
                                                use the image estimate
        """
        if not math.isfinite(d_ground):
            return d_lidar
        if d_lidar < 0.75 * d_ground or d_lidar > 1.3 * d_ground:
            return d_ground
        return d_lidar

    def ground_distance(self):
        """Monocular distance from the bottom edge of the box (object standing on the floor).

        With a level camera at height h, a floor point seen dv pixels below the image centre is
        at distance h * fy / dv. This complements the LiDAR, whose single scan plane can pass
        between the thin legs of a chair.
        """
        if self.last_bbox is None or self.now() - self.last_seen_t > 1.0:
            return float('inf')
        dv = self.last_bbox[4] - self.img_h / 2.0
        if dv < 4.0 or self.last_bbox[4] >= self.img_h - 2:   # above horizon / cut off
            return float('inf')
        return self.cam_h * self.fx / dv

    def near_found_object(self, yaw_ctrl):
        """True if the LiDAR point in direction yaw_ctrl is close to an object already found."""
        if not self.found or self.pose is None:
            return False
        x, y, yaw = self.pose
        rel = wrap(yaw_ctrl - yaw)
        d = min(self.range_at(rel), 6.0) + 0.15
        px, py, _ = self.to_world(x + d * math.cos(yaw_ctrl), y + d * math.sin(yaw_ctrl))
        return any(math.dist((px, py), (f['x'], f['y'])) < self.exclusion_radius
                   for f in self.found)

    def record_found(self, dist, err):
        x, y, yaw = self.pose
        d = min(dist, 3.0) + 0.15      # LiDAR hits the surface; add a rough object radius
        ox, oy = x + d * math.cos(yaw + err), y + d * math.sin(yaw + err)
        wx, wy, _ = self.to_world(ox, oy)
        item = {'class': self.target, 'name_ja': ja(self.target),
                'x': round(wx, 2), 'y': round(wy, 2), 't': round(self.now() - self.t_start, 1)}
        if self.odom_world is not None and self.gt_pose is not None:
            # how far pure wheel odometry has drifted by now (for the report)
            item['odom_drift'] = round(math.dist(self.odom_world[:2], self.gt_pose[:2]), 3)
        self.found.append(item)
        self.found_pub.publish(String(data=json.dumps(item, ensure_ascii=False)))
        self.get_logger().info('FOUND ' + json.dumps(item, ensure_ascii=False))
        self.arrived_t = self.now()
        name = ja(self.target)
        self.targets.pop(0)
        self.hits = 0
        self.set_state(self.ARRIVED,
                       f'{name} に到着! 位置 ({item["x"]:.1f}, {item["y"]:.1f}) を記録')

    def follow_path(self, now):
        cmd = Twist()
        wx, wy, wyaw = self.to_world(*self.pose)
        tx, ty = self.waypoints[0]
        if math.hypot(tx - wx, ty - wy) < 0.3:
            self.waypoints.pop(0)
            if not self.waypoints:
                self.set_state(self.SPIN)
            return cmd
        err = wrap(math.atan2(ty - wy, tx - wx) - wyaw)
        front = self.range_at(0.0, math.radians(20))
        if now - self.leg_t0 > 40.0:
            self.waypoints = []
            self.set_state(self.SPIN)
        elif abs(err) > 0.5:
            cmd.angular.z = clamp(2.0 * err, -self.max_ang, self.max_ang)
        elif front < self.obstacle_distance:
            self.waypoints = []           # something unexpected in the way: look around, replan
            self.set_state(self.SPIN)
        else:
            cmd.linear.x = (self.max_lin * clamp((front - self.obstacle_distance) / 0.6, 0.3, 1.0)
                            * clamp(1.0 - abs(err) / 0.8, 0.3, 1.0))
            cmd.angular.z = clamp(2.0 * err, -self.max_ang, self.max_ang)
        return cmd

    def choose_exploration_goal(self):
        """Memory first, then frontier exploration, then the simple open-direction heuristic."""
        if self.goto_memory():
            return
        if self.use_frontier:
            wx, wy, _ = self.to_world(*self.pose)
            wp = self.map.plan(wx, wy)
            if not wp:
                fr, trav = self.map.frontier_mask()
                sx, sy = self.map.cell(wx, wy)
                self.get_logger().warn(
                    f'no frontier path: frontiers={int(fr.sum())} trav={int(trav.sum())} '
                    f'start_trav={bool(trav[sy, sx])} seen={int(self.map.seen.sum())}')
            if wp:
                self.waypoints = wp
                self.leg_t0 = self.now()
                self.set_state(self.MOVE,
                               f'{ja(self.target)} が見つからないので、まだ見ていない場所へ移動')
                return
        self.choose_open_direction()

    def choose_open_direction(self):
        """Pick the open direction whose end point is farthest from places already explored."""
        x, y, yaw = self.pose
        best = None
        for deg in range(-180, 180, 10):
            rel = math.radians(deg)
            free = self.range_at(rel, math.radians(10))
            step = min(free - 0.6, self.explore_step)
            if step < 0.6:
                continue
            gx = x + step * math.cos(yaw + rel)
            gy = y + step * math.sin(yaw + rel)
            novelty = min((math.dist((gx, gy), v) for v in self.visited), default=5.0)
            score = step + 1.5 * min(novelty, 4.0)
            if best is None or score > best[0]:
                best = (score, wrap(yaw + rel), step)
        if best is None:          # boxed in: just turn around and try again
            self.goal_yaw, self.leg_len = wrap(yaw + math.pi), 0.0
        else:
            _, self.goal_yaw, self.leg_len = best
        self.spin_dir = 1.0 if wrap(self.goal_yaw - yaw) >= 0 else -1.0
        self.set_state(self.TURN_TO, f'{ja(self.target)} が見つからないので移動して探します')

    def publish_map(self):
        m = self.map
        grid = OccupancyGrid()
        grid.header.frame_id = 'world'
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = float(m.res)
        grid.info.width = m.n
        grid.info.height = m.n
        grid.info.origin.position.x = float(m.ox)
        grid.info.origin.position.y = float(m.oy)
        data = (m.occ == 2).astype('int8') * 100
        data[(m.occ != 2) & ~m.seen] = -1
        grid.data = data.flatten().tolist()
        self.map_pub.publish(grid)

    def publish(self, cmd):
        self.cmd_pub.publish(cmd)
        if self.pose is None:
            return
        wx, wy, wyaw = self.to_world(*self.pose)
        shown = self.found[-1]['class'] if self.state == self.ARRIVED and self.found else self.target
        status = {
            'state': self.state,
            'target': shown,
            'target_ja': ja(shown) if shown else None,
            'path': [[round(a, 2), round(b, 2)] for a, b in self.waypoints],
            'memory': [{'class': c, 'name_ja': ja(c), 'x': round(e['x'], 2), 'y': round(e['y'], 2),
                        'n': e['n']}
                       for c, es in self.memory.items() for e in es if e['n'] >= 3],
            'command': self.last_command,
            'remaining': list(self.targets),
            'found': self.found,
            'event': self.event,
            'hits': self.hits,
            'robot': [round(wx, 3), round(wy, 3), round(wyaw, 3)],
            'cmd': [round(cmd.linear.x, 2), round(cmd.angular.z, 2)],
        }
        self.status_pub.publish(String(data=json.dumps(status, ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = SearchController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
            node.destroy_node()
        except Exception:  # context already shut down by the launch system
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
