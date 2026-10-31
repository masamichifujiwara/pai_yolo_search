"""Grid map for frontier-based exploration.

Two layers are kept on a 2D grid (world frame):
  * occupancy from every LiDAR ray (walls / obstacles, and which cells are known to be free)
  * "seen" cells: free cells that were inside the *camera* field of view (within view_range).
    YOLO can only find an object that the camera has looked at, so the exploration frontier is
    defined on this layer and not on the 360 deg LiDAR coverage.

plan() returns waypoints (BFS on the grid, obstacles inflated by the robot radius) to the
nearest frontier, i.e. a seen cell next to a not-yet-seen cell.
"""
import math
from collections import deque

import numpy as np

UNKNOWN, FREE, OCC = 0, 1, 2


class ExplorationMap:

    def __init__(self, resolution=0.25, size=16.0, origin=(-8.0, -8.0), view_range=5.0,
                 camera_hfov=1.3, inflate_cells=2):
        self.res = resolution
        self.n = int(round(size / resolution))
        self.ox, self.oy = origin
        self.view_range = view_range
        self.half_fov = camera_hfov / 2.0
        self.inflate = inflate_cells
        self.occ = np.zeros((self.n, self.n), np.int8)     # UNKNOWN / FREE / OCC   [iy, ix]
        self.hits = np.zeros((self.n, self.n), np.int8)    # evidence counter for OCC
        self.seen = np.zeros((self.n, self.n), bool)
        self._steps = np.arange(0.0, view_range, resolution * 0.5)

    # ------------------------------------------------------------------ geometry
    def cell(self, x, y):
        return int((x - self.ox) / self.res), int((y - self.oy) / self.res)

    def center(self, ix, iy):
        return self.ox + (ix + 0.5) * self.res, self.oy + (iy + 0.5) * self.res

    def _valid(self, ix, iy):
        return (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)

    # ------------------------------------------------------------------ mapping
    def update(self, pose, scan):
        x, y, yaw = pose
        n = len(scan.ranges)
        rel = scan.angle_min + np.arange(n) * scan.angle_increment
        ang = rel + yaw
        r = np.asarray(scan.ranges, dtype=np.float64)
        r = np.where(np.isfinite(r), r, scan.range_max)
        reach = np.minimum(r, self.view_range)

        d = self._steps[None, :]
        free_mask = d < (reach[:, None] - self.res * 0.5)
        X = x + d * np.cos(ang)[:, None]
        Y = y + d * np.sin(ang)[:, None]
        ix = ((X - self.ox) / self.res).astype(np.int64)
        iy = ((Y - self.oy) / self.res).astype(np.int64)

        m = free_mask & self._valid(ix, iy)
        fx, fy = ix[m], iy[m]
        # a beam passing through a cell is evidence against an obstacle there
        self.hits[fy, fx] = np.maximum(self.hits[fy, fx] - 1, 0)
        self.occ[fy, fx] = np.where(self.hits[fy, fx] >= 2, OCC, FREE)

        in_fov = (np.abs(np.arctan2(np.sin(rel), np.cos(rel))) <= self.half_fov)[:, None]
        mc = m & in_fov
        self.seen[iy[mc], ix[mc]] = True

        hit = r < self.view_range
        hx = ((x + r[hit] * np.cos(ang[hit]) - self.ox) / self.res).astype(np.int64)
        hy = ((y + r[hit] * np.sin(ang[hit]) - self.oy) / self.res).astype(np.int64)
        v = self._valid(hx, hy)
        hx, hy = hx[v], hy[v]
        # an obstacle needs repeated hits (a single beam hitting the floor while the robot
        # pitches, or a grazing ray, must not block the planner)
        self.hits[hy, hx] = np.minimum(self.hits[hy, hx] + 2, 6)
        self.occ[hy, hx] = np.where(self.hits[hy, hx] >= 4, OCC, self.occ[hy, hx])

    # ------------------------------------------------------------------ planning
    @staticmethod
    def _dilate(mask, k=1):
        out = mask.copy()
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                out |= np.roll(np.roll(mask, dy, 0), dx, 1)
        return out

    def frontier_mask(self):
        occ = self.occ == OCC
        known_free = (self.occ == FREE) | self.seen
        trav = known_free & ~self._dilate(occ, self.inflate)
        unseen = ~self.seen & ~occ
        nb = np.zeros_like(unseen)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb |= np.roll(unseen, (dy, dx), (0, 1))
        frontier = self.seen & trav & nb
        cnt = sum(np.roll(frontier, (dy, dx), (0, 1)).astype(np.int8)
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        return frontier & (cnt >= 3), trav

    def reset_seen(self):
        self.seen[:] = False

    def plan_to(self, x, y, gx, gy, radius=1.5, max_len=12.0, waypoint_spacing=0.75):
        """Waypoints to the nearest traversable cell within `radius` of (gx, gy), or None."""
        _, trav = self.frontier_mask()
        ys, xs = np.mgrid[0:self.n, 0:self.n]
        cxs, cys = self.ox + (xs + 0.5) * self.res, self.oy + (ys + 0.5) * self.res
        goal_mask = ((cxs - gx) ** 2 + (cys - gy) ** 2 <= radius ** 2) & trav
        return self._bfs(x, y, goal_mask, trav, 0.0, max_len, waypoint_spacing)

    def plan(self, x, y, min_dist=1.0, max_len=4.0, waypoint_spacing=0.75):
        """Waypoints [(x, y), ...] towards the nearest frontier, or None."""
        frontier, trav = self.frontier_mask()
        return self._bfs(x, y, frontier, trav, min_dist, max_len, waypoint_spacing)

    def _bfs(self, x, y, goal_mask, trav, min_dist, max_len, waypoint_spacing):
        frontier = goal_mask
        sx, sy = self.cell(x, y)
        if not self._valid(sx, sy):
            return None
        free_any = (self.occ == FREE) | self.seen
        dist = np.full((self.n, self.n), -1, np.int32)
        dist[sy, sx] = 0
        parent = {}
        q = deque([(sx, sy)])
        goal = None
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        while q:
            cx, cy = q.popleft()
            if frontier[cy, cx] and dist[cy, cx] * self.res >= min_dist:
                goal = (cx, cy)
                break
            for dx, dy in nbrs:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.n and 0 <= ny < self.n) or dist[ny, nx] >= 0:
                    continue
                # cells right next to obstacles are allowed only to leave the start area
                if trav[ny, nx] or (free_any[ny, nx] and dist[cy, cx] < self.inflate + 1):
                    dist[ny, nx] = dist[cy, cx] + 1
                    parent[(nx, ny)] = (cx, cy)
                    q.append((nx, ny))
        if goal is None:
            return None
        path = [goal]
        while path[-1] != (sx, sy):
            path.append(parent[path[-1]])
        path.reverse()
        pts = [self.center(cx, cy) for cx, cy in path]

        waypoints, travelled, acc = [], 0.0, 0.0
        for i in range(1, len(pts)):
            seg = math.dist(pts[i - 1], pts[i])
            travelled += seg
            acc += seg
            if travelled >= max_len:
                waypoints.append(pts[i])
                break
            if acc >= waypoint_spacing or i == len(pts) - 1:
                waypoints.append(pts[i])
                acc = 0.0
        return waypoints or None

    # ------------------------------------------------------------------ export
    def coverage_cells(self):
        """Indices (ix, iy) of seen cells, for visualisation."""
        iy, ix = np.nonzero(self.seen)
        return ix, iy
