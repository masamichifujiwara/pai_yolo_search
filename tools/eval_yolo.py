"""Run YOLO models on captured frames and report the score of a target class (debug helper)."""
import glob
import sys
import time

from ultralytics import YOLO

img_dir, target = sys.argv[1], sys.argv[2]
models = sys.argv[3:] or ['yolo11n.pt', 'yolo11s.pt']
files = sorted(glob.glob(f'{img_dir}/d*.png'))
for mp in models:
    m = YOLO(mp)
    m.predict(files[0], verbose=False)
    print(f'== {mp.split("/")[-1]}')
    for f in files:
        t = time.perf_counter()
        r = m.predict(f, conf=0.1, verbose=False)[0]
        dt = (time.perf_counter() - t) * 1000
        dets = [(r.names[int(c)], float(s)) for c, s in zip(r.boxes.cls, r.boxes.conf)]
        best = max([s for n, s in dets if n == target], default=0.0)
        others = ', '.join(f'{n}:{s:.2f}' for n, s in dets if n != target)[:80]
        print(f'  {f.split("/")[-1]:18s} {target}={best:.2f}  ({dt:.0f} ms)  others: {others}')
