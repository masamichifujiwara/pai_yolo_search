"""Post-production of the submission video.

  title card -> explanation card -> main recording (sped up, with a "2倍速" badge) -> result card

usage:
  python3 make_video.py <raw.mp4> <results.json> <out.mp4> [--speed 2.0] [--trim-start S]

results.json:
  {"found": [{"name_ja": "消火栓", "x": 3.2, "y": -3.7, "gt": [3.2, -3.6], "t": 75.0}, ...],
   "total_time": 290.0, "repo": "https://github.com/..."}
"""
import argparse
import json
import math
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_B = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
BG = (24, 28, 36)
ACCENT = (255, 196, 60)


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold and os.path.exists(FONT_B) else FONT, size)


def card(lines, path):
    """lines: list of (text, size, color, bold, indent)"""
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 8], fill=ACCENT)
    y = 70
    for text, size, color, bold, indent in lines:
        if text:
            d.text((80 + indent, y), text, font=font(size, bold), fill=color)
        y += int(size * 1.55)
    img.save(path)


def badge(speed, path):
    img = Image.new('RGBA', (170, 52), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, 169, 51], radius=12, fill=(220, 40, 40, 230))
    d.text((18, 6), f'▶▶ {speed:g}倍速', font=font(28, True), fill=(255, 255, 255, 255))
    img.save(path)


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw')
    ap.add_argument('results')
    ap.add_argument('out')
    ap.add_argument('--speed', type=float, default=2.0)
    ap.add_argument('--trim-start', type=float, default=0.0)
    ap.add_argument('--trim-end', type=float, default=0.0, help='cut this many seconds at the end')
    a = ap.parse_args()
    res = json.load(open(a.results))
    tmp = tempfile.mkdtemp(prefix='pai_video_')
    white, grey, green = (255, 255, 255), (190, 196, 206), (120, 235, 140)

    card([
        ('PAI 最終課題', 30, grey, False, 0),
        ('AI物体探索ロボット', 64, white, True, 0),
        ('YOLO × ROS 2 × Gazebo', 36, ACCENT, False, 0),
        ('', 20, white, False, 0),
        ('指定した物体を、カメラ画像のAI推論で自律的に探し出し、', 30, white, False, 0),
        ('近づいて、その位置を地図に記録するロボット', 30, white, False, 0),
        ('', 20, white, False, 0),
        ('ベース: 最終課題サンプル「gazeboとROS2を連携してロボットを操作する」', 22, grey, False, 0),
        (f'※ 本編はシミュレーションを {a.speed:g} 倍速で再生しています', 22, grey, False, 0),
    ], f'{tmp}/title.png')

    card([
        ('しくみ', 44, ACCENT, True, 0),
        ('1. 前方カメラの画像を YOLO11（COCO 80クラス学習済み）で推論', 27, white, False, 0),
        ('2. 見えなければ 360°回転 → カメラでまだ見ていない場所へ移動（フロンティア探索）', 27, white,
         False, 0),
        ('3. 発見したら 向きを合わせて接近し、LiDAR と画像から距離を測って停止・位置を記録', 27, white,
         False, 0),
        ('4. 目標以外も含め、見かけた物はすべて位置を記憶（意味地図）', 27, white, False, 0),
        ('', 14, white, False, 0),
        ('ミッション:  消火栓 → 椅子 を探す（どちらもスタート地点からは見えない）', 26, green, False, 0),
        ('その後:  「人を探して」と日本語で指示 → 探索中に見かけた記憶から直行', 26, green, False, 0),
        ('人は途中で何度も見えるが、その時点の目標ではないので無視（位置だけ記憶）', 22, grey, False, 0),
        ('', 14, white, False, 0),
        ('サンプルからの発展: カメラ・LiDAR / AI推論 / 自律探索 / 地図と記憶 / 日本語の指示', 22, grey,
         False, 0),
    ], f'{tmp}/explain.png')

    lines = [('結果', 44, ACCENT, True, 0)]
    for i, f in enumerate(res['found']):
        gx, gy = f['gt']
        err = math.hypot(f['x'] - gx, f['y'] - gy)
        lines.append((f'{i + 1}. {f["name_ja"]}:  推定位置 ({f["x"]:.1f}, {f["y"]:.1f}) m   '
                      f'真値 ({gx:.1f}, {gy:.1f}) m   誤差 {err:.2f} m   ({f["t"]:.0f} 秒)',
                      26, white, False, 0))
    lines += [
        ('', 16, white, False, 0),
        (res.get('summary', f'すべての物体を自律的に発見（{res["total_time"]:.0f} 秒）'), 28, green,
         True, 0),
        ('', 16, white, False, 0),
        ('今後: SLAMによる自己位置推定 / 任意の言葉で探せるオープン語彙検出 / 実機への展開', 24,
         grey, False, 0),
    ]
    if res.get('repo'):
        lines.append((f'コード: {res["repo"]}', 22, grey, False, 0))
    card(lines, f'{tmp}/result.png')
    badge(a.speed, f'{tmp}/badge.png')

    enc = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-r', '30']
    for name, dur in (('title', 5), ('explain', 10), ('result', 8)):
        run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-t', str(dur), '-i', f'{tmp}/{name}.png',
             '-vf', 'format=yuv420p', *enc, f'{tmp}/{name}.mp4'])

    dur = float(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', a.raw]))
    end = dur - a.trim_end
    run(['ffmpeg', '-v', 'error', '-y', '-ss', str(a.trim_start), '-to', str(end), '-i', a.raw,
         '-i', f'{tmp}/badge.png', '-filter_complex',
         f'[0:v]setpts=PTS/{a.speed}[v];[v][1:v]overlay=14:H-h-14', *enc, f'{tmp}/main.mp4'])

    with open(f'{tmp}/list.txt', 'w') as fh:
        for name in ('title', 'explain', 'main', 'result'):
            fh.write(f"file '{tmp}/{name}.mp4'\n")
    run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', f'{tmp}/list.txt',
         '-c', 'copy', '-movflags', '+faststart', a.out])
    total = float(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', a.out]))
    print(f'wrote {a.out}  ({total:.1f} s)  cards in {tmp}')


if __name__ == '__main__':
    main()
