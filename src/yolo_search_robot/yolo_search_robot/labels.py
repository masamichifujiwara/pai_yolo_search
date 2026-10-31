"""COCO class names <-> Japanese names, and a tiny natural-language command parser."""

# Japanese display names for COCO classes that are likely to be used as search targets.
JA_NAMES = {
    'person': '人',
    'bicycle': '自転車',
    'car': '車',
    'motorcycle': 'バイク',
    'bus': 'バス',
    'truck': 'トラック',
    'traffic light': '信号機',
    'fire hydrant': '消火栓',
    'stop sign': '一時停止標識',
    'bench': 'ベンチ',
    'bird': '鳥',
    'cat': '猫',
    'dog': '犬',
    'horse': '馬',
    'backpack': 'リュック',
    'umbrella': '傘',
    'handbag': 'ハンドバッグ',
    'suitcase': 'スーツケース',
    'sports ball': 'ボール',
    'bottle': 'ボトル',
    'cup': 'コップ',
    'bowl': 'ボウル',
    'chair': '椅子',
    'couch': 'ソファ',
    'potted plant': '観葉植物',
    'bed': 'ベッド',
    'dining table': 'テーブル',
    'toilet': 'トイレ',
    'tv': 'テレビ',
    'laptop': 'ノートPC',
    'refrigerator': '冷蔵庫',
    'book': '本',
    'clock': '時計',
    'vase': '花瓶',
    'teddy bear': 'ぬいぐるみ',
}

# Extra spellings accepted in commands (hiragana / katakana / synonyms).
ALIASES = {
    'person': ['人間', 'ひと', 'ヒト', '人物', '人'],
    'fire hydrant': ['消火栓', 'しょうかせん', 'ハイドラント'],
    'chair': ['椅子', 'いす', 'イス', 'チェア'],
    'stop sign': ['一時停止標識', '止まれ標識', '停止標識', 'ストップサイン', '標識'],
    'potted plant': ['観葉植物', '植木', '鉢植え', '植物'],
    'suitcase': ['スーツケース', 'キャリーケース'],
    'sports ball': ['ボール', 'ぼーる'],
    'bottle': ['ボトル', 'ペットボトル', '瓶'],
    'cup': ['コップ', 'カップ'],
    'dog': ['犬', 'いぬ', 'イヌ'],
    'cat': ['猫', 'ねこ', 'ネコ'],
    'couch': ['ソファ', 'ソファー'],
    'dining table': ['テーブル', '机'],
    'tv': ['テレビ'],
    'backpack': ['リュック', 'バックパック'],
    'car': ['車', 'くるま', '自動車'],
    'bicycle': ['自転車'],
    'bench': ['ベンチ'],
}

STOP_WORDS = ('stop', '止ま', '停止して', 'ストップ', 'とまれ', '中止')
MANUAL_WORDS = ('manual', '手動')
AUTO_WORDS = ('auto', '自動', '再開', 'resume')


def ja(name: str) -> str:
    return JA_NAMES.get(name, name)


def parse_targets(text: str, known_classes=None):
    """Return the COCO classes mentioned in `text`, in the order they appear.

    Both English COCO names ("fire hydrant") and Japanese aliases ("消火栓") are recognised.
    """
    lowered = text.lower()
    hits = []  # (position, class)
    candidates = dict(ALIASES)
    for cls in (known_classes or JA_NAMES.keys()):
        candidates.setdefault(cls, [])
    for cls, words in candidates.items():
        best = None
        for w in [cls] + list(words) + [JA_NAMES.get(cls, '')]:
            if not w:
                continue
            pos = lowered.find(w.lower())
            if pos >= 0 and (best is None or pos < best):
                best = pos
        if best is not None:
            hits.append((best, cls))
    hits.sort()
    out = []
    for _, cls in hits:
        if cls not in out:
            out.append(cls)
    return out


def classify_command(text: str):
    """Return one of 'stop', 'manual', 'auto', 'targets' or None."""
    t = text.lower()
    if any(w in t for w in STOP_WORDS):
        return 'stop'
    if any(w in t for w in MANUAL_WORDS):
        return 'manual'
    if parse_targets(text):
        return 'targets'
    if any(w in t for w in AUTO_WORDS):
        return 'auto'
    return None
