#!/usr/bin/env python3
"""モデルファイルを事前にダウンロードしてキャッシュへ置く。

初回起動時の待ち時間をなくすためのもの。既にファイルがあれば何もしない
（＝バージョン更新のたびに再ダウンロードされない）。

使い方:
    python scripts/prefetch_models.py
"""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (  # noqa: E402
    FACE_LANDMARKER_TASK,
    FACE_LANDMARKER_URL,
    SFACE_WEIGHT_PATH,
    SFACE_WEIGHT_URL,
)

TARGETS = [
    ("MediaPipe FaceLandmarker", FACE_LANDMARKER_URL, FACE_LANDMARKER_TASK),
    ("DeepFace SFace weights", SFACE_WEIGHT_URL, SFACE_WEIGHT_PATH),
]


def fetch(name, url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {name}: 既に存在 -> {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[get ] {name} -> {dest}")
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    print(f"[ok  ] {name} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def main():
    failed = []
    for name, url, dest in TARGETS:
        try:
            fetch(name, url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {name}: {exc}")
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
