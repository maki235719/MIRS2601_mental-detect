#!/usr/bin/env python3
"""リポジトリ内に残っている生成物・モデルを DATA_DIR / MODEL_DIR へ引っ越す。

旧仕様では出力先がスクリプトと同じフォルダ（＝リポジトリ内）だったため、
person_profiles.json や face_landmarker.task がリポジトリを汚していた。
このスクリプトを1回実行すると作業ツリーがきれいになり、以降 git pull が
強制上書きなしで通るようになる。

使い方（Windows でも Pi でも同じ）:
    python scripts/migrate_local_data.py          # 実際に移動
    python scripts/migrate_local_data.py --dry-run # 何が動くか確認だけ
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, MODEL_DIR, REPO_DIR  # noqa: E402

# (ファイル名, 移動先ディレクトリ)
MOVES = [
    ("person_profiles.json", DATA_DIR),
    ("session_history.jsonl", DATA_DIR),
    ("stai_dataset.jsonl", DATA_DIR),
    ("stress_log.csv", DATA_DIR),
    ("stress_graph.png", DATA_DIR),
    ("stress_config.json", DATA_DIR),
    ("face_landmarker.task", MODEL_DIR),
]

PRUNE_DIRS = ["__pycache__", os.path.join("coral", "__pycache__")]


def main():
    dry = "--dry-run" in sys.argv
    moved = kept = 0

    for name, dest_dir in MOVES:
        src = os.path.join(REPO_DIR, name)
        if not os.path.exists(src):
            continue
        os.makedirs(dest_dir, exist_ok=True)
        dst = os.path.join(dest_dir, name)
        if os.path.exists(dst):
            # 移動先が既にある＝新しい方を正とし、リポジトリ側は削除
            print(f"[del ] {name} (移動先に既存: {dst})")
            if not dry:
                os.remove(src)
            kept += 1
            continue
        print(f"[move] {name} -> {dst}")
        if not dry:
            shutil.move(src, dst)
        moved += 1

    for d in PRUNE_DIRS:
        p = os.path.join(REPO_DIR, d)
        if os.path.isdir(p):
            print(f"[rmdir] {d}")
            if not dry:
                shutil.rmtree(p, ignore_errors=True)

    print(f"\n移動 {moved} 件 / 重複削除 {kept} 件")
    print(f"DATA_DIR : {DATA_DIR}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    if dry:
        print("(--dry-run なので実際には何もしていません)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
