"""
複数環境マージツール（オフライン）

各収集環境（PC）は face_detect/ 一式を OneDrive/SharePoint 同期フォルダの
サブフォルダに置いて実行する運用を想定:

    <root>\\envA\\face_detect\\session_history.jsonl
    <root>\\envA\\face_detect\\stai_dataset.jsonl
    <root>\\envB\\face_detect\\session_history.jsonl
    ...

このスクリプトは <root> 直下の各サブフォルダを1環境として、
session_history.jsonl / stai_dataset.jsonl を読み、
person_id を "envA:1" のように環境名で名前空間化して衝突を防ぎつつ
1本のファイルに結合する。差分マージはせず毎回フルリビルドする（冪等）。

使い方:
    python merge_data.py --root D:\\OneDrive\\MIRS_stress_data
    python merge_data.py --root D:\\OneDrive\\MIRS_stress_data --out .\\merged
"""

import argparse
import json
from pathlib import Path

FILES = ["session_history.jsonl", "stai_dataset.jsonl"]


def load_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def merge(root, out_dir):
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    envs = sorted(p for p in root.iterdir() if p.is_dir())
    merged = {name: [] for name in FILES}
    summary = []

    for env_dir in envs:
        env = env_dir.name
        src_dir = env_dir / "face_detect"
        if not src_dir.is_dir():
            continue
        counts = {}
        for name in FILES:
            src = src_dir / name
            if not src.exists():
                counts[name] = 0
                continue
            recs = load_jsonl(src)
            for r in recs:
                r["env"] = env
                if r.get("person_id") is not None:
                    r["local_person_id"] = r["person_id"]
                    r["person_id"] = f"{env}:{r['person_id']}"
            merged[name].extend(recs)
            counts[name] = len(recs)
        summary.append((env, counts))

    for name in FILES:
        out_path = out_dir / name
        with open(out_path, "w", encoding="utf-8") as f:
            for r in merged[name]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"環境数: {len(summary)}")
    for env, counts in summary:
        parts = ", ".join(f"{name}={n}" for name, n in counts.items())
        print(f"  {env}: {parts}")
    people = {r["person_id"] for r in merged["stai_dataset.jsonl"]}
    print(f"合計 stai_dataset.jsonl: {len(merged['stai_dataset.jsonl'])} 件（{len(people)} 名）")
    print(f"合計 session_history.jsonl: {len(merged['session_history.jsonl'])} 件")
    print(f"出力先: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="複数環境データのマージ")
    ap.add_argument("--root", required=True, help="環境フォルダ（envA, envB, ...）を含む親ディレクトリ")
    ap.add_argument("--out", default="merged", help="マージ結果の出力先（既定: ./merged）")
    args = ap.parse_args()
    merge(args.root, args.out)


if __name__ == "__main__":
    main()
