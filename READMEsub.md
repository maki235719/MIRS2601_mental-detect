# 複数環境データの同期・マージ手順

複数のPC（環境）で `peroson_detect.py` を使ってストレスデータを収集し、あとでまとめて分析するための手順です。

## 1. 共有フォルダを用意する（管理者が1回だけ）

1. SharePoint（またはOneDrive）に研究チーム専用の共有フォルダを作成する（例: `MIRS_stress_data`）。
2. 共有先は**組織内の特定メンバーのみ**に限定する。「リンクを知っている全員」など公開範囲の広い共有は使わない。

## 2. 各収集PCでの準備

1. 共有フォルダをOneDriveクライアントで同期する。
2. 同期フォルダの中に、そのPC/環境専用のサブフォルダを作り、`face_detect/` 一式（このプロジェクト全体）をコピーして配置する。フォルダ名がそのまま環境名になる。

```
MIRS_stress_data\           ← OneDrive同期フォルダ
  envA\face_detect\         ← 環境A（例: 教室PC）
  envB\face_detect\         ← 環境B（例: 自宅PC）
```

3. 通常どおり、その場所で `peroson_detect.py` を実行してデータを収集する。コード側の設定変更は不要。出力される `session_history.jsonl` / `stai_dataset.jsonl` / `person_profiles.json` は自動的にOneDrive経由で共有フォルダに同期される。

## 3. データをマージする

収集がある程度たまったら、共有フォルダにアクセスできる任意のPCで以下を実行する。

```powershell
python merge_data.py --root "D:\OneDrive\MIRS_stress_data" --out .\merged
```

- 各環境の `session_history.jsonl` / `stai_dataset.jsonl` を読み込み、`person_id` を `envA:1` のように環境名で区別して1つのファイルに統合する（別環境の同じ番号の人物を混同しない）。
- 何度実行しても同じ結果になる（毎回まるごと作り直すだけで、差分マージや重複は発生しない）。
- 実行後、環境ごとの件数と合計人数がコンソールに表示される。

## 4. マージ結果を使ってチューニングする

```powershell
python tune_stress.py --report --dataset .\merged\stai_dataset.jsonl
python tune_stress.py --apply  --dataset .\merged\stai_dataset.jsonl
```

`--dataset` を省略すると、通常どおりそのフォルダ内の `stai_dataset.jsonl`（単一環境分）が使われる。

## 対象外にしているもの（意図的）

- **`person_profiles.json`（顔の埋め込みデータ）はマージしない。** 生体データを環境間で集約するのはプライバシー上のリスクが増えるだけで、STAIチューニングの目的には不要なため。
- **`stress_log.csv`（直近セッションのみのフレームログ）もマージしない。** 実行のたびに上書きされる一時ファイルで、そもそも累積データではないため。

## セキュリティ上の注意

- 共有フォルダの権限確認は定期的に行う（メンバーの入れ替わりなど）。
- `merge_data.py` はローカルファイルの読み書きのみを行い、外部通信は一切しない。
- `merged/` フォルダとその中身は `.gitignore` 済み。gitリポジトリにコミットしないこと。
