# make / run.ps1 コマンド対応表

Windows など `make` が無い環境では `run.ps1` を使う（`.\run.ps1 <target>`、省略時は `run`）。

## ターゲット対応表

| 機能 | `make` | `run.ps1` | 実行内容 |
| :-- | :-- | :-- | :-- |
| 通常起動 | `make run` | `.\run.ps1 run` | `python peroson_detect.py` |
| 評価サーバー起動 | `make serve` | `.\run.ps1 serve` | `python stress_server.py`（http://localhost:8000/ にテストページ） |
| ROS2ノード起動 | `make ros2` | `.\run.ps1 ros2` | `peroson_detect.py --output-mode $ROS_MODE`（要 `source /opt/ros/jazzy/setup.bash`） |
| 出力先の表示 | `make paths` | `.\run.ps1 paths` | `DATA_DIR` / `MODEL_DIR` を表示（make はさらに `REPO_DIR` も表示） |
| セッション出力の削除 | `make clean` | `.\run.ps1 clean` | CSV・グラフ・`__pycache__` を削除 |
| 学習データの削除 | `make clean-profiles` | `.\run.ps1 clean-profiles` | `person_profiles.json` / `session_history.jsonl` を削除 |
| 全削除 | `make clean-all` | `.\run.ps1 clean-all` | `clean` + `clean-profiles` |
| プロファイル表示 | `make show-profiles` | `.\run.ps1 show-profiles` | 保存済みプロファイルを整形表示 |
| ターゲット一覧 | `make help` | ―（未知のターゲット指定時に一覧表示） | |
| 初回セットアップ（本番） | `make bootstrap` | ― | `bash scripts/bootstrap.sh` |
| バージョン更新（本番） | `make deploy` | ― | `bash scripts/deploy.sh` |
| 依存バージョン固定（本番） | `make freeze` | ― | `pip freeze` → `requirements.lock.txt` |
| 旧データの移行（本番） | `make migrate` | ― | `scripts/migrate_local_data.py` |
| モデル事前取得（本番） | `make prefetch` | ― | `scripts/prefetch_models.py` |

「―」は未対応。本番（Raspberry Pi）用の5つは `make` のみ。

## オプション（変数）の渡し方

| 項目 | `make` | `run.ps1` | 既定値 |
| :-- | :-- | :-- | :-- |
| Python 実行ファイル | `make run PYTHON=python3` | `$env:PYTHON="python3"; .\run.ps1 run` | `python` |
| ROS2 出力モード（映像表示ありは `both`） | `make ros2 ROS_MODE=both` | `$env:ROS_MODE="both"; .\run.ps1 ros2` | `ros2` |
| サーバーのポート | 環境変数 `PORT` | 環境変数 `PORT` | `8000` |
| サーバーのトークン | 環境変数 `STRESS_TOKEN` | 環境変数 `STRESS_TOKEN` | ― |
| データ出力先 | 環境変数 `MIRS_DATA_DIR` | 環境変数 `MIRS_DATA_DIR` | リポジトリ外のユーザーデータ領域 |

## 違い・注意

- 変数の渡し方: make は引数（`VAR=値`）、run.ps1 は環境変数。
- `clean` 系は run.ps1 だけ完了メッセージを表示する。make は `-` 付きなので削除に失敗しても止まらない。run.ps1 は `$ErrorActionPreference = "Stop"` で止まる。
- 未知のターゲットは run.ps1 が `unknown task` と一覧を出して終了コード 1 で終わる。
