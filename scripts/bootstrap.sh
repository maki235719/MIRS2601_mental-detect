#!/usr/bin/env bash
# Raspberry Pi 初回セットアップ（1回だけ実行する）
#
#   bash scripts/bootstrap.sh
#
# ポイント: 仮想環境とデータをリポジトリの外に作る。
# こうしておけば以降 git を強制上書きしても、パッケージとモデルは消えない
# ＝バージョン更新ごとの再ダウンロードが無くなる。
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${MIRS_VENV_DIR:-$HOME/.venvs/mirs}"
DATA_DIR="${MIRS_DATA_DIR:-$HOME/.local/share/mirs-stress}"
MODEL_DIR="${MIRS_MODEL_DIR:-$DATA_DIR/models}"

echo "== MIRS stress-eval bootstrap =="
echo "  repo : $REPO_DIR"
echo "  venv : $VENV_DIR"
echo "  data : $DATA_DIR"
echo "  model: $MODEL_DIR"

# --- OS パッケージ（カメラ表示・ビルドに必要なもの）---
if command -v apt-get >/dev/null 2>&1; then
  echo "-- apt packages"
  sudo apt-get update
  sudo apt-get install -y \
    python3-venv python3-dev python3-pip \
    libgl1 libglib2.0-0 libatlas-base-dev \
    libopenblas-dev ffmpeg
fi

mkdir -p "$DATA_DIR" "$MODEL_DIR" "$(dirname "$VENV_DIR")"

# --- venv（リポジトリ外）---
if [ ! -d "$VENV_DIR" ]; then
  echo "-- create venv"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# --- 依存パッケージ ---
# lock ファイルがあればそれを最優先（環境の再現性が上がる）
if [ -f "$REPO_DIR/requirements.lock.txt" ]; then
  echo "-- install from requirements.lock.txt"
  pip install -r "$REPO_DIR/requirements.lock.txt"
else
  echo "-- install from requirements.txt (lock 未作成)"
  pip install -r "$REPO_DIR/requirements.txt"
fi

# 次回 deploy 時の差分判定用ハッシュを保存
mkdir -p "$DATA_DIR/.stamp"
"$REPO_DIR/scripts/reqhash.sh" > "$DATA_DIR/.stamp/requirements.sha" || true

# --- 旧配置（リポジトリ内）に残っているデータを DATA_DIR へ引っ越す ---
python "$REPO_DIR/scripts/migrate_local_data.py" || true

# --- モデルを先に取得しておく（初回起動を速くする）---
echo "-- prefetch models"
MIRS_DATA_DIR="$DATA_DIR" MIRS_MODEL_DIR="$MODEL_DIR" \
  python "$REPO_DIR/scripts/prefetch_models.py" || \
  echo "   (取得失敗。初回起動時に自動DLされるので致命的ではない)"

# --- 環境変数を永続化 ---
PROFILE_LINE="export MIRS_DATA_DIR=\"$DATA_DIR\"; export MIRS_MODEL_DIR=\"$MODEL_DIR\"; export MIRS_VENV_DIR=\"$VENV_DIR\""
if ! grep -qF "MIRS_DATA_DIR" "$HOME/.bashrc" 2>/dev/null; then
  echo "$PROFILE_LINE" >> "$HOME/.bashrc"
  echo "-- ~/.bashrc に環境変数を追記した"
fi

echo
echo "完了。実行するには:"
echo "  source $VENV_DIR/bin/activate && python $REPO_DIR/peroson_detect.py"
