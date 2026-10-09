#!/usr/bin/env bash
# Raspberry Pi でのバージョン更新（毎回これだけ実行する）
#
#   bash scripts/deploy.sh              # 現在のブランチを更新
#   bash scripts/deploy.sh module       # ブランチを指定して更新
#
# 特徴:
#   * 本番のリポジトリは「GitHub の完全なコピー」に揃える（reset --hard）。
#     データとvenvはリポジトリ外にあるので消えない → 安全に強制上書きできる。
#   * requirements が変わっていない限り pip を実行しない
#     → バージョン更新ごとの全パッケージ再ダウンロードが無くなる。
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
VENV_DIR="${MIRS_VENV_DIR:-$HOME/.venvs/mirs}"
DATA_DIR="${MIRS_DATA_DIR:-$HOME/.local/share/mirs-stress}"
STAMP="$DATA_DIR/.stamp/requirements.sha"

echo "== deploy: branch=$BRANCH =="

# 1) 本番で誰かが直接いじっていた場合の記録を残す（消す前に保険）
if ! git diff --quiet || ! git diff --cached --quiet; then
  PATCH="$DATA_DIR/local-changes-$(date +%Y%m%d-%H%M%S).patch"
  mkdir -p "$DATA_DIR"
  git diff HEAD > "$PATCH"
  echo "!! 本番側のローカル変更を $PATCH に退避した（そのうえで破棄する）"
fi

# 2) GitHub の状態に完全一致させる
git fetch origin --prune
git checkout -B "$BRANCH" "origin/$BRANCH"
git reset --hard "origin/$BRANCH"
# 追跡外ファイルも掃除。ただし .gitignore 済みのものは残す(-x を付けない)。
git clean -fd

echo "-- now at: $(git log -1 --oneline)"

# 3) 依存パッケージ: ハッシュが変わった時だけインストール
NEW_HASH="$(bash "$REPO_DIR/scripts/reqhash.sh")"
OLD_HASH="$(cat "$STAMP" 2>/dev/null || echo none)"

if [ ! -d "$VENV_DIR" ]; then
  echo "!! venv が無い。先に scripts/bootstrap.sh を実行してください"
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [ "$NEW_HASH" = "$OLD_HASH" ]; then
  echo "-- requirements 変更なし → pip をスキップ"
else
  echo "-- requirements 変更あり → インストール実行"
  if [ -f requirements.lock.txt ]; then
    pip install -r requirements.lock.txt
  else
    pip install -r requirements.txt
  fi
  mkdir -p "$(dirname "$STAMP")"
  echo "$NEW_HASH" > "$STAMP"
fi

echo
echo "== 更新完了 =="
echo "実行: source $VENV_DIR/bin/activate && python $REPO_DIR/peroson_detect.py"
