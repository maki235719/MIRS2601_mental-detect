#!/usr/bin/env bash
# requirements 系ファイルの内容ハッシュを出す。
# deploy.sh がこれを比べて「変わっていなければ pip を走らせない」ために使う。
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cat "$REPO_DIR/requirements.txt" \
    "$REPO_DIR/requirements.lock.txt" 2>/dev/null \
  | sha256sum | awk '{print $1}'
