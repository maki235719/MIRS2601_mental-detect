# 定点観測ストレス評価スクリプトの開発用ショートカット
# 使い方: make run / make clean / make clean-profiles / make clean-all / make show-profiles
#         make bootstrap / make deploy / make freeze / make paths   (本番=Raspberry Pi 用)
# ※ Windows で make が無い場合は run.ps1 を使う: .\run.ps1 <target>
#
# 出力ファイルは MIRS_DATA_DIR（既定はリポジトリ外のユーザーデータ領域）に出る。
# 場所を確認したいときは make paths。

PYTHON ?= python

.PHONY: run clean clean-profiles clean-all show-profiles help \
        bootstrap deploy freeze paths migrate prefetch

help:
	@echo "dev  : run | clean | clean-profiles | clean-all | show-profiles | paths"
	@echo "prod : bootstrap | deploy | freeze | migrate | prefetch"

run:
	$(PYTHON) peroson_detect.py

# 出力先・モデル置き場を表示（リポジトリ外にあることの確認用）
paths:
	$(PYTHON) -c "import config;print('DATA_DIR :',config.DATA_DIR);print('MODEL_DIR:',config.MODEL_DIR);print('REPO_DIR :',config.REPO_DIR)"

# セッション出力（ログ・グラフ・キャッシュ）だけ消す
# 削除は rm 非依存にするため python 経由（Windowsのmake環境でも動くように）
clean:
	-$(PYTHON) -c "import os,shutil,config;[os.remove(p) for p in (config.STRESS_CSV_PATH,config.STRESS_GRAPH_PATH) if os.path.exists(p)];shutil.rmtree(os.path.join(config.REPO_DIR,'__pycache__'),ignore_errors=True)"

# 学習した平常状態・人物ID（定点観測データ）を消す＝開発中のリセット
clean-profiles:
	-$(PYTHON) -c "import os,config;[os.remove(p) for p in (config.PROFILES_JSON_PATH,config.SESSION_HISTORY_PATH) if os.path.exists(p)]"

# すべて消す
clean-all: clean clean-profiles

# 保存済みプロファイルを整形表示
show-profiles:
	$(PYTHON) -c "import json,config;print(json.dumps(json.load(open(config.PROFILES_JSON_PATH,encoding='utf-8')),indent=2,ensure_ascii=False))"

# ---- 本番(Raspberry Pi)用 ----------------------------------------------------

# 初回セットアップ（venv・データ領域をリポジトリ外に作り、モデルを先に取得）
bootstrap:
	bash scripts/bootstrap.sh

# バージョン更新（GitHub に完全一致させる。requirements に変更が無ければ pip はスキップ）
deploy:
	bash scripts/deploy.sh

# 現在の環境の厳密なバージョンを lock ファイルへ固める（これを commit する）
freeze:
	$(PYTHON) -m pip freeze --exclude-editable > requirements.lock.txt
	@echo "requirements.lock.txt を更新した。git add して commit してください。"

# 旧配置（リポジトリ内）に残った生成物・モデルを DATA_DIR へ引っ越す
migrate:
	$(PYTHON) scripts/migrate_local_data.py

# モデルを事前ダウンロード（既にあれば何もしない）
prefetch:
	$(PYTHON) scripts/prefetch_models.py
