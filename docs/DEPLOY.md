# デプロイ・運用手順（本番＝Raspberry Pi）

## なぜ毎回 `git pull` が失敗していたのか

原因は3つ重なっていました。パッケージのダウンロードは、そのうちの1つです。

### 原因1（最大）: 改行コードの不一致 — CRLF vs LF

`git diff` を取ると、変更したはずのない `Makefile` `README.md` `run.ps1`
`stai_items.json` `tune_stress.py` `peroson_detect.py` `.gitignore` が
**全行 modified** になっていました。中身は同一で、違うのは行末だけです。

- GitHub 上のファイル: `LF`
- Windows の作業ツリー: `CRLF`
- Raspberry Pi の作業ツリー: `LF`

`.gitattributes` が無いため Git が正規化せず、**開くだけ・触るだけでツリーが汚れる**
状態でした。ツリーが汚れていると `git pull` は
「Your local changes would be overwritten / 先に commit してください」で必ず止まります。
これが「毎回強制上書きしないと pull できない」の本体です。

→ 対策: `.gitattributes` で `* text=auto eol=lf` を宣言し、一度だけ正規化する。

### 原因2: 生成物とモデルがリポジトリ内に出力されていた

`config.py` が

```python
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))   # ← リポジトリ自身
```

だったため、`stress_log.csv` `person_profiles.json` `stress_config.json`、
さらに自動ダウンロードされる **`face_landmarker.task`（3.6MB）** が
すべてリポジトリの中に落ちていました。

`.gitignore` で除外はされていますが、

- リポジトリを作り直す・強制上書きすると**モデルも一緒に消えるので再ダウンロードになる**
- 一部は過去のコミットで追跡されていたため、ブランチを切り替えると
  「未追跡ファイルが上書きされる」衝突を起こす

→ 対策: 出力とモデルキャッシュを**リポジトリの外**（`MIRS_DATA_DIR` /
`MIRS_MODEL_DIR`）へ移す。リポジトリはコードだけの読み取り専用に近い状態になり、
`reset --hard` を安心して使えるようになります。

### 原因3: 依存パッケージが固定されていなかった

`README` は `pip install opencv-python mediapipe ...` を直に叩く手順で、
`requirements.txt` も lock も無い状態でした。実行のたびに pip が
その時点の最新版を取り直すため、GitHub のコードと実機の環境が食い違います。
また、`config.py` 自体が **未追跡（GitHub に存在しない）** なのに
`peroson_detect.py` が import しているため、実機では手で置く必要がありました。

→ 対策: `requirements.txt`＋`requirements.lock.txt` で固定し、
`config.py` をバージョン管理に入れる。venv はリポジトリ外に置き、
**requirements のハッシュが変わった時だけ** pip を実行する。

---

## 一度だけやる作業

### A. 改行コードを正規化する（開発PC側で1回）

```bash
cd face_detect
git config core.autocrlf false      # .gitattributes に任せる
git add --renormalize .
git status                          # 行末だけの差分が消えていることを確認
git commit -m "改行コードをLFに正規化し .gitattributes を追加"
```

### B. `config.py` をコミットする

未追跡のままだと実機に配布できません。

```bash
git add config.py requirements.txt .gitattributes .gitignore Makefile scripts docs
git commit -m "デプロイ整備: データ配置をリポジトリ外へ / 依存固定 / デプロイスクリプト"
git push -u origin fix/deploy-clean-tree
```

> 秘密情報や現場ごとの値を入れたくなったら `config.local.py`（gitignore 済み）に
> 分け、`config.py` の末尾で `try: from config_local import *` する形にしてください。

### C. Raspberry Pi 側の初期化

```bash
git clone https://github.com/maki235719/MIRS2601_mental-detect.git
cd MIRS2601_mental-detect
bash scripts/bootstrap.sh
```

`bootstrap.sh` がやること:

| 対象 | 置き場所 | 意味 |
| :--- | :--- | :--- |
| venv | `~/.venvs/mirs` | リポジトリ外。git を消しても壊れない |
| ログ・プロファイル | `~/.local/share/mirs-stress` | 実機の測定データが git で消えない |
| モデル (`.task`, `.onnx`) | `~/.local/share/mirs-stress/models`, `~/.deepface/weights` | **一度落ちたら再DLされない** |

### D. 実機に既にデータがある場合の引っ越し

```bash
python scripts/migrate_local_data.py --dry-run   # 確認
python scripts/migrate_local_data.py             # 実行
```

---

## 毎回のバージョン更新（これだけ）

```bash
cd MIRS2601_mental-detect
bash scripts/deploy.sh          # または: make deploy
```

内部の動作:

1. 実機側にローカル変更があれば `~/.local/share/mirs-stress/local-changes-*.patch`
   に退避してから破棄（保険）
2. `git fetch` → `git reset --hard origin/<branch>` → `git clean -fd`
   で GitHub と完全一致させる
   （データ・venv・モデルはリポジトリ外なので**消えません**）
3. `requirements.txt` / `requirements.lock.txt` の SHA-256 を前回と比較し、
   **変わっていなければ pip をスキップ** → 再ダウンロードなし

つまり「コードだけ変わった更新」は数秒で終わり、パッケージのダウンロードは
依存を実際に変更した時だけ発生します。

## 依存を変更したとき

```bash
# 開発PC or 実機で
pip install <新しいパッケージ>
make freeze                     # requirements.lock.txt を更新
git add requirements.txt requirements.lock.txt
git commit -m "依存を更新"
git push
```

実機で `make deploy` するとハッシュ差分を検出して1回だけインストールします。

> lock ファイルは環境ごとに中身が変わります（Windows と aarch64 で
> ホイールが違う）。**実機側で `make freeze` したものを本番用の正**とし、
> 開発PCでは `requirements.txt` の範囲指定で運用するのが安全です。
> 厳密にやるなら `requirements.lock.linux-aarch64.txt` のように分けてください。

---

## 常駐させる場合（任意）

`/etc/systemd/system/mirs-stress.service`

```ini
[Unit]
Description=MIRS stress evaluation
After=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/MIRS2601_mental-detect
Environment=MIRS_DATA_DIR=/home/pi/.local/share/mirs-stress
Environment=MIRS_MODEL_DIR=/home/pi/.local/share/mirs-stress/models
ExecStart=/home/pi/.venvs/mirs/bin/python peroson_detect.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now mirs-stress
```

---

## 残っている宿題（今回は未着手）

1. **入れ子リポジトリ** — `MIRS/`（`MIRS.git`、コミット0件）の中に
   `MIRS/face_detect/`（`MIRS2601_mental-detect.git`）が入っていて、
   外側が `face_detect` を gitlink として認識しているのに `.gitmodules` が
   ありません。外側で作業すると常に dirty になります。
   `git submodule add` で正式なサブモジュールにするか、`git subtree` で
   統合するか、外側の追跡をやめるかを決めてください。
2. **`coral/`** — `__pycache__` に `convert_emotion_model` と
   `test_coral_fallback` の `.pyc` だけが残り、`.py` 本体が
   コミットも作業ツリーにも存在しません。消えた作業の可能性があります。
3. **`deepface` の重さ** — tensorflow を引き込むため Pi では 500MB 級です。
   SFace の `.onnx` は既にあるので、OpenCV の `cv2.FaceRecognizerSF` に
   置き換えれば tensorflow を丸ごと外せます（起動も速くなります）。
