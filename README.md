# 定点観測ストレス評価システム (リアルタイム表情分析＆STAI連動機能付き)

本システムは、カメラ映像からリアルタイムで人物の表情（感情）および顔特徴（まばたき、頭部運動、口・瞼の緊張など）を抽出し、その人の「普段」からの逸脱（zスコア）を用いてストレス評価値（0〜100）を算出するシステムです。

用途に応じて次の3つの形で動かせます（詳細は後述）。

| 形態 | 起動 | 説明 |
| :--- | :--- | :--- |
| スタンドアロン | `python peroson_detect.py`（`output_mode=legacy`） | カメラ映像＋HUD表示。終了時にCSV/グラフ保存、STAI記録 |
| ROS2ノード | `output_mode=ros2` / `both` | 高ストレス継続をトピックで通知（Jazzy） |
| 評価サーバー | `python stress_server.py` | ロボットはJPEGをPOSTするだけ、評価はサーバー側 |

また、心理的不安指標 **STAI** の回答結果と連動させ、線形/リッジ回帰で各特徴の重みやゲイン・基準値を実データに合わせて自動チューニングできます。

---

## 💡 主な機能と特徴

### 1. リアルタイム表情・複数特徴分析＆ストレス値評価
*   **MediaPipe FaceLandmarker (Tasks API)**: 毎フレーム顔の478点ランドマークおよび顔表情（blendshapes）を高精度かつ軽量に検出。まばたき、眉間のしわ、瞼や口唇の緊張状態をリアルタイムで追跡します。
*   **HSEmotion (EfficientNet系オンプレミスモデル)**: 感情認識モデル（`enet_b2_8`）を統合し、表情から8クラス（怒り、軽蔑、嫌悪、恐怖、喜び、無表情、悲しみ、驚き）の確率分布を時間平滑化（EMA）を適用しながら安定検出します。
*   **マルチモーダル・ストレス算出**: 先行研究に基づき、以下の6つの成分のzスコアからストレス値をリアルタイム評価（0〜100点）します。
    *   **感情感情値 (emotion)**: ネガティブ感情の確率和
    *   **眉間のしわ (brow)**: 眉を寄せる強さ（眉間しわ AU4）
    *   **まばたき率 (blink)**: 直近30秒間の瞬目頻度（回/分）
    *   **頭部運動 (head)**: 頭部姿勢（回転行列）の角速度
    *   **口唇の緊張 (mouth)**: 唇を強く結ぶ強さ（AU23-24）
    *   **瞼の緊張 (eye)**: 目を細める/見開く強さ（瞼緊張 AU7）
*   **歩行・移動フィルタ**: 頭部の並進移動速度を監視し、歩行や大きな体の動きによるノイズを自動でゲート（抑制）することで、「全身の移動」を「頭単体のストレス運動」と誤評価するのを防ぎます。

### 2. 平常状態のパーソナライズ（ベースライン学習）
*   **`passive`（既定）**: 明示的な待機フェーズはありません。利用中に人物IDごと・特徴量ごとに直近900サンプルを蓄積し、**中央値とMAD（頑健統計）**でベースラインを作るため、一時的な緊張などの外れ値は自動的に除外されます。各特徴が60サンプル貯まるとストレス値の表示が始まります（それまでHUDに `Learning baseline for ID:n (k/60)` と進捗表示）。「普段の顔をしてください」と指示しないので、そのポーズ自体が基準を歪めません。
*   **`explicit`**: そのIDを初めて検知してから7秒間（`CALIB_SECONDS`）、無表情を保つ従来方式（HUDに `CALIBRATING... keep a neutral face`）。`config.py` の `BASELINE_MODE` で切り替えます。
*   ベースラインは人物ごとに `person_profiles.json` へ保存・累積され、表情の癖や姿勢による個人差を排除します。

### 3. 顔識別と人物IDトラッキング (Face ID Tracking)
*   **深層顔認識 (DeepFace SFace / InsightFace)**: 幾何ランドマーク情報だけでは区別が難しい「別人」の分離を高精度に行うため、深層学習を用いた顔埋め込み（Embedding）を利用します。
*   **フレームをまたぐ安定性 (Hysteresis)**: 一瞬の向きの違いや誤検知でIDが頻繁に切り替わらないよう、数フレーム連続で別人と判定されて初めてIDを切り替えるヒステリシス制御（switch patience）を導入。
*   **セッション間の固定ID**: 顔埋め込みと学習した平常統計は人物プロファイル（`person_profiles.json`）に永続化され、再起動後も同じ顔に同じIDを割り当てて再開できます。バックエンドは `FACE_ID_BACKEND`（`auto` = deepface → insightface → landmark）で選択します。

### 4. STAI得点の記録方法 (`collect` / `survey` モード)
*   **`collect` モード**: 別紙などで採点済みのSTAI得点を、セッション終了時にコンソールから直接入力します。既定ではSTAI-S（状態不安, 20〜80点）をそのまま入力しますが、`stress_config.json` の `"stai_s_form"` を `"short6"` にすると、短縮版 **STAI-S6**（6〜24点）を入力するだけで自動的に20〜80スケールへ正規化（×20/6）して記録できます。STAI-T（特性不安）は常に20〜80点で入力します。
*   **`survey` モード**: セッション終了時に、実施する尺度（[1] STAI-Sのみ20項目 / [2] S+T 40項目）を選び、コンソール上で `1`〜`4` で回答します。逆転項目は自動採点され、STAI-S/-T（20〜80点）を算出。人物ごとに、セッション平均zスコアと合わせて `stai_dataset.jsonl`（チューニング用）と `stai_dataset.csv`（閲覧用）に追記されます。最初の項目を空欄にするとその人物をスキップできます。
*   `run` モード（既定）では終了時に何も聞きません。

### 5. `tune_stress.py` による自動最適化 (Tuning)
*   蓄積されたSTAIスコアとセッション平均特徴量のデータセット（`stai_dataset.jsonl`）を読み込み、STAI-S得点に最も連動（高い正の相関）するよう、6つのストレス成分の重み・ゲイン・ベースレベルを自動で最適化します。
*   サンプル数が少ない場合は過適合を防ぐために正則化（リッジ回帰）やスケール調整のみのフォールバックを自動で行い、最適化された設定は `stress_config.json` に即座に適用されます。

---

## 📂 ファイル構成

### リポジトリ（コード）

| ファイル名 | 役割・説明 |
| :--- | :--- |
| **`peroson_detect.py`** | 本体。映像取得、特徴抽出、ストレス算出、HUD描画、ログ保存、STAI記録（`main()` はROS2/サーバーからも呼ばれる） |
| **`config.py`** | 起動時に固定の定数（モデル名・閾値・EMA係数・ベースライン方式・顔識別・ROS2/アラート・保存先パス）を集約 |
| **`tune_stress.py`** | オフラインチューニング（`--report` / `--apply` / `--no-trait`） |
| **`stai_items.json`** | STAI質問紙の項目文・逆転項目・回答アンカー。`MIRS_DATA_DIR` 側に置くとそちらを優先 |
| **`stress_alert.py`** | 高ストレス継続判定 `SustainedAlert`（ROS非依存。`python stress_alert.py` で自己チェック） |
| **`stress_alert_node.py`** | ROS2ノード（`output_mode` が `ros2`/`both` のとき `peroson_detect.py` から起動） |
| **`stress_server.py` / `stress_client.py`** | 外部評価サーバーとロボット側の送信例 |
| **`coral/`** | Coral Edge TPU 用モデル変換スクリプトとフォールバックのテスト |
| **`scripts/`** | `bootstrap.sh`（Raspberry Pi初期化）、`deploy.sh`（更新）、`prefetch_models.py`、`migrate_local_data.py`、`reqhash.sh` |
| **`docs/DEPLOY.md`** | 本番（Raspberry Pi）のデプロイ・運用手順 |
| **`requirements.txt`** | 依存パッケージ（宣言側。実機は `make freeze` で作る lock を使う） |
| **`Makefile`** / **`run.ps1`** | 開発・運用ショートカット（`make help` / `.\run.ps1 <task>`。`clean*` は `MIRS_DATA_DIR` を対象にする） |
| **`test_calibration.py` / `test_stai_s_form.py`** | 自己チェック（pytest不要、`python test_*.py`） |

### データ（リポジトリ外）

生成物とモデルキャッシュは、`git pull` / `git reset` で消えたり作業ツリーを汚したりしないよう**リポジトリの外**に出力されます。既定はリポジトリの一つ上の階層の **`MIRS_data/`**。場所は `make paths` で確認でき、環境変数で変更できます。

*   `MIRS_DATA_DIR`: ログ・プロファイル・データセットの保存先
*   `MIRS_MODEL_DIR`: 自動DLモデルのキャッシュ先（既定 `MIRS_DATA_DIR/models`）

| ファイル名（`MIRS_DATA_DIR` 内） | 内容 |
| :--- | :--- |
| `stress_config.json` | 動作モード・重み・ゲイン・基準・std下限・Coral設定等（初回起動で自動生成。`tune_stress.py` の書き戻し先） |
| `person_profiles.json` | 人物ごとの顔埋め込み・ベースライン統計 |
| `session_history.jsonl` / `.csv` | セッション要約履歴 |
| `stai_dataset.jsonl` / `.csv` | STAI得点＋セッション平均特徴量 |
| `stress_log.csv` / `stress_graph.png` | 直近セッションの時系列ログとグラフ（終了時に上書き） |
| `models/` | `face_landmarker.task` 等（SFaceの重みは `~/.deepface/weights`） |

---

## 🛠️ セットアップ手順

### システム要件
*   Python 3.9 〜 3.12 推奨（ROS2 Jazzy は Ubuntu 24.04 の Python 3.12）
*   ウェブカメラ（内蔵/USB）または動画ファイル
*   Windows / macOS / Linux（本番は Raspberry Pi）

### ライブラリのインストール
```bash
pip install -r requirements.txt
```
主な依存: `opencv-python`, `numpy`, `mediapipe`, `hsemotion-onnx`, `onnxruntime`, `matplotlib`, `deepface`（SFace顔識別。tensorflow を引き込むため重い）, `tf-keras`。

*   **より高精度な顔認識（任意）**: `pip install insightface`（Windowsはコンパイラが必要な場合あり）
*   **Raspberry Pi 本番環境**: `make bootstrap`（venv・データ領域・モデル取得）→ 更新は `make deploy`。詳細は [docs/DEPLOY.md](docs/DEPLOY.md)。

### モデルファイル
MediaPipe FaceLandmarker・SFace・HSEmotion の重みは**初回起動時に自動ダウンロード**されます。事前取得は `make prefetch`。

---

## 🚀 基本的な使い方

### 1. 起動
```bash
python peroson_detect.py      # または make run / .\run.ps1 run
```
カメラは `config.py` の `CAMERA_SOURCE`（整数=デバイス番号 / 文字列=動画ファイル・RTSP URL）。

### 2. ベースライン学習
`passive`（既定）では何もせず普通に過ごすだけです。HUDの `Learning baseline for ID:n (k/60)` が満了するとストレスゲージが表示されます。`explicit` に切り替えた場合は7秒間の無表情キープが必要です（前述）。

### 3. 分析中のUI
顔の枠・人物ID（`ID:n  sim:…`）・ストレスゲージ（0〜100、緑 <33 / 黄 / 赤 ≥66）と、各成分（表情・眉間・まばたき・頭部運動・口・瞼）の値が表示されます。頭部の並進移動が大きい（歩行中）フレームでは頭部運動の寄与を自動で抑制します（HUDの `loco` で確認し、`config.py` の `LOCO_GATE_LOW/HIGH` を調整）。

### 4. 終了と保存
映像ウィンドウで **`q`**、またはコンソールで **`Ctrl + C`**。`stress_log.csv`・`stress_graph.png`・人物プロファイル・セッション履歴が `MIRS_DATA_DIR` に保存され、`collect` / `survey` モードならSTAI記録に進みます。

---

## 📈 STAI連動によるパラメータ調整手順

### ステップ1: 動作モードの変更
`MIRS_DATA_DIR/stress_config.json`（初回起動で自動生成）の `"mode"` を変更します。

*   `"survey"`: 終了時にアプリ内でSTAIに回答（逆転項目を自動採点）
*   `"collect"`: 別紙で採点済みの得点を入力。`"stai_s_form": "short6"` で STAI-S6（6〜24点）を入力して20〜80へ正規化（×20/6）

### ステップ2: データの蓄積
1.  様々なストレス状況（PC作業・計算タスク・リラックス等）で数分ずつ測定し、`q` で終了します。
2.  プロンプトに従い回答/入力すると `stai_dataset.jsonl` に追記されます。
3.  相関の精度のため**最低 4〜5 セッション以上**を蓄積してください。

### ステップ3: チューニング
```bash
python tune_stress.py --report            # 現在の設定とSTAI-Sの相関(r)・RMSE/MAE を表示（設定は変更しない）
python tune_stress.py --apply             # 最適化して stress_config.json を更新（旧版は .bak に退避）
python tune_stress.py --apply --no-trait  # STAI-T を共変量に使わない
```
サンプル数に応じて OLS → リッジ → スケールのみ調整、と自動でフォールバックします。適用後は `"mode"` を `"run"` に戻します。

---

## ⚡ Coral USB Accelerator (Edge TPU) 対応（任意）

感情分類（HSEmotion）を USB接続の Coral Edge TPU にオフロードできます。Coral未接続/未設定の環境では
自動的に従来どおりCPU(onnxruntime)で動作するため、無効化していれば挙動は一切変わりません。

### 有効化手順
1. **Coral Edge TPU runtime（USBドライバ）** を導入する（[coral.ai/software](https://coral.ai/software/) の
   "Edge TPU runtime" をOS別にインストール）。
2. **推論用ライブラリ**: `pip install tensorflow`（`tensorflow.lite.Interpreter` を使用。Python 3.9以下の
   環境なら軽量な `tflite-runtime`/`pycoral` でも可）。
3. **Edge TPU用モデルの変換**（Linux/WSL上で一度だけ）:
   ```bash
   cd coral
   pip install onnx onnx2tf onnxruntime tensorflow numpy opencv-python-headless
   # Coral公式リポジトリから edgetpu_compiler を導入（Linux x86-64専用）
   python convert_emotion_model.py --model-name enet_b2_8 --calib-dir ./calib_faces
   ```
   `--calib-dir` には顔クロップ画像を20〜100枚程度置くことを推奨します（無いとランダム画像で量子化され精度が落ちます）。
   生成された `coral/models/enet_b2_8_edgetpu.tflite` を、`MIRS_DATA_DIR` 配下の `models/coral/emotion_enet_b2_8_edgetpu.tflite`（既定の `model_path`）へコピー/リネームします。
4. `stress_config.json` を編集して有効化します。
   ```json
   {
     "coral": {
       "enabled": true,
       "model_path": "models/coral/emotion_enet_b2_8_edgetpu.tflite",
       "device": ""
     }
   }
   ```
   `model_path` は `MIRS_DATA_DIR` からの相対パス（絶対パスも可）。`device` は Coral を複数接続する場合のみ `":0"` 等を指定します（通常は空文字でOK）。

### 注意点
* `edgetpu_compiler` は Linux(x86-64) 専用です。変換はWSL/Docker/Linux上で行い、本体(`peroson_detect.py`)は
  別環境上でそのまま変換済み `.tflite` を読み込むだけです。
* Coral公式には既製の「感情分類」Edge TPUモデルが存在しないため、上記スクリプトで自前変換する必要があります。
  int8量子化により、CPU(float32)版に比べて精度がわずかに劣化する場合があります。導入後は `stress_log.csv` を
  CPU版と見比べることを推奨します。
* Coral未接続・モデル未配置・ライブラリ未導入など、いかなる理由でも起動時に自動でCPUにフォールバックし、
  コンソールにその旨が表示されます（エラー終了はしません）。

---

## 🧹 データのリセット方法
対象は `MIRS_DATA_DIR`（`make paths` で確認）です。

*   `make clean`: `stress_log.csv` / `stress_graph.png` / `__pycache__` を削除
*   `make clean-profiles`: `person_profiles.json` / `session_history.jsonl` を削除（人物IDとベースラインのリセット）
*   `make clean-all`: 上記すべて
*   `make show-profiles`: 保存済みプロファイルを整形表示

`stress_config.json` と `stai_dataset.*` は消えません（不要なら手動で削除）。

---

## ⚠️ 免責事項 (Disclaimer)

*   **目的制限**: 本ソフトウェアは、リアルタイム表情分析およびストレス測定の実験・研究、またはホビー用途を目的として開発されたものです。
*   **非医療機器**: 本ソフトウェアは**医療機器ではなく、専門的な精神医学的・心理学的診断、治療、または予防を目的としたものではありません**。ストレス値や不安度の算出結果は学術的アプローチによる統計的推定であり、医学的な妥当性や診断を保証するものではありません。
*   **免責**: 本ソフトウェアの利用、機能、診断結果、または本ソフトウェアに含まれる情報によって、直接的・間接的を問わず発生したいかなる損害（肉体的・精神的不調、業務の中断、利益の損失、データの損失、その他付随的・派生的なすべての損害）について、**開発者および提供者は一切の責任および義務を負いません**。本ソフトウェアの使用は、すべてユーザー自身の自己責任において行ってください。

---

## 🤖 ROS2 ノードとして使う（Jazzy）

`config.py` の `OUTPUT_MODE`（または `stress_config.json` の `"output_mode"`）で出力を切り替えます。起動は常に `python peroson_detect.py`。

| OUTPUT_MODE | 映像表示 | ROS2 publish |
|---|---|---|
| `legacy`（既定） | あり | なし |
| `ros2` | なし（ヘッドレス） | あり |
| `both` | あり | あり |

高ストレス（`ALERT_THRESHOLD` 以上）が `ALERT_HOLD_SEC` 秒以上続くと `/stress/alert` が `True` になります。判定は `ALERT_MODE`（`any`=誰か1人でも / `mean`=その場全体の平均）。判定ロジックは ROS 非依存の [stress_alert.py](stress_alert.py)（`python stress_alert.py` で自己チェック）。

| トピック | 型 | 内容 |
|---|---|---|
| `/stress/alert` | `std_msgs/Bool`（transient_local） | 高ストレス継続中か |
| `/stress/alert_ids` | `std_msgs/Int32MultiArray` | アラート中の人物ID |
| `/stress/max`, `/stress/mean` | `std_msgs/Float32` | 在席者の最大/平均ストレス |

ノードパラメータ: `alert_threshold`, `alert_hold_sec`, `alert_stale_sec`, `alert_mode`（`--ros-args -p alert_mode:=mean` で上書き）。

```bash
# WSL / Ubuntu 24.04 (Jazzy)
python3 -m venv --system-site-packages .venv && source .venv/bin/activate   # rclpy を見せる
pip install -r requirements.txt
source /opt/ros/jazzy/setup.bash
make ros2                           # = python peroson_detect.py --output-mode ros2（映像も出すなら make ros2 ROS_MODE=both）
ros2 topic echo /stress/alert
```

`--output-mode legacy|ros2|both` は config / `stress_config.json` の `output_mode` を一時的に上書きします。`.\run.ps1 ros2` でも同じです。

WSL2 は既定でUSBカメラを認識しないため、`config.py` の `CAMERA_SOURCE` に動画ファイルのパスを指定して試すか、`usbipd` でカメラをattachしてください。

---

## 🌐 サーバー運用（ストレス評価を外部サーバーで行う）

ロボットは映像を送るだけにして、評価（本ソフト）をサーバー側で動かせます。[stress_server.py](stress_server.py) が JPEG を HTTP POST で受け取り、`peroson_detect.main()` に流してストレス値を JSON で返します。

```
POST /   body=JPEG   ヘッダ: X-Timestamp=送信時刻(epoch秒) / X-Token=共有トークン(任意)
→ {"score": 42.3, "face_id": 1, "timestamp": <送った時刻>}
```

| ステータス | 意味 |
|---|---|
| 200 | 評価済み。`score` は顔なし・ベースライン学習中は `null` |
| 400 / 401 | 画像が不正 / トークン不一致 |
| 503 | モデル読み込み中 |
| 504 | 10秒以内に処理が終わらなかった |

```bash
# サーバー（make serve / .\run.ps1 serve でも可）
STRESS_TOKEN=secret PORT=8000 python stress_server.py
# ロボット側（動作確認。Webカメラ映像を送って score を表示）
STRESS_TOKEN=secret python stress_client.py http://<サーバーIP>:8000/
```

ロボットの ROS2 ノードからは [stress_client.py](stress_client.py) の `evaluate(frame, url, ts)` を画像コールバックで呼びます。`ts` は撮影時刻（`msg.header.stamp` を秒に直した値）で、まばたき率や頭部速度の計算に使われます。返ってきた score を `SustainedAlert` に渡せば、アラート判定はロボット側に残せます。

### 実ロボットなしで結果をHTMLで確認する
サーバーを起動してブラウザで `http://localhost:8000/` を開くと、テストページが出ます。「開始」を押すと**このPCのWebカメラ映像を疑似ロボットとして送信**し、ストレス値・人物ID・アラート状態（`ALERT_THRESHOLD` 以上が `ALERT_HOLD_SEC` 秒継続で ON。ROS2の `/stress/alert` 相当）と直近300サンプルの推移グラフをリアルタイム表示します。JSON応答にも `alert` が含まれます。

* ブラウザのカメラは `localhost` か https でしか使えません。別PCから見る場合は、サーバー側PCで開くか SSH ポートフォワードしてください。
* `STRESS_TOKEN` を設定している場合は、ページのトークン欄に同じ値を入れます。

### 注意点
* **ストリームは1本のみ**: ベースライン・平滑化・まばたき計測の状態が1つなので、複数台のロボットで共有するとスコアが混ざります。10fps 前後で安定して送ってください。
* **時計**: ロボット側の時刻を使うため、NTP 等で時計を安定させてください。サーバーとの時計のずれは問題ありません（最初のフレームで基準を合わせます）。
* **Tailscale 越し**: 別Wi-Fi間でも `http://<Tailscale IP>:8000/` で届きます。通信は WireGuard で暗号化されますが、同じ tailnet の全員が届くため `STRESS_TOKEN` は設定してください。届かない場合は Windows ファイアウォール（8000番）と Tailscale の ACL を確認します。
* **遅延**: 1枚ずつ結果を待つ方式のため、往復遅延がそのままフレーム間隔になります（往復100msで約10fps）。
