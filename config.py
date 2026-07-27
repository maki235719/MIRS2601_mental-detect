"""
peroson_detect.py の設定値（定数）を集約したモジュール。

起動時に変わらないパラメータ（モデル名・閾値・EMA係数・ファイルパス等）を
ここに置く。実行のたびにチューニングされる値（重み・base/gain・std_floor の
一部）は stress_config.json 側にあり、peroson_detect.py の
load_stress_config() が読み込んで上書きする（tune_stress.py が書き戻す先）。
"""

import os

# ============================================================================
# 設定
# ============================================================================

# HSEmotion が返す感情ラベル（8クラス） → 日本語表示（描画はフォント都合で英語）
EMOTION_JP = {
    "Anger":     "怒り",
    "Contempt":  "軽蔑",
    "Disgust":   "嫌悪",
    "Fear":      "恐怖",
    "Happiness": "喜び",
    "Neutral":   "無表情",
    "Sadness":   "悲しみ",
    "Surprise":  "驚き",
}

# ストレス方向に働くネガティブ感情（この確率和が高いほどストレス寄り）
NEGATIVE_EMOTIONS = ["Anger", "Contempt", "Disgust", "Fear", "Sadness"]

# 何フレームごとに感情解析(HSEmotion)を実行するか。
# 顔ランドマーク(FaceLandmarker)は毎フレーム実行する（まばたき検出に必要かつ軽量）。
ANALYZE_EVERY = 3

# 使用する感情モデル（enet_b2_8 = 8クラス, 入力260px, b0より高精度）
HSEMOTION_MODEL = "enet_b2_8"

# 感情確率ベクトルの時間平滑化係数（0-1, 大きいほど反応が速い）
EMO_EMA_ALPHA = 0.4
# 最終ストレススコアの表示平滑化係数
STRESS_EMA_ALPHA = 0.3

# --- ストレススコア合成の設定 ---
# 各成分をベースラインからのzスコアにし、重み付き和を取る（合計が1.0になるよう配分）
# 論文（Giannakakis 2017 / 顔AUストレス解析2021 ほか）で報告された相関の強い
# 顔特徴を追加: head=頭部の動き, mouth=口唇の緊張, eye=瞼の緊張。重みは実測に応じ調整可。
STRESS_COMPONENT_WEIGHTS = {
    "emotion": 0.40,  # ネガティブ感情の増加
    "brow":    0.20,  # 眉間のしわ（browDown / AU4）の増加
    "blink":   0.10,  # まばたき率の増加
    "head":    0.12,  # 頭部運動（角速度）の増加
    "mouth":   0.10,  # 口唇の緊張（mouthPress / AU23-24）の増加
    "eye":     0.08,  # 瞼の緊張（eyeSquint / AU7）の増加
}
# z=0（＝平常時）を何点にするか、および z 1あたり何点上げるか
STRESS_BASE_LEVEL = 25.0
STRESS_Z_GAIN = 15.0

# zスコアの分母（標準偏差）の下限。平常状態が静かすぎるとノイズで暴れるのを防ぐ
EMO_STD_FLOOR = 0.05
BROW_STD_FLOOR = 0.02
BLINK_STD_FLOOR = 3.0    # 回/分
HEAD_STD_FLOOR = 2.0     # deg/秒（平滑化角速度）
MOUTH_STD_FLOOR = 0.02
EYE_STD_FLOOR = 0.02

# 頭部運動（角速度）の時間平滑化係数（0-1, 大きいほど反応が速い）
HEAD_EMA_ALPHA = 0.4

# --- 移動（並進）と頭単体の動きの分離 ---
# 歩行など「全身の移動」は頭部姿勢行列の並進成分(位置)に強く出る。一方、頭を振る等の
# 「頭単体の動き」は回転成分(向き)に出て並進は小さい。並進速度が大きいフレームは
# 「移動中」とみなし、頭部運動(回転)のストレス寄与をゲート(抑制)して、移動を頭の動き＝
# ストレスと誤評価しないようにする。単位は行列の並進(MediaPipe: おおよそcm)/秒。
# 実際の値は画面HUDの loco: 表示で確認できるので、着席時/歩行時を見て閾値を調整する。
LOCO_EMA_ALPHA = 0.4         # 並進速度の時間平滑化係数（0-1, 大きいほど反応が速い）
LOCO_GATE_LOW = 8.0          # これ以下は「静止」→頭部運動をフル採用（gate=1.0）
LOCO_GATE_HIGH = 25.0        # これ以上は「移動中」→頭部運動を無効化（gate=0.0）

# 較正（ベースライン測定）フェーズの長さ（秒）
CALIB_SECONDS = 7.0

# まばたき検出（blendshape eyeBlink のしきい値・ヒステリシス）と集計窓
BLINK_ON_THRESHOLD = 0.5
BLINK_OFF_THRESHOLD = 0.35
BLINK_WINDOW_SEC = 30.0  # まばたき率を計算する直近の窓（秒）

# 顔クロップの余白（ランドマーク外接矩形に対する比率）
FACE_CROP_MARGIN = 0.15

# ストレスバーの色分けしきい値（0-100）
STRESS_LOW_THRESHOLD = 33   # これ未満は緑（低ストレス）
STRESS_HIGH_THRESHOLD = 66  # これ以上は赤（高ストレス）、間は黄

# --- 顔識別（同一人物判定・タグ付け）の設定 ---
# 顔の埋め込みベクトル(embedding)を計算し、コサイン類似度で「同じ顔か」を判定する。
# フレームごとの識別に連続性を持たせるため、別人と判定し続けて初めてタグを切り替える
# （ヒステリシス）。insightface が無い場合は自動的に無効化され、本体は従来どおり動く。
FACE_ID_ENABLED = True          # 顔識別機能を使うか
# 使用バックエンド: "auto"|"deepface"|"insightface"|"landmark"
# auto の解決順は deepface → insightface → landmark。
# ※ landmark（幾何）方式は別人でも類似度が0.99に張り付き分離できないため最終手段。
#   深層埋め込み(deepface/insightface)が別人分離には桁違いに強い。
FACE_ID_BACKEND = "auto"
FACE_ID_EVERY = ANALYZE_EVERY   # 何フレームごとに顔認識を実行するか（重い場合は5〜10に上げる）
FACE_ID_ALIGN = True            # 認識前にクロップを目の傾きで水平化する（精度向上）
FACE_ID_SWITCH_PATIENCE = 8     # 別人と判定し続けてからタグを切り替えるフレーム数（連続性）
FACE_ID_EMBED_EMA = 0.1         # 登録済み埋め込みを毎回どれだけ更新するか（0-1）

# --- deepface（既定・推奨）---
FACE_ID_MODEL_DEEPFACE = "SFace"   # 軽量・高速でCPUリアルタイム向き
FACE_ID_SIM_THRESHOLD_DEEPFACE = 0.40  # 類似度がこれ以上なら同一人物（画面のsimを見て調整）

# --- insightface（任意・高精度。Python3.13のWindowsでは導入が不安定）---
FACE_ID_MODEL = "buffalo_l"     # 認識モデル: buffalo_s=軽量, buffalo_l=高精度
FACE_ID_SIM_THRESHOLD = 0.35    # 類似度がこれ以上なら同一人物（0-1）
FACE_ID_DET_SIZE = 320          # insightface 内部検出器の入力サイズ（小さいほど高速）

# --- landmark（最終手段。幾何形状ベクトル。別人分離は弱い）---
FACE_ID_SIM_THRESHOLD_LMK = 0.99  # 類似度がこれ以上なら同一人物（0-1）

# ============================================================================
# ファイル配置（コード＝リポジトリ / データ＝リポジトリ外）
# ============================================================================
# 【重要】生成物（ログ・プロファイル・自動DLしたモデル）をリポジトリ内に置くと、
# 実行するたびに作業ツリーが汚れて `git pull` が「先にcommitしてください」で
# 止まる。そのため出力先とモデルキャッシュはリポジトリの外に置く。
#
# 環境変数で上書き可能:
#   MIRS_DATA_DIR  : ログ・人物プロファイル・STAIデータセットの保存先
#   MIRS_MODEL_DIR : 自動ダウンロードするモデル(.task)のキャッシュ先
#                    （既定は MIRS_DATA_DIR/models。複数台で共有しても良い）

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def _default_base_dir():
    """OS ごとの標準的なユーザーデータ置き場を返す。"""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local"
        )
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share"
        )
    return os.path.join(base, "mirs-stress")


def _resolve_dir(env_name, default_path):
    path = os.environ.get(env_name) or default_path
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    return path


DATA_DIR = _resolve_dir("MIRS_DATA_DIR", _default_base_dir())
MODEL_DIR = _resolve_dir("MIRS_MODEL_DIR", os.path.join(DATA_DIR, "models"))

# 後方互換のエイリアス（旧コードが OUTPUT_DIR を参照している場合のため）
OUTPUT_DIR = DATA_DIR

STRESS_CSV_PATH = os.path.join(OUTPUT_DIR, "stress_log.csv")
STRESS_GRAPH_PATH = os.path.join(OUTPUT_DIR, "stress_graph.png")
# チューニング可能なパラメータ（重み・ゲイン・基準・分散下限）の外部設定ファイル。
# tune_stress.py が STAI 連動になるよう最適化して書き戻す先でもある。
STRESS_CONFIG_PATH = os.path.join(OUTPUT_DIR, "stress_config.json")
# STAI（状態不安STAI-S / 特性不安STAI-T）ラベルとセッション平均特徴の対応データセット。
# collect モードで1セッション1レコード追記し、tune_stress.py がこれを読んでフィットする。
STAI_DATASET_PATH = os.path.join(OUTPUT_DIR, "stai_dataset.jsonl")
# STAI 質問紙の項目文・逆転項目・選択肢アンカーを外部化した編集可能ファイル。
# survey モードで終了時にこの質問紙を提示し、逆転採点して STAI-S/-T を算出する。
# 無ければ雛形（プレースホルダ項目文）を自動生成するので、正式な日本語項目文に差し替える。
# ※これは「入力データ」でありバージョン管理対象なのでリポジトリ内を優先して読む。
#   DATA_DIR 側に置けば現場ごとの差し替えも可能（そちらを優先）。
_STAI_ITEMS_IN_DATA = os.path.join(DATA_DIR, "stai_items.json")
_STAI_ITEMS_IN_REPO = os.path.join(REPO_DIR, "stai_items.json")
STAI_ITEMS_PATH = (
    _STAI_ITEMS_IN_DATA if os.path.exists(_STAI_ITEMS_IN_DATA) else _STAI_ITEMS_IN_REPO
)
# 定点観測用の永続データ（人物ごとの顔埋め込み・平常状態統計）とセッション履歴
PROFILES_JSON_PATH = os.path.join(OUTPUT_DIR, "person_profiles.json")
SESSION_HISTORY_PATH = os.path.join(OUTPUT_DIR, "session_history.jsonl")
# モデルは MODEL_DIR にキャッシュする。ここに残っている限り再ダウンロードされない。
FACE_LANDMARKER_TASK = os.path.join(MODEL_DIR, "face_landmarker.task")
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# DeepFace(SFace) の重みファイル。DeepFace 内蔵のダウンローダは不安定なので、
# 確実な urllib で事前にキャッシュへ配置する（無ければ）。
DEEPFACE_WEIGHTS_DIR = os.path.join(os.path.expanduser("~"), ".deepface", "weights")
SFACE_WEIGHT_NAME = "face_recognition_sface_2021dec.onnx"
SFACE_WEIGHT_PATH = os.path.join(DEEPFACE_WEIGHTS_DIR, SFACE_WEIGHT_NAME)
SFACE_WEIGHT_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


# 実行モード（stress_config.json の "mode" で切り替える。既定は run＝従来どおり何も聞かない）:
#   "run"     = 調整済みで実運用。終了時に何も聞かない。
#   "collect" = 終了時に別途採点済みの STAI 得点(20-80)を手入力してデータセットへ蓄積。
#   "survey"  = 終了時にアプリ内で STAI 質問紙(20項目)に回答→逆転項目を含め自動採点→
#               算出した STAI-S/-T をデータセットへ保存（別紙での実施・手入力が不要）。
RUN_MODE = "run"

# --- STAI 質問紙（survey モード）---
# STAI は 4 件法（各項目 1〜4）。20 項目合計で 20〜80 点になる。
STAI_SCALE_MIN = 1
STAI_SCALE_MAX = 4
STAI_ITEMS_PER_SCALE = 20
