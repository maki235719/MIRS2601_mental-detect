"""
カメラ映像からリアルタイムで表情（感情）を認識し、ストレス度を評価するスクリプト（高精度版）

■ 旧版からの主な変更点
  - 顔検出/ランドマーク: OpenCV Haar → MediaPipe FaceLandmarker（Tasks API, 478点＋blendshapes）
  - 感情認識: DeepFace(FER2013系) → HSEmotion(EfficientNet系, 8クラス) に置換して精度向上
  - ストレス評価: 感情の線形加重 → 「起動時の平常状態からの逸脱(zスコア)」で再設計
      * 感情のネガティブ度に加え、blendshapesから得た
        「眉間のしわ(browDown)」「まばたき率」を微細特徴として合成
      * 個人ごとに較正するので、恣意的な固定係数に依存しない
  - 感情確率を時間平滑化(EMA)＋低信頼度ゲートで表示のブレを抑制
  - 顔識別: 深層顔認識の埋め込み(既定=DeepFaceのSFace)で「同じ顔か」を判定し、
      フレームをまたいで安定した人物ID(タグ)を付与。1フレームの誤判定でIDが
      暴れないようヒステリシスで連続性を持たせる。幾何ランドマーク方式は別人でも
      類似度が飽和し分離できないため最終手段に格下げ（insightfaceも選択可）。

必要ライブラリ:
    pip install opencv-python mediapipe hsemotion-onnx onnxruntime tf-keras matplotlib
    （任意・高精度化）pip install insightface

使い方:
    python face-detect2.py
    起動直後に数秒間の「較正」フェーズがあります。画面の指示どおり
    平常（リラックスした無表情）の状態を保ってください。
    その後リアルタイム評価に移ります。'q' キー（またはCtrl+C）で終了します。
    終了時に stress_log.csv と stress_graph.png を保存します。
"""

import csv
import json
import os
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# 顔識別（同一人物判定）用。未インストールでも本体は従来どおり動くよう optional にする。
try:
    from insightface.app import FaceAnalysis
    _HAS_INSIGHTFACE = True
except Exception:
    _HAS_INSIGHTFACE = False

# DeepFace（深層顔認識・既定バックエンド）。幾何ランドマークより別人分離が桁違いに強い。
try:
    from deepface import DeepFace
    _HAS_DEEPFACE = True
except Exception:
    _HAS_DEEPFACE = False

import matplotlib
matplotlib.use("Agg")  # imshowウィンドウと競合しない保存専用バックエンド
import matplotlib.pyplot as plt


# ============================================================================
# 設定（定数は config.py に集約。tune_stress.py が書き戻す可変値は stress_config.json）
# ============================================================================

from config import (
    NEGATIVE_EMOTIONS,
    ANALYZE_EVERY,
    HSEMOTION_MODEL,
    EMO_EMA_ALPHA,
    STRESS_EMA_ALPHA,
    STRESS_COMPONENT_WEIGHTS,
    STRESS_BASE_LEVEL,
    STRESS_Z_GAIN,
    EMO_STD_FLOOR,
    BROW_STD_FLOOR,
    BLINK_STD_FLOOR,
    HEAD_STD_FLOOR,
    MOUTH_STD_FLOOR,
    EYE_STD_FLOOR,
    HEAD_EMA_ALPHA,
    LOCO_EMA_ALPHA,
    LOCO_GATE_LOW,
    LOCO_GATE_HIGH,
    CALIB_SECONDS,
    BLINK_ON_THRESHOLD,
    BLINK_OFF_THRESHOLD,
    BLINK_WINDOW_SEC,
    FACE_CROP_MARGIN,
    STRESS_LOW_THRESHOLD,
    STRESS_HIGH_THRESHOLD,
    FACE_ID_ENABLED,
    FACE_ID_BACKEND,
    FACE_ID_EVERY,
    FACE_ID_ALIGN,
    FACE_ID_SWITCH_PATIENCE,
    FACE_ID_EMBED_EMA,
    FACE_ID_MODEL_DEEPFACE,
    FACE_ID_SIM_THRESHOLD_DEEPFACE,
    FACE_ID_MODEL,
    FACE_ID_SIM_THRESHOLD,
    FACE_ID_DET_SIZE,
    FACE_ID_SIM_THRESHOLD_LMK,
    STRESS_CSV_PATH,
    STRESS_GRAPH_PATH,
    STRESS_CONFIG_PATH,
    STAI_DATASET_PATH,
    STAI_ITEMS_PATH,
    PROFILES_JSON_PATH,
    SESSION_HISTORY_PATH,
    FACE_LANDMARKER_TASK,
    FACE_LANDMARKER_URL,
    DEEPFACE_WEIGHTS_DIR,
    SFACE_WEIGHT_PATH,
    SFACE_WEIGHT_URL,
    RUN_MODE,
    STAI_SCALE_MIN,
    STAI_SCALE_MAX,
    STAI_ITEMS_PER_SCALE,
)


# ============================================================================
# 設定ファイル（チューニング可能パラメータの外部化）
# ============================================================================

def _default_stress_config():
    """現行のハードコード値から config 雛形（辞書）を組み立てる。"""
    return {
        "mode": RUN_MODE,
        "weights": dict(STRESS_COMPONENT_WEIGHTS),
        "base_level": STRESS_BASE_LEVEL,
        "z_gain": STRESS_Z_GAIN,
        "std_floors": {
            "emo": EMO_STD_FLOOR,
            "brow": BROW_STD_FLOOR,
            "blink": BLINK_STD_FLOOR,
            "head": HEAD_STD_FLOOR,
            "mouth": MOUTH_STD_FLOOR,
            "eye": EYE_STD_FLOOR,
        },
    }


def load_stress_config(path=STRESS_CONFIG_PATH):
    """stress_config.json を読み、ストレス合成のパラメータ（重み・ゲイン・基準・分散下限）と
    実行モードをモジュール全体へ反映する。ファイルが無ければ現行値で雛形を書き出して従来どおり動く。
    tune_stress.py が書き戻したチューニング結果を、ここで一元的に取り込む。"""
    global STRESS_BASE_LEVEL, STRESS_Z_GAIN, RUN_MODE
    global EMO_STD_FLOOR, BROW_STD_FLOOR, BLINK_STD_FLOOR
    global HEAD_STD_FLOOR, MOUTH_STD_FLOOR, EYE_STD_FLOOR

    if not os.path.exists(path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_default_stress_config(), f, ensure_ascii=False, indent=2)
            print(f"設定ファイルの雛形を作成しました: {path}")
        except Exception as e:
            print(f"設定ファイルの作成に失敗しました（既定値で続行）: {e}")
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"設定ファイルの読み込みに失敗しました（既定値で続行）: {e}")
        return

    RUN_MODE = str(cfg.get("mode", RUN_MODE)).lower()

    # 重み（キーは既存のものだけ採用し、欠けは現行値を維持）
    for k, v in (cfg.get("weights") or {}).items():
        if k in STRESS_COMPONENT_WEIGHTS:
            STRESS_COMPONENT_WEIGHTS[k] = float(v)

    if "base_level" in cfg:
        STRESS_BASE_LEVEL = float(cfg["base_level"])
    if "z_gain" in cfg:
        STRESS_Z_GAIN = float(cfg["z_gain"])

    floors = cfg.get("std_floors") or {}
    if "emo" in floors:
        EMO_STD_FLOOR = float(floors["emo"])
    if "brow" in floors:
        BROW_STD_FLOOR = float(floors["brow"])
    if "blink" in floors:
        BLINK_STD_FLOOR = float(floors["blink"])
    if "head" in floors:
        HEAD_STD_FLOOR = float(floors["head"])
    if "mouth" in floors:
        MOUTH_STD_FLOOR = float(floors["mouth"])
    if "eye" in floors:
        EYE_STD_FLOOR = float(floors["eye"])

    # 分散下限を参照するテーブル（後段で定義済み）も同期しておく。
    _sync_feature_std_floors()
    print(
        f"設定を読み込みました（mode={RUN_MODE}）: "
        f"weights={STRESS_COMPONENT_WEIGHTS}, base={STRESS_BASE_LEVEL}, gain={STRESS_Z_GAIN}"
    )


def _sync_feature_std_floors():
    """*_STD_FLOOR の現在値を _FEATURE_STD_FLOORS（Welford用テーブル）へ反映する。
    config で下限を上書きした場合に、蓄積統計側の下限とズレないようにするため。"""
    _FEATURE_STD_FLOORS["emo"] = EMO_STD_FLOOR
    _FEATURE_STD_FLOORS["brow"] = BROW_STD_FLOOR
    _FEATURE_STD_FLOORS["blink"] = BLINK_STD_FLOOR
    _FEATURE_STD_FLOORS["head"] = HEAD_STD_FLOOR
    _FEATURE_STD_FLOORS["mouth"] = MOUTH_STD_FLOOR
    _FEATURE_STD_FLOORS["eye"] = EYE_STD_FLOOR


# ============================================================================
# モデル準備
# ============================================================================

def ensure_face_landmarker_model():
    """FaceLandmarker のモデルバンドル(.task)が無ければダウンロードする"""
    if not os.path.exists(FACE_LANDMARKER_TASK):
        print("FaceLandmarker モデルをダウンロードしています...")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_TASK)
        print("ダウンロード完了。")


def ensure_sface_model():
    """SFace の重み(.onnx)が無ければ urllib で DeepFace のキャッシュへダウンロードする。
    DeepFace 内蔵ダウンローダが失敗する環境でも確実に配置するためのフォールバック。"""
    if os.path.exists(SFACE_WEIGHT_PATH):
        return
    print("SFace モデルをダウンロードしています...")
    os.makedirs(DEEPFACE_WEIGHTS_DIR, exist_ok=True)
    req = urllib.request.Request(SFACE_WEIGHT_URL, headers={"User"
    "-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(SFACE_WEIGHT_PATH, "wb") as f:
        f.write(r.read())
    print("ダウンロード完了。")


def create_face_landmarker():
    """blendshapes 出力付きの FaceLandmarker(VIDEOモード) を生成する"""
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_TASK),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,  # 頭部姿勢（回転）を取得するため
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


# ============================================================================
# 特徴抽出
# ============================================================================

def landmarks_to_bbox(landmarks, frame_w, frame_h, margin=FACE_CROP_MARGIN):
    """正規化ランドマーク列 → 画素座標の外接矩形 (x, y, w, h)。余白付き＆画面内にクリップ。"""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1, x2 = min(xs) * frame_w, max(xs) * frame_w
    y1, y2 = min(ys) * frame_h, max(ys) * frame_h
    mw = (x2 - x1) * margin
    mh = (y2 - y1) * margin
    x1 = int(max(0, x1 - mw))
    y1 = int(max(0, y1 - mh))
    x2 = int(min(frame_w, x2 + mw))
    y2 = int(min(frame_h, y2 + mh))
    return x1, y1, x2 - x1, y2 - y1


def blendshapes_to_dict(blendshape_categories):
    """blendshape の Category リスト → {名前: スコア} の辞書"""
    return {c.category_name: c.score for c in blendshape_categories}


def brow_down_value(bs):
    """眉間のしわ（corrugator近似）。browDownLeft/Right の平均。高いほど眉を寄せている。"""
    return (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2.0


def blink_signal(bs):
    """まばたき信号。左右 eyeBlink の大きい方（片目つむりにも反応）。"""
    return max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))


def emotion_negative_affect(prob_dict):
    """感情確率(合計約1)からネガティブ度スカラーを算出。範囲はおおよそ -1〜+1。"""
    neg = sum(prob_dict.get(e, 0.0) for e in NEGATIVE_EMOTIONS)
    return neg - prob_dict.get("Happiness", 0.0)


def mouth_tension_value(bs):
    """口唇の緊張（AU23/24 近似）。左右 mouthPress の平均に mouthPucker を補助加重。
    高いほど唇を強く結んでいる。ストレスと相関（Giannakakis 2017 ほか）。"""
    press = (bs.get("mouthPressLeft", 0.0) + bs.get("mouthPressRight", 0.0)) / 2.0
    pucker = bs.get("mouthPucker", 0.0)
    return press + 0.5 * pucker


def eye_tension_value(bs):
    """瞼の緊張（AU7 lid tightener 近似）。左右 eyeSquint の平均に eyeWide(AU5) を補助加重。
    主観的ストレスと相関（Automatic stress analysis from AUs, 2021）。"""
    squint = (bs.get("eyeSquintLeft", 0.0) + bs.get("eyeSquintRight", 0.0)) / 2.0
    wide = (bs.get("eyeWideLeft", 0.0) + bs.get("eyeWideRight", 0.0)) / 2.0
    return squint + 0.5 * wide


def matrix_to_euler(matrix):
    """4x4 顔姿勢変換行列の回転部から オイラー角(pitch, yaw, roll)[度] を分解する。
    matrix が None の場合は (0,0,0) を返す。"""
    if matrix is None:
        return 0.0, 0.0, 0.0
    m = np.asarray(matrix, dtype=np.float64)
    r = m[:3, :3]
    sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(-r[2, 0], sy)
        roll = np.arctan2(r[1, 0], r[0, 0])
    else:  # ジンバルロック近傍
        pitch = np.arctan2(-r[1, 2], r[1, 1])
        yaw = np.arctan2(-r[2, 0], sy)
        roll = 0.0
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)


def matrix_to_translation(matrix):
    """4x4 顔姿勢変換行列の並進部 (tx, ty, tz) を返す。回転とは独立な「頭の位置」。
    matrix が None の場合はゼロベクトルを返す。"""
    if matrix is None:
        return np.zeros(3)
    m = np.asarray(matrix, dtype=np.float64)
    return m[:3, 3].copy()


# ============================================================================
# ストレススコア
# ============================================================================

def zscore(value, mean, std, std_floor):
    return (value - mean) / max(std, std_floor)


def locomotion_gate(loco_speed):
    """並進速度→頭部運動を信用する度合いを返す。
    1.0=静止(頭部運動を全採用) 〜 0.0=移動中(頭部運動を無効化)。間は線形。
    歩行時に「移動」を「頭単体の動き＝ストレス」と誤評価しないためのゲート。"""
    if loco_speed <= LOCO_GATE_LOW:
        return 1.0
    if loco_speed >= LOCO_GATE_HIGH:
        return 0.0
    return 1.0 - (loco_speed - LOCO_GATE_LOW) / (LOCO_GATE_HIGH - LOCO_GATE_LOW)


def stress_z_components(emo_neg, brow, blink_rate, head_motion, mouth, eye, baseline,
                        head_gate=1.0):
    """各成分をベースラインからのzスコアにして {emo,brow,blink,head,mouth,eye} で返す。
    重み・ゲインに依存しない“パラメータ非依存”な量なので、これを記録しておけば
    tune_stress.py が重みだけを差し替えてオフラインで再スコア/精度検証できる。
    head_gate は 0-1 の頭部運動の信頼度（移動中は小さくして頭部寄与を抑制する）。"""
    return {
        "emo": zscore(emo_neg, baseline["emo_mean"], baseline["emo_std"], EMO_STD_FLOOR),
        "brow": zscore(brow, baseline["brow_mean"], baseline["brow_std"], BROW_STD_FLOOR),
        "blink": zscore(
            blink_rate, baseline["blink_mean"], baseline["blink_std"], BLINK_STD_FLOOR
        ),
        # 移動中(head_gate<1)は頭部運動の寄与を抑え、歩行を頭単体の動きと混同しない
        "head": zscore(
            head_motion, baseline["head_mean"], baseline["head_std"], HEAD_STD_FLOOR
        ) * head_gate,
        "mouth": zscore(mouth, baseline["mouth_mean"], baseline["mouth_std"], MOUTH_STD_FLOOR),
        "eye": zscore(eye, baseline["eye_mean"], baseline["eye_std"], EYE_STD_FLOOR),
    }


def stress_from_z(z):
    """6成分zの辞書 → 重み付き和 → 0-100 に写像する。重み/ゲイン/基準は config で可変。
    重みキー(emotion)と成分キー(emo)の対応に注意（履歴上の命名差）。"""
    z_total = (
        STRESS_COMPONENT_WEIGHTS["emotion"] * z["emo"]
        + STRESS_COMPONENT_WEIGHTS["brow"] * z["brow"]
        + STRESS_COMPONENT_WEIGHTS["blink"] * z["blink"]
        + STRESS_COMPONENT_WEIGHTS["head"] * z["head"]
        + STRESS_COMPONENT_WEIGHTS["mouth"] * z["mouth"]
        + STRESS_COMPONENT_WEIGHTS["eye"] * z["eye"]
    )
    score = STRESS_BASE_LEVEL + STRESS_Z_GAIN * z_total
    return max(0.0, min(100.0, score))


def compose_stress(emo_neg, brow, blink_rate, head_motion, mouth, eye, baseline,
                   head_gate=1.0):
    """各成分をベースラインからのzスコアにし、重み付き和 → 0-100 に写像する。
    head_gate は 0-1 の頭部運動の信頼度（移動中は小さくして頭部寄与を抑制する）。"""
    z = stress_z_components(
        emo_neg, brow, blink_rate, head_motion, mouth, eye, baseline, head_gate
    )
    return stress_from_z(z)


def stress_color(score):
    """ストレススコアに応じた枠/バーの色 (BGR) を返す（緑→黄→赤）"""
    if score < STRESS_LOW_THRESHOLD:
        return (0, 200, 0)      # 緑
    elif score < STRESS_HIGH_THRESHOLD:
        return (0, 200, 255)    # 黄
    else:
        return (0, 0, 255)      # 赤


# ============================================================================
# まばたきカウンタ
# ============================================================================

class BlinkCounter:
    """eyeBlink 信号のヒステリシス立ち上がりで瞬目を数え、直近窓の瞬目率(回/分)を返す。"""

    def __init__(self):
        self.is_closed = False
        self.timestamps = deque()  # まばたき発生時刻(秒)

    def update(self, signal, now):
        if not self.is_closed and signal >= BLINK_ON_THRESHOLD:
            self.is_closed = True
            self.timestamps.append(now)  # 目を閉じた瞬間を1回とカウント
        elif self.is_closed and signal <= BLINK_OFF_THRESHOLD:
            self.is_closed = False
        # 窓外の古い記録を捨てる
        while self.timestamps and now - self.timestamps[0] > BLINK_WINDOW_SEC:
            self.timestamps.popleft()

    def rate_per_min(self, now, start_time):
        """直近窓の瞬目率(回/分)。経過が窓長に満たない間は経過時間で正規化する。"""
        elapsed = now - start_time
        window = min(BLINK_WINDOW_SEC, elapsed)
        if window < 1.0:
            return 0.0
        count = sum(1 for t in self.timestamps if now - t <= window)
        return count * 60.0 / window


# ============================================================================
# 頭部運動トラッカ
# ============================================================================

class HeadMotionTracker:
    """頭部姿勢のオイラー角(pitch/yaw/roll)から角速度(deg/秒)を求め、EMA平滑化して返す。
    ストレス下で頭部運動が増える傾向（Giannakakis, head pose features）を特徴量化する。"""

    def __init__(self):
        self.prev_angles = None
        self.prev_time = None
        self.smoothed = 0.0

    def update(self, angles, now):
        pitch, yaw, roll = angles
        if self.prev_angles is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 1e-3:
                d = np.array([pitch, yaw, roll]) - np.array(self.prev_angles)
                speed = float(np.sqrt(np.sum(d ** 2)) / dt)  # deg/秒
                self.smoothed = (
                    HEAD_EMA_ALPHA * speed + (1 - HEAD_EMA_ALPHA) * self.smoothed
                )
        self.prev_angles = (pitch, yaw, roll)
        self.prev_time = now
        return self.smoothed


class LocomotionTracker:
    """頭部姿勢行列の並進成分(頭の位置)から移動速度を求め、EMA平滑化して返す。
    回転(頭の向き)とは独立なので、歩行など全身の移動を頭単体の動き(回転)と分離できる。"""

    def __init__(self):
        self.prev_t = None
        self.prev_time = None
        self.smoothed = 0.0

    def reset(self):
        """顔をロストした後などに前回位置を破棄し、再取得時の速度スパイクを防ぐ。"""
        self.prev_t = None
        self.prev_time = None

    def update(self, translation, now):
        t = np.asarray(translation, dtype=np.float64)
        if self.prev_t is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 1e-3:
                speed = float(np.linalg.norm(t - self.prev_t) / dt)
                self.smoothed = (
                    LOCO_EMA_ALPHA * speed + (1 - LOCO_EMA_ALPHA) * self.smoothed
                )
        self.prev_t = t
        self.prev_time = now
        return self.smoothed


# ============================================================================
# 顔識別（同一人物判定・タグ付け・連続性）
# ============================================================================

def bbox_iou(a, b):
    """2つの矩形 (x1,y1,x2,y2) の IoU（重なり具合, 0-1）を返す。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def create_face_recognizer():
    """顔認識器(insightface)を生成する。無効/未導入なら None を返す。"""
    if not (FACE_ID_ENABLED and _HAS_INSIGHTFACE):
        return None
    try:
        app = FaceAnalysis(
            name=FACE_ID_MODEL,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(FACE_ID_DET_SIZE, FACE_ID_DET_SIZE))
        return app
    except Exception as e:
        print(f"顔識別器の初期化に失敗したため、識別機能は無効になります: {e}")
        return None


def pick_face_for_bbox(recognizer, frame_bgr, target_bbox):
    """フレームから顔を検出し、target_bbox に最も重なる顔を返す（無ければ最大の顔）。
    target_bbox が None のときは最大の顔を返す。顔が無ければ None。"""
    faces = recognizer.get(frame_bgr)
    if not faces:
        return None
    if target_bbox is None:
        return max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
    tx, ty, tw, th = target_bbox
    target = (tx, ty, tx + tw, ty + th)
    best, best_iou = None, 0.0
    for f in faces:
        iou = bbox_iou(target, tuple(f.bbox))
        if iou > best_iou:
            best_iou, best = iou, f
    if best is not None:
        return best
    # 重なりゼロでも顔はある → 最大の顔で代用
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


# ランドマーク方式（insightface無し）で顔署名に使う代表点のインデックス。
# 表情で動きにくい輪郭・目・眉・鼻・頬など、個人差の出やすい点を選ぶ。
_LMK_EMBED_IDS = [
    33, 133, 362, 263,       # 目尻・目頭
    70, 105, 336, 300,       # 眉
    168, 6, 197, 195, 5, 4,  # 鼻筋・鼻先
    61, 291, 0, 17,          # 口角・唇中央（上下）
    234, 454, 10, 152,       # 顔の左右端・上下端
    50, 280,                 # 頬
]


def _mean_point(landmarks, ids):
    return np.mean([[landmarks[i].x, landmarks[i].y] for i in ids], axis=0)


def landmark_face_embedding(landmarks):
    """MediaPipe の顔ランドマークから、位置・大きさ・面内回転(roll)に不変な
    形状ベクトルを作る（insightface が無いときの簡易な顔識別用）。
    両目中心で正規化するため、正面〜やや斜めの顔で同一人物を安定して識別できる。"""
    if landmarks is None or len(landmarks) < 468:
        return None
    eye_l = _mean_point(landmarks, (33, 133))    # 左目中心
    eye_r = _mean_point(landmarks, (362, 263))   # 右目中心
    center = (eye_l + eye_r) / 2.0
    diff = eye_r - eye_l
    dist = float(np.linalg.norm(diff))
    if dist < 1e-6:
        return None
    # 両目を水平に戻す回転（面内回転の除去）
    angle = np.arctan2(diff[1], diff[0])
    ca, sa = np.cos(-angle), np.sin(-angle)
    rot = np.array([[ca, -sa], [sa, ca]])
    pts = []
    for i in _LMK_EMBED_IDS:
        lm = landmarks[i]
        p = rot @ (np.array([lm.x, lm.y]) - center) / dist  # 中心化→回転補正→スケール正規化
        pts.append(p)
    vec = np.asarray(pts, dtype=np.float64).flatten()
    vec = vec - vec.mean()          # 共通成分を除き個人差を強調
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return None
    return vec / norm               # L2正規化（コサイン類似度で照合するため）


def align_face_crop(frame, landmarks, bbox):
    """認識精度を上げるため、両目が水平になるようフレームを面内回転してから
    bbox 範囲を切り出す。FACE_ID_ALIGN=False のときは素のクロップを返す。"""
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    if not FACE_ID_ALIGN or landmarks is None or len(landmarks) < 468:
        return frame[y : y + h, x : x + w]
    fh, fw = frame.shape[:2]
    eye_l = _mean_point(landmarks, (33, 133)) * np.array([fw, fh])   # 画素座標
    eye_r = _mean_point(landmarks, (362, 263)) * np.array([fw, fh])
    dx, dy = (eye_r - eye_l)
    angle = np.degrees(np.arctan2(dy, dx))            # 目線の傾き（度）
    center = tuple(((eye_l + eye_r) / 2.0).tolist())
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)  # 目線を水平に戻す回転
    rotated = cv2.warpAffine(frame, rot, (fw, fh), flags=cv2.INTER_LINEAR)
    return rotated[y : y + h, x : x + w]


def deepface_face_embedding(face_bgr):
    """顔クロップ(BGR)を DeepFace(SFace)で埋め込みに変換しL2正規化して返す。
    detector_backend='skip' で検出を省き、こちらが渡したクロップをそのまま使う。"""
    if face_bgr is None or face_bgr.size == 0:
        return None
    reps = DeepFace.represent(
        face_bgr,
        model_name=FACE_ID_MODEL_DEEPFACE,
        enforce_detection=False,
        detector_backend="skip",
        align=False,
    )
    if not reps:
        return None
    emb = np.asarray(reps[0]["embedding"], dtype=np.float64)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-9 else None


def resolve_face_id_backend():
    """設定と導入状況から実際に使う顔識別バックエンドを決める。使えなければ None。"""
    if not FACE_ID_ENABLED:
        return None
    choice = FACE_ID_BACKEND
    if choice == "auto":
        if _HAS_DEEPFACE:
            return "deepface"
        if _HAS_INSIGHTFACE:
            return "insightface"
        return "landmark"
    if choice == "deepface":
        return "deepface" if _HAS_DEEPFACE else None
    if choice == "insightface":
        return "insightface" if _HAS_INSIGHTFACE else None
    if choice == "landmark":
        return "landmark"
    return None


class FaceIdentityTracker:
    """顔の埋め込みを既知の人物（ギャラリー）と照合して同一人物か判定し、
    フレームをまたいで安定した ID（タグ）を割り当てる。

    連続性の要: 現在の ID と違う相手が検出されても、それが switch_patience 回
    連続するまではタグを切り替えない（1フレームの誤判定でIDが暴れるのを防ぐ）。
    埋め込みは L2 正規化済み前提なので、コサイン類似度＝内積で計算する。
    """

    def __init__(
        self,
        sim_threshold=FACE_ID_SIM_THRESHOLD,
        switch_patience=FACE_ID_SWITCH_PATIENCE,
        embed_ema=FACE_ID_EMBED_EMA,
    ):
        self.sim_threshold = sim_threshold
        self.switch_patience = switch_patience
        self.embed_ema = embed_ema
        self.gallery = {}          # id -> 代表埋め込み(正規化済み)
        self.next_id = 1
        self.current_id = None
        self._pending_id = None    # 切り替え候補（None は「新規人物」候補）
        self._pending_count = 0

    def seed(self, gallery, next_id):
        """永続化された既知人物（id -> 正規化済み埋め込み）を注入する。
        これによりセッションをまたいで同じ顔に同じIDが付く（定点観測の土台）。"""
        if gallery:
            self.gallery = {int(k): np.asarray(v, dtype=np.float64) for k, v in gallery.items()}
        self.next_id = max(int(next_id), (max(self.gallery) + 1) if self.gallery else 1)

    def _reset_pending(self):
        self._pending_id = None
        self._pending_count = 0

    def _best_match(self, emb):
        """ギャラリー中で最も類似する (id, 類似度) を返す。空なら (None, -1)。"""
        best_id, best_sim = None, -1.0
        for fid, ref in self.gallery.items():
            sim = float(np.dot(emb, ref))
            if sim > best_sim:
                best_sim, best_id = sim, fid
        return best_id, best_sim

    def _create(self, emb):
        fid = self.next_id
        self.next_id += 1
        self.gallery[fid] = emb.copy()
        return fid

    def _update_embedding(self, fid, emb):
        blended = (1 - self.embed_ema) * self.gallery[fid] + self.embed_ema * emb
        self.gallery[fid] = blended / (np.linalg.norm(blended) + 1e-9)

    def update(self, embedding):
        """埋め込みを1つ受け取り、(現在ID, 最良類似度, 切替が起きたか) を返す。"""
        best_id, best_sim = self._best_match(embedding)
        candidate = best_id if best_sim >= self.sim_threshold else None

        # 初回はその場で確定（新規なら新IDを発番）
        if self.current_id is None:
            self.current_id = candidate if candidate is not None else self._create(embedding)
            self._update_embedding(self.current_id, embedding)
            self._reset_pending()
            return self.current_id, best_sim, True

        # 現在のIDと一致 → 連続性を維持し、代表埋め込みを微更新
        if candidate == self.current_id:
            self._update_embedding(self.current_id, embedding)
            self._reset_pending()
            return self.current_id, best_sim, False

        # 現在と違う相手（別人 or 新規）が来た → 連続して続くか様子を見る
        if candidate == self._pending_id:
            self._pending_count += 1
        else:
            self._pending_id = candidate
            self._pending_count = 1

        if self._pending_count >= self.switch_patience:
            if candidate is None:
                self.current_id = self._create(embedding)
            else:
                self.current_id = candidate
                self._update_embedding(self.current_id, embedding)
            self._reset_pending()
            return self.current_id, best_sim, True

        # まだ確信できない → タグは維持（連続性）
        return self.current_id, best_sim, False


# ============================================================================
# 永続化・人物プロファイル（定点観測）
# ============================================================================
# 人物ごとの「顔埋め込み」と「平常状態統計」を JSON に保存し、セッションをまたいで
# 蓄積する。これにより (1) 同じ顔に同じIDが付き、(2) その人の“普段”に対する相対
# ストレスを、観測を重ねるほど正確に評価できる（=定点観測）。

# 各特徴の Welford 統計に対応する標準偏差の下限（既存の *_STD_FLOOR を流用）
_FEATURE_STD_FLOORS = {
    "emo": EMO_STD_FLOOR,
    "brow": BROW_STD_FLOOR,
    "blink": BLINK_STD_FLOOR,
    "head": HEAD_STD_FLOOR,
    "mouth": MOUTH_STD_FLOOR,
    "eye": EYE_STD_FLOOR,
}
_FEATURES = list(_FEATURE_STD_FLOORS.keys())


def _new_welford():
    return {"count": 0, "mean": 0.0, "M2": 0.0}


def welford_update(stat, x):
    """Welford のオンライン更新。全観測を保持せず平均・分散をセッション横断で更新する。"""
    stat["count"] += 1
    delta = x - stat["mean"]
    stat["mean"] += delta / stat["count"]
    stat["M2"] += delta * (x - stat["mean"])


def welford_std(stat, floor):
    """Welford 統計から母標準偏差を求め floor でガードする（平常が静かすぎる時の暴れ防止）。"""
    if stat["count"] < 1:
        return floor
    var = stat["M2"] / stat["count"]
    return max(float(np.sqrt(max(var, 0.0))), floor)


def face_id_backend_tag(id_backend):
    """埋め込みはバックエンド間で非互換なので、どのバックエンドで作った埋め込みかを識別する。"""
    if id_backend == "deepface":
        return f"deepface:{FACE_ID_MODEL_DEEPFACE}"
    if id_backend == "insightface":
        return f"insightface:{FACE_ID_MODEL}"
    if id_backend == "landmark":
        return "landmark"
    return "none"


class PersonStore:
    """person_profiles.json の読み書きと、平常状態統計のセッション横断蓄積を担う。"""

    def __init__(self, backend_tag, persons=None, next_id=1):
        self.backend_tag = backend_tag
        self.persons = persons or {}      # id(int) -> プロファイル辞書
        self.next_id = next_id
        self._backend_mismatch = False

    @classmethod
    def load(cls, path, backend_tag):
        if not os.path.exists(path):
            return cls(backend_tag)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"プロファイル読み込みに失敗しました（新規で開始します）: {e}")
            return cls(backend_tag)
        persons = {int(k): v for k, v in data.get("persons", {}).items()}
        store = cls(backend_tag, persons, int(data.get("next_id", 1)))
        saved_backend = data.get("backend")
        if saved_backend and saved_backend != backend_tag:
            print(
                f"顔認識バックエンドが前回({saved_backend})と異なります({backend_tag})。"
                "保存済みの顔埋め込みは無効化し、平常状態統計のみ引き継ぎます。"
            )
            store._backend_mismatch = True
        return store

    def gallery_for_seed(self):
        """FaceIdentityTracker に注入する {id: 埋め込み}。バックエンド不一致時は空。"""
        if self._backend_mismatch:
            return {}
        return {
            pid: p["embedding"]
            for pid, p in self.persons.items()
            if p.get("embedding")
        }

    def baseline_dict(self, person_id):
        """Welford 統計を compose_stress が要求する {emo_mean, emo_std, ...} 形へ変換。
        平常状態がまだ無ければ None（相対評価不能）を返す。"""
        p = self.persons.get(int(person_id))
        if not p or not p.get("baseline"):
            return None
        bl = p["baseline"]
        if not any(bl.get(f, {}).get("count", 0) > 0 for f in _FEATURES):
            return None
        out = {}
        for f in _FEATURES:
            stat = bl.get(f, _new_welford())
            out[f"{f}_mean"] = float(stat.get("mean", 0.0))
            out[f"{f}_std"] = welford_std(stat, _FEATURE_STD_FLOORS[f])
        return out

    def ingest_calibration(self, person_id, samples):
        """このセッションの較正サンプルを該当人物の Welford 統計へ畳み込む（無ければ作成）。
        samples: {"emo":[...],"brow":[...],"head":[...],"mouth":[...],"eye":[...],"blink": rate}
        戻り値: マージ後のベースライン辞書（compose_stress 用）。"""
        pid = int(person_id)
        p = self.persons.get(pid)
        if p is None:
            p = {
                "embedding": None,
                "created": datetime.now().isoformat(timespec="seconds"),
                "updated": None,
                "session_count": 0,
                "baseline": {f: _new_welford() for f in _FEATURES},
            }
            self.persons[pid] = p
        bl = p.setdefault("baseline", {})
        for f in _FEATURES:
            bl.setdefault(f, _new_welford())
        for f in ("emo", "brow", "head", "mouth", "eye"):
            for x in samples.get(f, []):
                welford_update(bl[f], float(x))
        blink_rate = samples.get("blink")
        if blink_rate is not None:
            welford_update(bl["blink"], float(blink_rate))  # 平常瞬目率（1セッション1サンプル）
        return self.baseline_dict(pid)

    def save(self, path, gallery, session_summaries, history_path):
        """最新の埋め込み・session_count を書き戻し、セッション履歴(JSONL)を追記する。"""
        now = datetime.now().isoformat(timespec="seconds")
        for pid, emb in (gallery or {}).items():
            pid = int(pid)
            p = self.persons.get(pid)
            if p is None:
                p = {
                    "embedding": None,
                    "created": now,
                    "updated": None,
                    "session_count": 0,
                    "baseline": {f: _new_welford() for f in _FEATURES},
                }
                self.persons[pid] = p
            p["embedding"] = [float(x) for x in np.asarray(emb).ravel()]
        for pid in session_summaries:
            p = self.persons.get(int(pid))
            if p is not None:
                p["session_count"] = int(p.get("session_count", 0)) + 1
                p["updated"] = now
        data = {
            "version": 1,
            "backend": self.backend_tag,
            "next_id": int(self.next_id),
            "persons": {str(pid): p for pid, p in self.persons.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"人物プロファイルを保存しました: {path}")
        if session_summaries:
            with open(history_path, "a", encoding="utf-8") as f:
                for pid, s in session_summaries.items():
                    p = self.persons.get(int(pid), {})
                    rec = {
                        "date": now,
                        "person_id": int(pid),
                        "duration_sec": round(s["duration_sec"], 1),
                        "samples": s["samples"],
                        "mean_stress": round(s["mean_stress"], 1),
                        "max_stress": round(s["max_stress"], 1),
                        "baseline_sessions": int(p.get("session_count", 0)),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"セッション履歴を追記しました: {history_path}")


# ============================================================================
# レポート保存
# ============================================================================

def save_stress_report(stress_history):
    """セッション終了時にストレススコアの時系列をCSVとグラフ画像に保存する"""
    if not stress_history:
        print("記録されたストレスデータがないため、ログ・グラフは保存しません。")
        return

    with open(STRESS_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "elapsed_sec",
                "frame_count",
                "face_id",
                "stress_score",
                "dominant_emotion",
                "emo_negative",
                "brow_down",
                "blink_rate_per_min",
                "head_motion",
                "mouth_tension",
                "eye_tension",
            ]
        )
        for row in stress_history:
            elapsed, fc, face_id, score, emo, emo_neg, brow, blink, head, mouth, eye = row
            writer.writerow(
                [
                    f"{elapsed:.2f}",
                    fc,
                    "" if face_id is None else face_id,
                    f"{score:.1f}",
                    emo,
                    f"{emo_neg:.3f}",
                    f"{brow:.3f}",
                    f"{blink:.1f}",
                    f"{head:.2f}",
                    f"{mouth:.3f}",
                    f"{eye:.3f}",
                ]
            )
    print(f"ストレスログを保存しました: {STRESS_CSV_PATH}")

    times = [row[0] for row in stress_history]
    scores = [row[3] for row in stress_history]  # row[3]=stress_score（row[2]はface_id）

    plt.figure(figsize=(10, 4))
    plt.plot(times, scores, color="tab:red", linewidth=1.5)
    plt.xlabel("Elapsed Time (sec)")
    plt.ylabel("Stress Score (0-100)")
    plt.title("Stress Score over Session")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(STRESS_GRAPH_PATH)
    plt.close()
    print(f"ストレスグラフを保存しました: {STRESS_GRAPH_PATH}")


# ============================================================================
# メイン
# ============================================================================

def main():
    # チューニング済みパラメータと実行モードを最初に取り込む（無ければ雛形を作成）。
    load_stress_config()

    ensure_face_landmarker_model()

    print("感情モデルを読み込んでいます（初回はダウンロードが走ります）...")
    fer = HSEmotionRecognizer(model_name=HSEMOTION_MODEL)
    landmarker = create_face_landmarker()

    # 顔識別器（同一人物判定）。既定は深層埋め込み(SFace/DeepFace)で別人を確実に分離する。
    print("顔識別モデルを準備しています...")
    id_backend = resolve_face_id_backend()
    recognizer = None
    id_tracker = None
    if id_backend == "deepface":
        print("顔識別モデル(SFace)を読み込んでいます（初回はダウンロードが走ります）...")
        try:
            if FACE_ID_MODEL_DEEPFACE == "SFace":
                ensure_sface_model()          # 内蔵DLが不安定なので urllib で確実に配置
            DeepFace.build_model(FACE_ID_MODEL_DEEPFACE)  # 初回DL・初回推論の遅延を起動時に寄せる
            id_tracker = FaceIdentityTracker(sim_threshold=FACE_ID_SIM_THRESHOLD_DEEPFACE)
            print("顔識別が有効です（SFace / DeepFace）。同じ顔には同じIDタグが付きます。")
        except Exception as e:
            print(f"SFaceの初期化に失敗したため、顔識別を無効化します: {e}")
            id_backend = None
    elif id_backend == "insightface":
        recognizer = create_face_recognizer()
        if recognizer is not None:
            id_tracker = FaceIdentityTracker(sim_threshold=FACE_ID_SIM_THRESHOLD)
            print("顔識別が有効です（insightface / 高精度）。同じ顔には同じIDタグが付きます。")
        else:
            id_backend = None
    elif id_backend == "landmark":
        id_tracker = FaceIdentityTracker(sim_threshold=FACE_ID_SIM_THRESHOLD_LMK)
        print(
            "顔識別が有効です（ランドマーク方式 / 簡易・別人分離は弱い）。"
            "別人を分けたい場合は DeepFace/insightface を使ってください。"
        )

    # 永続化された人物プロファイル（顔埋め込み・平常状態統計）を読み込み、
    # トラッカへ既知人物を注入する（セッションをまたいで同じ顔に同じID＝定点観測）。
    store = PersonStore.load(PROFILES_JSON_PATH, face_id_backend_tag(id_backend))
    if id_tracker is not None:
        id_tracker.seed(store.gallery_for_seed(), store.next_id)
        known = len(store.gallery_for_seed())
        if known:
            print(f"既知の人物 {known} 名のプロファイルを読み込みました（IDを引き継ぎます）。")
    # 人物ごとの有効ベースライン（読み込み済み統計から復元、較正で上書き/蓄積）
    person_baselines = {
        pid: store.baseline_dict(pid) for pid in store.persons
    }
    person_baselines = {pid: bl for pid, bl in person_baselines.items() if bl is not None}

    cap = cv2.VideoCapture(0)  # 0 = 既定のカメラ
    if not cap.isOpened():
        print("カメラを開けませんでした。デバイス番号や接続を確認してください。")
        landmarker.close()
        return

    frame_count = 0
    session_start = time.time()
    last_ts_ms = -1

    # 直近の解析状態（描画とストレス計算で共有）
    smoothed_probs = None          # 平滑化した感情確率辞書
    last_bbox = None               # 直近の顔矩形
    last_emo_neg = 0.0
    last_brow = 0.0
    last_head = 0.0                # 平滑化した頭部角速度(deg/秒)
    last_loco = 0.0                # 平滑化した移動(並進)速度。歩行判定に使う
    last_mouth = 0.0               # 口唇の緊張
    last_eye = 0.0                 # 瞼の緊張
    last_face_id = None            # 直近に割り当てた人物タグ（連続性を持つID）
    last_face_sim = 0.0            # 直近の照合類似度（デバッグ表示用）

    blink_counter = BlinkCounter()
    head_tracker = HeadMotionTracker()
    loco_tracker = LocomotionTracker()

    # 較正フェーズ用のサンプル蓄積
    calibrating = True
    calib_emo = []
    calib_brow = []
    calib_head = []
    calib_mouth = []
    calib_eye = []
    baseline = None                # 顔識別が無い場合のセッション限りフォールバック
    smoothed_stress = None
    stress_history = []
    # 人物ごとのセッション要約（session_history.jsonl / プロファイル更新に使う）
    session_summaries = {}         # pid -> {"samples","sum","max","first_t","last_t"}

    print("起動しました。まず数秒間、平常な表情を保ってください（較正中）。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレームを取得できませんでした。")
                break

            frame_count += 1
            now = time.time()
            elapsed = now - session_start

            # --- FaceLandmarker は毎フレーム実行（VIDEOモードは単調増加のtimestampが必要） ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(elapsed * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms
            result = landmarker.detect_for_video(mp_image, ts_ms)

            face_present = bool(result.face_landmarks)
            frame_h, frame_w = frame.shape[:2]

            if face_present:
                landmarks = result.face_landmarks[0]
                last_bbox = landmarks_to_bbox(landmarks, frame_w, frame_h)

                # 微細特徴（blendshapes）: 毎フレーム更新
                if result.face_blendshapes:
                    bs = blendshapes_to_dict(result.face_blendshapes[0])
                    last_brow = brow_down_value(bs)
                    last_mouth = mouth_tension_value(bs)
                    last_eye = eye_tension_value(bs)
                    blink_counter.update(blink_signal(bs), now)

                # 頭部運動（回転＝頭の向き）と移動（並進＝頭の位置）: 毎フレーム更新。
                # 同じ姿勢行列から回転と並進を別々に取り、歩行(並進)を頭単体の動き(回転)と分離。
                if result.facial_transformation_matrixes:
                    matrix = result.facial_transformation_matrixes[0]
                    last_head = head_tracker.update(matrix_to_euler(matrix), now)
                    last_loco = loco_tracker.update(matrix_to_translation(matrix), now)

                # 感情解析（HSEmotion）: 間引いて実行
                if frame_count % ANALYZE_EVERY == 0:
                    x, y, w, h = last_bbox
                    if w > 0 and h > 0:
                        face_bgr = frame[y : y + h, x : x + w]
                        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                        try:
                            _, scores = fer.predict_emotions(face_rgb, logits=False)
                            probs = {
                                fer.idx_to_class[i]: float(scores[i])
                                for i in range(len(scores))
                            }
                            # 感情確率ベクトルを EMA 平滑化
                            if smoothed_probs is None:
                                smoothed_probs = probs
                            else:
                                smoothed_probs = {
                                    k: EMO_EMA_ALPHA * probs[k]
                                    + (1 - EMO_EMA_ALPHA) * smoothed_probs.get(k, 0.0)
                                    for k in probs
                                }
                            last_emo_neg = emotion_negative_affect(smoothed_probs)
                        except Exception as e:
                            print(f"感情解析エラー: {e}")

                # 顔識別（同一人物判定・タグ付け）: 間引いて実行
                if id_tracker is not None and frame_count % FACE_ID_EVERY == 0:
                    try:
                        if id_backend == "deepface":
                            # 既定: 深層埋め込み(SFace)。目線を水平化したクロップを渡す
                            crop = align_face_crop(frame, landmarks, last_bbox)
                            emb = deepface_face_embedding(crop)
                        elif id_backend == "insightface":
                            # 高精度: insightface の埋め込み
                            face = pick_face_for_bbox(recognizer, frame, last_bbox)
                            emb = getattr(face, "normed_embedding", None) if face else None
                        else:
                            # 最終手段: 顔ランドマークから形状ベクトル
                            emb = landmark_face_embedding(landmarks)
                        if emb is not None:
                            fid, sim, changed = id_tracker.update(emb)
                            last_face_id, last_face_sim = fid, sim
                            if changed:
                                print(
                                    f"顔識別: 人物 ID={fid} を追跡中"
                                    f"（類似度 {sim:.2f}）"
                                )
                    except Exception as e:
                        print(f"顔識別エラー: {e}")
            else:
                # 顔ロスト中は前回位置を破棄し、再取得時の見かけの移動スパイクを防ぐ
                loco_tracker.reset()

            blink_rate = blink_counter.rate_per_min(now, session_start)

            # ================= 較正フェーズ =================
            if calibrating:
                if face_present:
                    calib_brow.append(last_brow)
                    # 移動中の頭部運動は平常ベースラインに含めない（移動＝ストレスと誤らせない）
                    if locomotion_gate(last_loco) > 0.5:
                        calib_head.append(last_head)
                    calib_mouth.append(last_mouth)
                    calib_eye.append(last_eye)
                    if smoothed_probs is not None:
                        calib_emo.append(last_emo_neg)

                if elapsed >= CALIB_SECONDS:
                    # ベースライン統計を確定
                    emo_arr = np.array(calib_emo) if calib_emo else np.array([0.0])
                    brow_arr = np.array(calib_brow) if calib_brow else np.array([0.0])
                    head_arr = np.array(calib_head) if calib_head else np.array([0.0])
                    mouth_arr = np.array(calib_mouth) if calib_mouth else np.array([0.0])
                    eye_arr = np.array(calib_eye) if calib_eye else np.array([0.0])
                    baseline = {
                        "emo_mean": float(emo_arr.mean()),
                        "emo_std": float(emo_arr.std()),
                        "brow_mean": float(brow_arr.mean()),
                        "brow_std": float(brow_arr.std()),
                        "blink_mean": blink_rate,          # 較正中の平常瞬目率
                        "blink_std": BLINK_STD_FLOOR,      # 瞬目率は分散推定が不安定なので下限を使う
                        "head_mean": float(head_arr.mean()),
                        "head_std": float(head_arr.std()),
                        "mouth_mean": float(mouth_arr.mean()),
                        "mouth_std": float(mouth_arr.std()),
                        "eye_mean": float(eye_arr.mean()),
                        "eye_std": float(eye_arr.std()),
                    }
                    # 顔識別で人物が特定できていれば、蓄積統計へマージして“普段”を更新。
                    # 既知の人物なら過去の平常が効いて初回フレームから相対評価できる（定点観測）。
                    calib_person_id = last_face_id
                    if calib_person_id is not None:
                        session_samples = {
                            "emo": calib_emo,
                            "brow": calib_brow,
                            "head": calib_head,
                            "mouth": calib_mouth,
                            "eye": calib_eye,
                            "blink": blink_rate,
                        }
                        merged = store.ingest_calibration(calib_person_id, session_samples)
                        if merged is not None:
                            person_baselines[calib_person_id] = merged
                            baseline = merged  # 蓄積後の平常をこのセッションの基準に採用
                            print(
                                f"人物 ID={calib_person_id} の平常状態を更新しました"
                                f"（累計 {store.persons[calib_person_id]['baseline']['brow']['count']} サンプル）。"
                            )

                    calibrating = False
                    print(
                        "較正完了。ベースライン: "
                        f"emo={baseline['emo_mean']:.2f}, "
                        f"brow={baseline['brow_mean']:.2f}, "
                        f"blink={baseline['blink_mean']:.1f}/min, "
                        f"head={baseline['head_mean']:.1f}deg/s, "
                        f"mouth={baseline['mouth_mean']:.2f}, "
                        f"eye={baseline['eye_mean']:.2f}"
                    )
            # ================= 評価フェーズ =================
            else:
                # その人の“普段”に対して評価する。蓄積プロファイルがあればそれを、
                # 無ければ（顔識別オフ時など）セッション較正のフォールバックを使う。
                active_baseline = person_baselines.get(last_face_id, baseline)
                if face_present and active_baseline is not None:
                    # zスコア成分を一度だけ計算し、スコア化と（STAI連動チューニング用の）
                    # セッション平均zの集計の両方に使い回す。
                    z_comp = stress_z_components(
                        last_emo_neg, last_brow, blink_rate,
                        last_head, last_mouth, last_eye, active_baseline,
                        head_gate=locomotion_gate(last_loco),  # 移動中は頭部運動を抑制
                    )
                    raw_score = stress_from_z(z_comp)
                    if smoothed_stress is None:
                        smoothed_stress = raw_score
                    else:
                        smoothed_stress = (
                            STRESS_EMA_ALPHA * raw_score
                            + (1 - STRESS_EMA_ALPHA) * smoothed_stress
                        )
                    dominant = (
                        max(smoothed_probs, key=smoothed_probs.get)
                        if smoothed_probs
                        else ""
                    )
                    # 人物ごとのセッション要約を集計（定点観測の日次トレンド用）。
                    # z_sum には6成分zを累積し、セッション平均zを STAI データセットに残す。
                    if last_face_id is not None:
                        s = session_summaries.setdefault(
                            last_face_id,
                            {"samples": 0, "sum": 0.0, "max": 0.0,
                             "first_t": elapsed, "last_t": elapsed,
                             "z_sum": {f: 0.0 for f in _FEATURES}},
                        )
                        s["samples"] += 1
                        s["sum"] += smoothed_stress
                        s["max"] = max(s["max"], smoothed_stress)
                        s["last_t"] = elapsed
                        for f in _FEATURES:
                            s["z_sum"][f] += z_comp[f]
                    stress_history.append(
                        (
                            elapsed,
                            frame_count,
                            last_face_id,
                            smoothed_stress,
                            dominant,
                            last_emo_neg,
                            last_brow,
                            blink_rate,
                            last_head,
                            last_mouth,
                            last_eye,
                        )
                    )

            # ================= 描画 =================
            if face_present and last_bbox is not None:
                x, y, w, h = last_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if smoothed_probs:
                    dom = max(smoothed_probs, key=smoothed_probs.get)
                    conf = smoothed_probs[dom] * 100.0
                    label = f"{dom} {conf:.0f}%"
                else:
                    label = "..."
                cv2.rectangle(frame, (x, y - 25), (x + max(w, 200), y), (0, 255, 0), -1)
                cv2.putText(
                    frame, label, (x + 3, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
                )
                # 人物タグ（連続性のある識別ID）を顔枠の下に表示
                if last_face_id is not None:
                    id_label = f"ID:{last_face_id}  sim:{last_face_sim:.2f}"
                    cv2.rectangle(
                        frame, (x, y + h), (x + max(w, 170), y + h + 24), (0, 200, 0), -1
                    )
                    cv2.putText(
                        frame, id_label, (x + 3, y + h + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
                    )

            # 較正の進捗 or ストレスゲージ
            if calibrating:
                remaining = max(0.0, CALIB_SECONDS - elapsed)
                cv2.putText(
                    frame,
                    f"CALIBRATING... keep a neutral face ({remaining:.0f}s)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
            else:
                gauge_x, gauge_y, gauge_w, gauge_h = 20, 20, 200, 20
                display_score = smoothed_stress if smoothed_stress is not None else 0.0
                color = stress_color(display_score)
                cv2.rectangle(
                    frame, (gauge_x, gauge_y),
                    (gauge_x + gauge_w, gauge_y + gauge_h), (200, 200, 200), 2,
                )
                fill_w = int(gauge_w * display_score / 100)
                if fill_w > 0:
                    cv2.rectangle(
                        frame, (gauge_x, gauge_y),
                        (gauge_x + fill_w, gauge_y + gauge_h), color, -1,
                    )
                cv2.putText(
                    frame, f"Stress: {display_score:.0f}",
                    (gauge_x, gauge_y + gauge_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )
                # 参考値（小さく表示）
                cv2.putText(
                    frame,
                    f"brow:{last_brow:.2f} blink:{blink_rate:.0f}/min",
                    (gauge_x, gauge_y + gauge_h + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                )
                cv2.putText(
                    frame,
                    f"head:{last_head:.0f}deg/s mouth:{last_mouth:.2f} eye:{last_eye:.2f}",
                    (gauge_x, gauge_y + gauge_h + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                )
                # 移動(並進)速度と頭部ゲート。歩行中は loco が上がり gate が 0 に近づく。
                # 閾値(LOCO_GATE_LOW/HIGH)を調整する際はこの loco の値を目安にする。
                loco_gate = locomotion_gate(last_loco)
                moving = loco_gate < 1.0
                cv2.putText(
                    frame,
                    f"loco:{last_loco:.1f} head_gate:{loco_gate:.2f}"
                    + ("  MOVING" if moving else ""),
                    (gauge_x, gauge_y + gauge_h + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 165, 255) if moving else (200, 200, 200), 1,
                )

            if not face_present:
                cv2.putText(
                    frame, "No face detected", (20, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

            cv2.imshow("Stress Evaluation (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Ctrl+Cで中断されました。")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()  # mediapipe の終了時例外を避けるため明示的に閉じる
        save_stress_report(stress_history)

        # 人物プロファイル（顔埋め込み・平常状態統計）を書き戻し、セッション履歴を追記する。
        summaries = {
            pid: {
                "duration_sec": max(0.0, s["last_t"] - s["first_t"]),
                "samples": s["samples"],
                "mean_stress": s["sum"] / s["samples"] if s["samples"] else 0.0,
                "max_stress": s["max"],
            }
            for pid, s in session_summaries.items()
        }
        if id_tracker is not None:
            store.next_id = id_tracker.next_id
        try:
            store.save(PROFILES_JSON_PATH, gallery=(id_tracker.gallery if id_tracker else {}),
                       session_summaries=summaries, history_path=SESSION_HISTORY_PATH)
        except Exception as e:
            print(f"人物プロファイルの保存に失敗しました: {e}")

        # 終了時のSTAIラベル付け。ここで貯めた (セッション平均z, STAI) のペアを
        # tune_stress.py が学習に使う。モードにより採点済み得点の手入力(collect)か、
        # アプリ内での質問紙実施＋自動採点(survey)かを切り替える。
        if RUN_MODE == "collect":
            try:
                collect_stai_labels(session_summaries)
            except Exception as e:
                print(f"STAIラベルの記録に失敗しました: {e}")
        elif RUN_MODE == "survey":
            try:
                run_stai_survey(session_summaries)
            except Exception as e:
                print(f"STAI 問診の実施に失敗しました: {e}")


def _prompt_stai(label):
    """STAI 得点(20-80)をコンソールから読む。空欄/範囲外/非数値は None を返す。"""
    try:
        raw = input(label).strip()
    except EOFError:
        return None
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        print("  数値ではないためスキップしました。")
        return None
    if not (20.0 <= val <= 80.0):
        print("  STAI は 20〜80 の範囲です。範囲外のためスキップしました。")
        return None
    return val


def collect_stai_labels(session_summaries):
    """セッション終了時に人物ごとの STAI-S/-T を入力させ、セッション平均zと一緒に
    stai_dataset.jsonl へ1レコードずつ追記する（STAI-S が入力された人物のみ保存）。"""
    if not session_summaries:
        print("このセッションでは評価サンプルが無いため、STAIの記録は行いません。")
        return
    print("\n=== STAI ラベル入力（collect モード）===")
    print("各人物について STAI 得点を入力してください（空欄でその人物をスキップ）。")
    saved = 0
    now = datetime.now().isoformat(timespec="seconds")
    for pid, s in session_summaries.items():
        n = s["samples"]
        if n <= 0:
            continue
        print(f"\n[人物 ID={pid}] このセッションの平均ストレス {s['sum'] / n:.1f} / 100")
        stai_s = _prompt_stai("  STAI-S 状態不安 (20-80, 空欄=この人物をスキップ): ")
        if stai_s is None:
            print("  → スキップしました。")
            continue
        stai_t = _prompt_stai("  STAI-T 特性不安 (20-80, 空欄=可): ")
        rec = {
            "date": now,
            "person_id": int(pid) if pid is not None else None,
            "stai_state": stai_s,
            "stai_trait": stai_t,
            "duration_sec": round(max(0.0, s["last_t"] - s["first_t"]), 1),
            "samples": n,
            "mean_stress": round(s["sum"] / n, 2),
            "z": {f: round(s["z_sum"][f] / n, 4) for f in _FEATURES},
        }
        with open(STAI_DATASET_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        saved += 1
    if saved:
        print(f"\nSTAI ラベルを {saved} 件記録しました: {STAI_DATASET_PATH}")
        print("十分たまったら `python tune_stress.py --report` で相関を確認できます。")
    else:
        print("\nSTAI ラベルは記録されませんでした。")


# ============================================================================
# STAI 質問紙（survey モード：アプリ内で実施＋自動採点）
# ============================================================================
# collect モードが「別紙で実施・採点済みの得点」を手入力させるのに対し、survey モードは
# 終了時にアプリ内で 20 項目に回答させ、逆転項目を含めて自動採点して STAI-S/-T を算出する。
# 項目文・逆転項目・選択肢アンカーは stai_items.json（編集可能）に外部化する。
# （STAI_SCALE_MIN/MAX, STAI_ITEMS_PER_SCALE は config.py で定義）


def _default_stai_items():
    """stai_items.json の雛形（辞書）を返す。項目文はプレースホルダで、正式な日本語 STAI に
    差し替えて使う（著作権のある正文はコードに直書きしない）。reverse は 1-based（各尺度20項目内）。

    ※ 逆転項目は STAI のバージョン（原版 / Form Y / 日本語新版など）で異なる。既定は
       Form Y の値を入れてあるが、使用する正式版に合わせて必ず確認・差し替えること。"""
    anchors = [
        "1: 全くあてはまらない",
        "2: いくらかあてはまる",
        "3: よくあてはまる",
        "4: 非常によくあてはまる",
    ]
    return {
        "_note": (
            "items は各尺度20項目のプレースホルダ。正式な日本語STAIの項目文に差し替えてください。"
            "reverse は逆転採点する項目番号(1-based, 20項目内)で、使用する版に合わせて確認・修正すること。"
        ),
        "state": {
            "title": "STAI-S 状態不安（今この瞬間の気持ち）",
            "reverse": [1, 2, 5, 8, 10, 11, 15, 16, 19, 20],
            "anchors": list(anchors),
            "items": [f"（状態不安 項目{i} の文をここに）" for i in range(1, STAI_ITEMS_PER_SCALE + 1)],
        },
        "trait": {
            "title": "STAI-T 特性不安（ふだんの気持ち）",
            "reverse": [1, 6, 7, 10, 13, 16, 19],
            "anchors": list(anchors),
            "items": [f"（特性不安 項目{i} の文をここに）" for i in range(1, STAI_ITEMS_PER_SCALE + 1)],
        },
    }


def _valid_scale_def(scale):
    """1尺度の定義が採点に使える形か（items が20件、reverse が1-20の範囲）を軽く検証する。"""
    if not isinstance(scale, dict):
        return False
    items = scale.get("items")
    if not isinstance(items, list) or len(items) != STAI_ITEMS_PER_SCALE:
        return False
    reverse = scale.get("reverse", [])
    if not isinstance(reverse, list):
        return False
    return all(isinstance(i, int) and 1 <= i <= STAI_ITEMS_PER_SCALE for i in reverse)


def load_stai_items(path=STAI_ITEMS_PATH):
    """stai_items.json を読み込む。無ければ雛形を書き出して返す（load_stress_config と同じ流儀）。
    内容が壊れている/項目数が合わない場合は警告して雛形にフォールバックする。"""
    if not os.path.exists(path):
        items = _default_stai_items()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f"STAI 質問紙の雛形を作成しました: {path}")
            print("→ 正式な日本語 STAI の項目文・逆転項目に差し替えてから使ってください。")
        except Exception as e:
            print(f"STAI 質問紙ファイルの作成に失敗しました（雛形で続行）: {e}")
        return items

    try:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception as e:
        print(f"STAI 質問紙ファイルの読み込みに失敗しました（雛形で続行）: {e}")
        return _default_stai_items()

    if not _valid_scale_def(items.get("state")):
        print("stai_items.json の state 定義が不正（項目数/逆転項目）です。雛形で続行します。")
        return _default_stai_items()
    return items


def score_stai(answers, reverse_indices):
    """STAI の採点本体。各回答(1〜4)を、逆転項目は (5 - 回答) に変換して合計する。
    reverse_indices は 1-based（20項目内）。20項目そろっていれば戻り値は 20〜80。"""
    reverse = set(int(i) for i in reverse_indices)
    total = 0
    for pos, raw in enumerate(answers, start=1):  # pos は 1-based の項目番号
        val = int(raw)
        total += (STAI_SCALE_MAX + STAI_SCALE_MIN - val) if pos in reverse else val
    return total


def _prompt_stai_item(item_no, text, anchors):
    """1 項目を提示して 1〜4 を読む。範囲外/非数値は再入力を促す。空欄/EOF は中断(None)。"""
    hint = " / ".join(anchors)
    print(f"\nQ{item_no}. {text}")
    print(f"    ({hint})")
    while True:
        try:
            raw = input(f"    回答 [{STAI_SCALE_MIN}-{STAI_SCALE_MAX}, 空欄=中断]: ").strip()
        except EOFError:
            return None
        if not raw:
            return None
        try:
            val = int(raw)
        except ValueError:
            print("    数値で入力してください。")
            continue
        if STAI_SCALE_MIN <= val <= STAI_SCALE_MAX:
            return val
        print(f"    {STAI_SCALE_MIN}〜{STAI_SCALE_MAX} で入力してください。")


def administer_stai(scale_key, items_cfg):
    """指定尺度（"state"/"trait"）の20項目を順に提示・回答収集し、逆転採点した合計点を返す。
    途中で中断（空欄/EOF）された場合は None を返す。"""
    scale = items_cfg.get(scale_key)
    if not _valid_scale_def(scale):
        print(f"STAI {scale_key} の定義が不正なため実施できません。")
        return None
    title = scale.get("title", scale_key)
    anchors = scale.get("anchors", [])
    print(f"\n----- {title} -----")
    print(f"各項目に {STAI_SCALE_MIN}〜{STAI_SCALE_MAX} で答えてください（空欄で中断）。")
    answers = []
    for i, text in enumerate(scale["items"], start=1):
        val = _prompt_stai_item(i, text, anchors)
        if val is None:
            print("  → 中断しました。")
            return None
        answers.append(val)
    total = score_stai(answers, scale.get("reverse", []))
    print(f"  → {title} 合計 = {total} 点（20-80）")
    return total


def _prompt_scale_choice():
    """このセッションで実施する尺度を選ぶ。"s"=STAI-S のみ / "st"=S+T。既定は "s"。"""
    print("\n実施する尺度を選んでください:")
    print("  [1] STAI-S 状態不安のみ（20項目）")
    print("  [2] STAI-S + STAI-T（40項目）")
    try:
        raw = input("選択 [1/2, 既定=1]: ").strip()
    except EOFError:
        return "s"
    return "st" if raw == "2" else "s"


def run_stai_survey(session_summaries):
    """survey モードの本体。終了時にアプリ内で STAI 質問紙を実施し、逆転採点して算出した
    STAI-S/-T を、collect と同一 schema で stai_dataset.jsonl へ人物ごとに追記する。"""
    if not session_summaries:
        print("このセッションでは評価サンプルが無いため、STAIの記録は行いません。")
        return

    items = load_stai_items()
    scale = _prompt_scale_choice()

    print("\n=== STAI 問診（survey モード）===")
    print("各人物について質問紙に回答してください（最初の項目を空欄にするとその人物をスキップ）。")
    saved = 0
    now = datetime.now().isoformat(timespec="seconds")
    for pid, s in session_summaries.items():
        n = s["samples"]
        if n <= 0:
            continue
        print(f"\n[人物 ID={pid}] このセッションの平均ストレス {s['sum'] / n:.1f} / 100")
        stai_s = administer_stai("state", items)
        if stai_s is None:
            print("  → この人物をスキップしました。")
            continue
        stai_t = administer_stai("trait", items) if scale == "st" else None
        rec = {
            "date": now,
            "person_id": int(pid) if pid is not None else None,
            "stai_state": float(stai_s),
            "stai_trait": float(stai_t) if stai_t is not None else None,
            "duration_sec": round(max(0.0, s["last_t"] - s["first_t"]), 1),
            "samples": n,
            "mean_stress": round(s["sum"] / n, 2),
            "z": {f: round(s["z_sum"][f] / n, 4) for f in _FEATURES},
        }
        with open(STAI_DATASET_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        saved += 1
    if saved:
        print(f"\nSTAI 得点を自動採点して {saved} 件記録しました: {STAI_DATASET_PATH}")
        print("十分たまったら `python tune_stress.py --report` で相関を確認できます。")
    else:
        print("\nSTAI 得点は記録されませんでした。")


if __name__ == "__main__":
    main()
