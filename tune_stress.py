"""
STAI 連動チューナ（オフライン）

peroson_detect.py を collect モードで走らせて貯めた stai_dataset.jsonl
（1セッション = セッション平均zスコア6成分 ＋ STAI-S/-T）を読み、

  --report : 現在の stress_config.json のストレス推定が STAI-S とどれだけ
             連動しているか（ピアソン相関 r / RMSE / MAE）を表示する。config は変更しない＝精度評価。
  --apply  : STAI-S を目標に、6成分の重み・ゲイン・基準を最適化して
             stress_config.json に書き戻す（旧 config は .bak に退避）。

scipy 非依存（numpy のみ）。少数サンプルでも落ちないよう、データ量に応じて
OLS → リッジ → スケールのみ調整、と自動でフォールバックする。

使い方:
    python tune_stress.py --report
    python tune_stress.py --apply
    python tune_stress.py --apply --no-trait   # STAI-T を共変量に使わない
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np

from config import STRESS_CONFIG_PATH as CONFIG_PATH
from config import STAI_DATASET_PATH as DATASET_PATH

# 成分キー（データセット/z側） → config の重みキー（履歴上 emo↔emotion だけ名前が違う）
FEATURES = ["emo", "brow", "blink", "head", "mouth", "eye"]
WEIGHT_KEY = {
    "emo": "emotion", "brow": "brow", "blink": "blink",
    "head": "head", "mouth": "mouth", "eye": "eye",
}

# STAI(20-80) 予測を 0-100 ゲージへ写すための線形リスケール（表示スケール維持）。
STAI_MIN, STAI_MAX = 20.0, 80.0
_SCALE = 100.0 / (STAI_MAX - STAI_MIN)   # 100/60


# ----------------------------------------------------------------------------
# 入出力
# ----------------------------------------------------------------------------

def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"設定ファイルが見つかりません: {path}\n"
            "先に peroson_detect.py を一度起動すると雛形が生成されます。"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(path=DATASET_PATH):
    """stai_dataset.jsonl から使用可能なレコードを読む。
    使用可能 = STAI-S があり、6成分zが揃っているもの。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"STAIデータセットが見つかりません: {path}\n"
            "stress_config.json の mode を collect にして peroson_detect.py を走らせ、"
            "終了時に STAI を入力してレコードを貯めてください。"
        )
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("stai_state") is None:
                continue
            z = r.get("z") or {}
            if not all(k in z for k in FEATURES):
                continue
            recs.append(r)
    return recs


def dataset_matrices(recs):
    """レコード列 → (Z: N×6, y_state: N, y_trait: N or None, pids: list)"""
    Z = np.array([[float(r["z"][f]) for f in FEATURES] for r in recs], dtype=np.float64)
    y_state = np.array([float(r["stai_state"]) for r in recs], dtype=np.float64)
    traits = [r.get("stai_trait") for r in recs]
    y_trait = (
        np.array([float(t) for t in traits], dtype=np.float64)
        if all(t is not None for t in traits) else None
    )
    pids = [r.get("person_id") for r in recs]
    return Z, y_state, y_trait, pids


# ----------------------------------------------------------------------------
# 統計・フィット
# ----------------------------------------------------------------------------

def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(np.abs(a - b)))


def fit_linear(X, y, ridge=0.0):
    """y ≈ a + X·beta を最小二乗（任意でリッジ）で解く。切片は正則化しない。
    戻り値: (a, beta[k])。"""
    n, k = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    if ridge > 0.0:
        R = np.eye(k + 1) * ridge
        R[0, 0] = 0.0
        beta_full = np.linalg.solve(A.T @ A + R, A.T @ y)
    else:
        beta_full, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(beta_full[0]), beta_full[1:]


def composite_from_config(Z, cfg):
    """現在の config の重みで各レコードの合成量 Σ w_i z_i を返す（base/gainは相関に無関係）。"""
    w = np.array([float(cfg["weights"][WEIGHT_KEY[f]]) for f in FEATURES])
    return Z @ w


def predicted_score_from_config(Z, cfg):
    """現在の config での 0-100 予測スコア（クリップ込み）。"""
    comp = composite_from_config(Z, cfg)
    score = float(cfg["base_level"]) + float(cfg["z_gain"]) * comp
    return np.clip(score, 0.0, 100.0)


# ----------------------------------------------------------------------------
# レポート（精度評価）
# ----------------------------------------------------------------------------

def report(recs, cfg):
    Z, y_state, y_trait, pids = dataset_matrices(recs)
    n = len(recs)
    pred = predicted_score_from_config(Z, cfg)
    stored = np.array([float(r.get("mean_stress", np.nan)) for r in recs])

    print("=" * 60)
    print(f"STAIデータセット: {n} レコード（人物 {len(set(pids))} 名）")
    print(f"STAI-S 範囲 {y_state.min():.0f}〜{y_state.max():.0f}  平均 {y_state.mean():.1f}")
    print("-" * 60)
    r_pred = pearson(pred, y_state)
    print("現在の config による推定 vs STAI-S:")
    print(f"  ピアソン相関 r = {r_pred:.3f}"
          + ("  （相関の評価には最低3〜5セッション欲しい）" if n < 5 else ""))
    print(f"  RMSE = {rmse(pred, y_state):.1f}   MAE = {mae(pred, y_state):.1f}（0-100スケール上）")
    if np.all(np.isfinite(stored)):
        print(f"  参考: 記録済み平均ストレス vs STAI-S の r = {pearson(stored, y_state):.3f}")
    print("-" * 60)
    print("成分ごとの STAI-S との単相関（どの特徴が効いているかの目安）:")
    for j, f in enumerate(FEATURES):
        print(f"  {f:6s}: r = {pearson(Z[:, j], y_state):+.3f}")
    print("=" * 60)
    if n < 3:
        print("※ サンプルが少なすぎます。--apply は不安定なので、まず数セッション貯めてください。")


# ----------------------------------------------------------------------------
# チューニング（--apply）
# ----------------------------------------------------------------------------

def _weight_vector(cfg):
    """config の重みを FEATURES 順の numpy ベクトルにする。"""
    return np.array([float(cfg["weights"][WEIGHT_KEY[f]]) for f in FEATURES])


def _center_scale(composite, y_state):
    """合成量→STAI-S を1次回帰し、STAI(20-80)を0-100へ写す base_level / z_gain を返す。
    重みの“向き”とは独立に絶対スケール/オフセットだけをデータへ合わせる（ゲージ飽和を防ぐ）。"""
    p, q = fit_linear(composite.reshape(-1, 1), y_state)
    q = float(q[0])
    base = (p - STAI_MIN) * _SCALE
    gain = q * _SCALE
    return base, gain


def _fit_weight_direction(Z, y_state, y_trait, pids, use_trait):
    """STAI-S を目標に6成分zの回帰係数（＝相関最大化方向）を求め、正にクリップして返す。
    データ量に応じて OLS / リッジを選び、複数被験者かつ十分量なら STAI-T を共変量にする。
    戻り値: (係数c[6], 手法名, 使ったtraitフラグ)。"""
    n = len(Z)
    include_trait = use_trait and (y_trait is not None) and (len(set(pids)) >= 2) and (n >= 8)
    X = Z if not include_trait else np.hstack([Z, y_trait.reshape(-1, 1)])
    if n >= 8:
        ridge, method = 0.0, "OLS（十分なサンプル）"
    else:
        ridge, method = 1.0, "リッジ回帰（サンプル少・正則化）"
    _, beta_full = fit_linear(X, y_state, ridge=ridge)
    beta = beta_full[:len(FEATURES)]
    neg = [FEATURES[i] for i in range(len(FEATURES)) if beta[i] < 0]
    c = np.clip(beta, 0.0, None)
    if include_trait:
        method += "＋STAI-T共変量"
    return c, method, neg


def tune(recs, cfg, use_trait=True):
    """STAI-S を目標に重み・ゲイン・基準を最適化した新しい config 辞書を返す。
    手順: (1)回帰で6成分の“重みの向き”を決める → (2)その合成量を STAI へ1次回帰して
    0-100 の絶対スケール(base/gain)を別途合わせる。(2)を分けることで STAI-T 共変量を
    使ってもゲージが飽和しない。サンプル僅少時は重みを据え置きスケールのみ合わせる。"""
    Z, y_state, y_trait, pids = dataset_matrices(recs)
    n = len(recs)
    if n < 3:
        raise SystemExit("サンプルが3未満です。--apply には最低3セッション必要です。")
    new_cfg = json.loads(json.dumps(cfg))  # deep copy

    if n >= 4:
        c, method, neg = _fit_weight_direction(Z, y_state, y_trait, pids, use_trait)
        if neg:
            print(f"注意: STAI と逆相関だった成分を0に丸めました: {', '.join(neg)}")
        if c.sum() <= 1e-9:
            print("有効な正の重みが得られなかったため、重みは据え置きにします。")
            method = "スケールのみ調整（回帰が退化）"
            w = _weight_vector(cfg)
        else:
            w = c / c.sum()
            new_cfg["weights"] = {WEIGHT_KEY[FEATURES[i]]: float(w[i]) for i in range(len(FEATURES))}
    else:
        method = "スケールのみ調整（サンプル僅少・重みは据え置き）"
        w = _weight_vector(cfg)

    # 絶対スケール/オフセットはいずれの手法でも合成量→STAI の1次回帰で決める
    composite = Z @ w
    base, gain = _center_scale(composite, y_state)
    if gain <= 0:
        print("注意: 合成量が STAI と正相関しませんでした。重みを見直してください（相関を確認）。")
    new_cfg["z_gain"] = round(float(gain), 4)
    new_cfg["base_level"] = round(float(base), 4)

    # フィット後の相関を（在サンプル＋LOOで）評価して表示
    pred_new = predicted_score_from_config(Z, new_cfg)
    r_in = pearson(pred_new, y_state)
    r_loo = leave_one_out_r(recs, use_trait=use_trait)

    print("-" * 60)
    print(f"手法: {method}")
    print("新しい重み: " + ", ".join(
        f"{k}={new_cfg['weights'][k]:.3f}" for k in new_cfg["weights"]))
    print(f"z_gain={new_cfg['z_gain']}  base_level={new_cfg['base_level']}")
    print(f"在サンプル相関 r = {r_in:.3f}"
          + (f"   Leave-One-Out r = {r_loo:.3f}" if r_loo == r_loo else ""))
    print("-" * 60)
    return new_cfg


def leave_one_out_r(recs, use_trait=True):
    """1件ずつ抜いて学習→抜いた1件を予測、の予測列と実測の相関（過適合の目安）。
    サンプルが少なすぎる/退化する場合は nan。"""
    n = len(recs)
    if n < 5:
        return float("nan")
    preds, actuals = [], []
    for i in range(n):
        train = recs[:i] + recs[i + 1:]
        try:
            cfg_i = tune_silent(train, use_trait=use_trait)
        except Exception:
            return float("nan")
        Zi, ys, _, _ = dataset_matrices([recs[i]])
        preds.append(float(predicted_score_from_config(Zi, cfg_i)[0]))
        actuals.append(float(ys[0]))
    return pearson(preds, actuals)


def tune_silent(recs, use_trait=True):
    """LOO 用に出力せずフィットだけ行う簡易版（tune のコアと同じ手順）。"""
    Z, y_state, y_trait, pids = dataset_matrices(recs)
    n = len(recs)
    cfg = {"weights": {WEIGHT_KEY[f]: 1.0 / len(FEATURES) for f in FEATURES},
           "z_gain": 0.0, "base_level": 0.0}
    if n >= 4:
        c, _, _ = _fit_weight_direction(Z, y_state, y_trait, pids, use_trait)
        if c.sum() > 1e-9:
            w = c / c.sum()
            cfg["weights"] = {WEIGHT_KEY[FEATURES[i]]: float(w[i]) for i in range(len(FEATURES))}
    w = _weight_vector(cfg)
    base, gain = _center_scale(Z @ w, y_state)
    cfg["z_gain"] = gain
    cfg["base_level"] = base
    return cfg


def save_config(new_cfg, path=CONFIG_PATH):
    if os.path.exists(path):
        bak = path + ".bak"
        shutil.copyfile(path, bak)
        print(f"旧 config をバックアップしました: {bak}")
    new_cfg["_tuned_at"] = datetime.now().isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, ensure_ascii=False, indent=2)
    print(f"チューニング結果を書き戻しました: {path}")
    print("mode を run にして peroson_detect.py を起動すると調整済みで動きます。")


# ----------------------------------------------------------------------------
# エントリポイント
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="STAI 連動チューナ")
    ap.add_argument("--report", action="store_true",
                    help="現在の config の STAI 連動度（相関）を表示（変更しない）")
    ap.add_argument("--apply", action="store_true",
                    help="STAI-S に連動するよう重み等を最適化し config に書き戻す")
    ap.add_argument("--no-trait", action="store_true",
                    help="STAI-T を共変量に使わない")
    args = ap.parse_args()

    if not (args.report or args.apply):
        args.report = True  # 既定はレポート

    cfg = load_config()
    recs = load_dataset()
    if not recs:
        raise SystemExit("使用可能なレコードがありません（STAI-S 入力済みのセッションが必要です）。")

    report(recs, cfg)

    if args.apply:
        print("\n[--apply] STAI-S 連動チューニングを実行します...")
        new_cfg = tune(recs, cfg, use_trait=not args.no_trait)
        new_cfg["mode"] = cfg.get("mode", "run")  # モードは維持
        save_config(new_cfg)


if __name__ == "__main__":
    main()
