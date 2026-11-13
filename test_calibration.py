"""IDごとのベースライン学習（passive: 中央値/MAD、explicit: Welford較正）の自己チェック。
pytest不要、`python test_calibration.py` で実行。"""
import peroson_detect as pd


def test_passive_insufficient_samples_returns_none():
    store = pd.PersonStore(backend_tag="test")
    store.passive_append(1, "brow", 0.1)
    assert store.passive_baseline_dict(1) is None


def test_passive_baseline_uses_median_and_is_robust_to_outliers():
    store = pd.PersonStore(backend_tag="test")
    n = pd.PASSIVE_MIN_SAMPLES
    for f in pd._FEATURES:
        for _ in range(n):
            store.passive_append(1, f, 0.10)
    bl = store.passive_baseline_dict(1)
    assert bl is not None
    assert abs(bl["brow_mean"] - 0.10) < 1e-9

    # 一時的な緊張フレーム（外れ値）が数個混ざっても中央値はほぼ動かない
    for _ in range(3):
        store.passive_append(1, "brow", 5.0)
    bl2 = store.passive_baseline_dict(1)
    assert abs(bl2["brow_mean"] - 0.10) < 1e-6, bl2


def test_passive_baseline_isolated_per_id():
    store = pd.PersonStore(backend_tag="test")
    n = pd.PASSIVE_MIN_SAMPLES
    for f in pd._FEATURES:
        for _ in range(n):
            store.passive_append(1, f, 0.10)
            store.passive_append(2, f, 0.90)
    bl1 = store.passive_baseline_dict(1)
    bl2 = store.passive_baseline_dict(2)
    assert abs(bl1["brow_mean"] - 0.10) < 1e-9
    assert abs(bl2["brow_mean"] - 0.90) < 1e-9  # 他人の値が混ざっていない


def test_build_baseline_from_samples():
    bl = pd._build_baseline_from_samples(
        emo=[1.0, 2.0, 3.0], brow=[0.1, 0.2, 0.3],
        head=[0.0], mouth=[0.5], eye=[0.5], blink_rate=12.0,
    )
    assert abs(bl["emo_mean"] - 2.0) < 1e-9
    assert abs(bl["brow_mean"] - 0.2) < 1e-9
    assert bl["blink_mean"] == 12.0


def test_ingest_calibration_isolated_per_id():
    store = pd.PersonStore(backend_tag="test")
    samples_a = {"emo": [0.1] * 5, "brow": [0.1] * 5, "head": [0.1] * 5,
                 "mouth": [0.1] * 5, "eye": [0.1] * 5, "blink": 10.0}
    samples_b = {"emo": [0.9] * 5, "brow": [0.9] * 5, "head": [0.9] * 5,
                 "mouth": [0.9] * 5, "eye": [0.9] * 5, "blink": 20.0}
    store.ingest_calibration(1, samples_a)
    store.ingest_calibration(2, samples_b)
    bl1 = store.baseline_dict(1)
    bl2 = store.baseline_dict(2)
    assert abs(bl1["brow_mean"] - 0.1) < 1e-9
    assert abs(bl2["brow_mean"] - 0.9) < 1e-9  # 元バグ（他人の値の流用）の再発防止


def main():
    test_passive_insufficient_samples_returns_none()
    test_passive_baseline_uses_median_and_is_robust_to_outliers()
    test_passive_baseline_isolated_per_id()
    test_build_baseline_from_samples()
    test_ingest_calibration_isolated_per_id()
    print("OK: passive(中央値/MAD)・explicit(Welford)ベースラインのID分離と頑健性を確認しました。")


if __name__ == "__main__":
    main()
