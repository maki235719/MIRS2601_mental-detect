"""
「高ストレスが一定時間以上継続したか」を判定するROS非依存モジュール。

  mode="any"  : 誰か1人でも score>=threshold が hold_sec 連続したら alert
  mode="mean" : stale_sec 以内に観測された全員の平均が threshold 以上で hold_sec 連続したら alert

使い方:
    a = SustainedAlert(threshold=66, hold_sec=10, stale_sec=3, mode="any")
    a.update(now, pid, score)      # サンプルごとに呼ぶ（pid=None も可）
    r = a.evaluate(now)            # {"alert": bool, "max": float, "mean": float, "ids": [pid,...]}

単体チェック: python stress_alert.py
"""


class SustainedAlert:
    def __init__(self, threshold, hold_sec, stale_sec, mode="any"):
        if mode not in ("any", "mean"):
            raise ValueError(f"mode must be 'any' or 'mean': {mode}")
        self.threshold = threshold
        self.hold_sec = hold_sec
        self.stale_sec = stale_sec
        self.mode = mode
        self._last = {}    # pid -> (最終観測時刻, 最新スコア)
        self._since = {}   # pid -> 閾値以上になり始めた時刻（"mean" では None キーを使う）

    def update(self, now, pid, score):
        self._last[pid] = (now, score)

    def evaluate(self, now):
        # 顔ロスト等で stale_sec 超観測されない人物は除外（最後の値で固着しない）
        for pid in [p for p, (t, _) in self._last.items() if now - t > self.stale_sec]:
            del self._last[pid]
            self._since.pop(pid, None)
        scores = {p: s for p, (_, s) in self._last.items()}
        mean = sum(scores.values()) / len(scores) if scores else 0.0
        peak = max(scores.values(), default=0.0)

        if self.mode == "any":
            values = scores
        else:
            values = {None: mean} if scores else {}
        for key in list(self._since):
            if key not in values:
                del self._since[key]
        ids = []
        for key, v in values.items():
            if v >= self.threshold:
                since = self._since.setdefault(key, now)
                if now - since >= self.hold_sec:
                    ids.append(key)
            else:
                self._since.pop(key, None)
        if self.mode == "mean" and ids:
            ids = sorted(p for p in scores if p is not None)  # 場全体: 在席の全員
        return {"alert": bool(ids), "max": peak, "mean": mean, "ids": ids}


def _demo():
    # any: 1人が10秒継続で発報、途中で下回るとリセット
    a = SustainedAlert(66, 10, 3, "any")
    for t in range(0, 10):
        a.update(t, 1, 80); a.update(t, 2, 20)
        assert not a.evaluate(t)["alert"], t
    a.update(10, 1, 80); a.update(10, 2, 20)
    r = a.evaluate(10)
    assert r["alert"] and r["ids"] == [1] and r["max"] == 80, r
    a.update(11, 1, 10)
    assert not a.evaluate(11)["alert"]            # 下回ったら解除
    a.update(12, 1, 80)
    assert not a.evaluate(12)["alert"]            # 継続タイマーは再スタート

    # stale: 観測が途絶えた人物は除外される
    a = SustainedAlert(66, 10, 3, "any")
    a.update(0, 1, 90); a.evaluate(0)
    assert not a.evaluate(20)["alert"] and a.evaluate(20)["max"] == 0.0

    # mean: 1人だけ高くても平均が低ければ発報しない / 平均が高ければ発報
    a = SustainedAlert(66, 5, 3, "mean")
    for t in range(0, 8):
        a.update(t, 1, 100); a.update(t, 2, 0)
        assert not a.evaluate(t)["alert"]
    a = SustainedAlert(66, 5, 3, "mean")
    for t in range(0, 6):
        a.update(t, 1, 80); a.update(t, 2, 70)
        r = a.evaluate(t)
    assert r["alert"] and r["ids"] == [1, 2], r
    print("stress_alert self-check OK")


if __name__ == "__main__":
    _demo()
