"""
ストレス評価サーバー。ロボットから JPEG フレームを HTTP POST で受け取り、
既存の peroson_detect.main() にそのまま流してストレス値を JSON で返す。

  POST /  body=JPEG  (ヘッダ X-Timestamp=送信時刻のepoch秒, 任意で X-Token)
          →  {"score": 42.3, "face_id": 1, "timestamp": <送った時刻をそのまま返す>}
  alert は高ストレス継続中か（ROS2の /stress/alert 相当）。GET / はブラウザ用テストページ。
  score は顔なし/ベースライン学習中は null。503=モデル読み込み中。

起動: STRESS_TOKEN=secret PORT=8000 python stress_server.py
ストリームは1本前提（ベースライン・平滑化・まばたき等の状態が1つのため）。10fps 前後で送ること。
"""

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import config
import peroson_detect
from stress_alert import SustainedAlert

TOKEN = os.environ.get("STRESS_TOKEN")  # 未設定なら認証なし（LAN内限定で使うこと）
frames = queue.Queue(maxsize=1)
lock = threading.Lock()      # 1リクエストずつ処理
done = threading.Event()     # main() が1フレーム処理し終えた合図
ready = threading.Event()    # モデル読み込み完了（main() が最初に read した）
result = {}
alert = SustainedAlert(config.ALERT_THRESHOLD, config.ALERT_HOLD_SEC, config.ALERT_STALE_SEC, config.ALERT_MODE)
TEST_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stress_test.html")


class QueueCap:
    """cv2.VideoCapture の代役。受信フレームを main() に渡す。"""

    ts = None  # 直近フレームの送信時刻（ロボットの epoch 秒）

    def isOpened(self):
        return True

    def now(self):
        return self.ts if self.ts is not None else time.time()

    def read(self):
        ready.set()
        while True:  # timeout 付きにして Ctrl+C を効かせる
            try:
                self.ts, frame = frames.get(timeout=0.5)
                return True, frame
            except queue.Empty:
                pass

    def release(self):
        pass


def on_sample(now, pid, score):
    result.update(face_id=pid, score=round(float(score), 1))
    alert.update(now, pid, float(score))


def should_stop():  # main() は毎フレーム末にここを呼ぶ
    done.set()
    return False


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # 実ロボット無しで確認するテストページ（ブラウザのWebカメラを送る）
        with open(TEST_PAGE, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if TOKEN and self.headers.get("X-Token") != TOKEN:
            return self._send(401, {"error": "unauthorized"})
        if not ready.is_set():
            return self._send(503, {"error": "loading"})
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return self._send(400, {"error": "bad image"})
        try:
            ts = float(self.headers["X-Timestamp"])
        except (TypeError, ValueError):
            ts = time.time()  # 未送信/不正ならサーバー受信時刻
        with lock:
            result.clear()
            done.clear()
            frames.put((ts, frame))
            if not done.wait(10):
                return self._send(504, {"error": "timeout"})
            self._send(200, {"score": None, "timestamp": ts, **result,
                             "alert": alert.evaluate(ts)["alert"]})

    def log_message(self, *args):  # 毎フレームのアクセスログを抑止
        pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", 8000))), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    # mediapipe 等はメインスレッドで回す
    peroson_detect.main(on_sample=on_sample, should_stop=should_stop, show_video=False, cap=QueueCap())
