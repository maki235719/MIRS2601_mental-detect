"""ロボット側の送信例。evaluate(frame) を ROS2 ノードの画像コールバックから呼べばよい。
動作確認: python stress_client.py http://<server>:8000/  （Webカメラ映像を送って score を表示）"""

import json
import os
import sys
import time
import urllib.request

import cv2


def evaluate(frame, url, ts=None, token=os.environ.get("STRESS_TOKEN")):
    """BGR フレームを送り {"score": float|None, "face_id": int, "timestamp": float} を返す。
    ts は撮影時刻（epoch秒）。ROS2 なら msg.header.stamp を渡す。省略時は送信直前の時刻。"""
    jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    headers = {"X-Timestamp": repr(ts if ts is not None else time.time())}
    if token:
        headers["X-Token"] = token
    req = urllib.request.Request(url, jpg, headers)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    while ok:
        print(evaluate(frame, sys.argv[1]))
        ok, frame = cap.read()
