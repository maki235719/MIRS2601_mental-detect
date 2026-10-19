"""
ストレス評価を ROS2 ノードとして動かし、高ストレスの継続を他ノードへ通知する。

publish (std_msgs のみ。独自msg・colcon build は不要):
  /stress/alert      Bool             高ストレスが hold 秒以上継続（transient_local: 後から購読しても最新値が届く）
  /stress/max        Float32          在席者の最大ストレス
  /stress/mean       Float32          在席者の平均ストレス
  /stress/alert_ids  Int32MultiArray  アラート中の人物ID

parameters: alert_threshold / alert_hold_sec / alert_stale_sec / alert_mode("any"|"mean")
起動: source /opt/ros/jazzy/setup.bash && python peroson_detect.py   （config の OUTPUT_MODE が ros2/both）
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, Int32MultiArray

import config
import peroson_detect
from stress_alert import SustainedAlert


class StressAlertNode(Node):
    def __init__(self):
        super().__init__("stress_alert_node")
        self.declare_parameter("alert_threshold", float(config.ALERT_THRESHOLD))
        self.declare_parameter("alert_hold_sec", float(config.ALERT_HOLD_SEC))
        self.declare_parameter("alert_stale_sec", float(config.ALERT_STALE_SEC))
        self.declare_parameter("alert_mode", config.ALERT_MODE)
        p = self.get_parameter
        self.detector = SustainedAlert(
            p("alert_threshold").value,
            p("alert_hold_sec").value,
            p("alert_stale_sec").value,
            p("alert_mode").value,
        )
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_alert = self.create_publisher(Bool, "/stress/alert", latched)
        self.pub_max = self.create_publisher(Float32, "/stress/max", 10)
        self.pub_mean = self.create_publisher(Float32, "/stress/mean", 10)
        self.pub_ids = self.create_publisher(Int32MultiArray, "/stress/alert_ids", latched)
        self._alert = None
        self._publish_alert(False, [])  # 起動直後に「アラート無し」を通知

    def _publish_alert(self, alert, ids):
        self._alert = alert
        self.pub_alert.publish(Bool(data=alert))
        self.pub_ids.publish(Int32MultiArray(data=[i for i in ids if i is not None]))

    def on_sample(self, now, pid, score):
        self.detector.update(now, pid, score)

    def tick(self, now):
        """毎フレーム呼ぶ。顔が居なくなった時の stale 失効もここで拾い、状態変化のみ alert を送る。"""
        rclpy.spin_once(self, timeout_sec=0)
        r = self.detector.evaluate(now)
        self.pub_max.publish(Float32(data=float(r["max"])))
        self.pub_mean.publish(Float32(data=float(r["mean"])))
        if r["alert"] != self._alert:
            self._publish_alert(r["alert"], r["ids"])
            self.get_logger().warn(f"stress alert={r['alert']} ids={r['ids']}")


def run_ros_node(show_video=False):
    import time
    rclpy.init()
    node = StressAlertNode()
    try:
        peroson_detect.main(
            on_sample=node.on_sample,
            should_stop=lambda: (node.tick(time.time()), not rclpy.ok())[1],
            show_video=show_video,
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    peroson_detect.load_stress_config()
    run_ros_node(show_video=False)
