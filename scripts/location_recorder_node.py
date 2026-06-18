#!/usr/bin/env python3

import os
import sys
import termios
import threading
import tty

import rospkg
import rospy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist


LINEAR_SPEED  = 0.15   # m/s
ANGULAR_SPEED = 0.5    # rad/s

KEY_BINDINGS = {
    'w': ( LINEAR_SPEED,  0.0),
    'x': (-LINEAR_SPEED,  0.0),
    'a': (0.0,  ANGULAR_SPEED),
    'd': (0.0, -ANGULAR_SPEED),
    's': (0.0,  0.0),
}

BANNER = """
╔══════════════════════════════════════╗
║      Room Guide — Record Mode        ║
╠══════════════════════════════════════╣
║  w       forward                     ║
║  x       backward                    ║
║  a / d   turn left / right           ║
║  s       stop                        ║
║  r       record current location     ║
║  q       quit                        ║
╚══════════════════════════════════════╝
"""


class LocationRecorder:
    def __init__(self):
        rospy.init_node("location_recorder_node")

        self.yaml_path = rospy.get_param("~locations_path", "")

        if not self.yaml_path:
            rospy.logfatal("Locations path not provided!")
            rospy.signal_shutdown("Missing location path")
            return

        self.locations = self._load_yaml()

        self.current_pose = None
        self._pose_lock = threading.Lock()

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber(
            "/amcl_pose", PoseWithCovarianceStamped, self._pose_cb
        )

        print(BANNER)
        rospy.loginfo("[recorder] Saving to: %s", self.yaml_path)

    def _load_yaml(self):
        if os.path.exists(self.yaml_path):
            with open(self.yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            return data.get("locations") or {}
        return {}

    def _save_yaml(self):
        with open(self.yaml_path, "w") as f:
            yaml.dump({"locations": self.locations}, f, default_flow_style=False)

    def _pose_cb(self, msg: PoseWithCovarianceStamped):
        with self._pose_lock:
            self.current_pose = msg.pose.pose

    def _record(self):
        with self._pose_lock:
            pose = self.current_pose

        if pose is None:
            print("\n[!] No pose received yet — is AMCL running?")
            return

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
        try:
            name = input("\nLocation name: ").strip()
        finally:
            tty.setraw(sys.stdin.fileno())

        if not name:
            print("[!] Empty name, skipping.")
            return

        self.locations[name] = {
            "x":     round(pose.position.x, 3),
            "y":     round(pose.position.y, 3),
            "z":     round(pose.orientation.z, 3),
            "w":     round(pose.orientation.w, 3),
        }
        self._save_yaml()
        print(f"Saved '{name}' \n{self.locations[name]}\n")

    def run(self):
        self._old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            rate = rospy.Rate(10)
            current_lin = 0.0
            current_ang = 0.0

            while not rospy.is_shutdown():
                import select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1).lower()

                    if ch == 'q':
                        break
                    elif ch == 'r':
                        current_lin = 0.0
                        current_ang = 0.0
                        self._record()
                    elif ch in KEY_BINDINGS:
                        current_lin, current_ang = KEY_BINDINGS[ch]

                twist = Twist()
                twist.linear.x  = current_lin
                twist.angular.z = current_ang
                self.cmd_pub.publish(twist)
                rate.sleep()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self.cmd_pub.publish(Twist())
            print("\n[recorder] Stopped.")


if __name__ == "__main__":
    try:
        LocationRecorder().run()
    except rospy.ROSInterruptException:
        pass
