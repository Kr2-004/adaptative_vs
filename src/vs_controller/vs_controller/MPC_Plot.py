#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
import csv

ROBOTS = ["puzzlebot1", "puzzlebot2", "puzzlebot3", "puzzlebot4"]

class MPCLogger(Node):
    def __init__(self):
        super().__init__("mpc_logger")

        # Storage: v_ref per robot + cmd log (v_cmd, w_cmd, v_ref)
        self.data = {
            r: {
                "v_ref": 0.0,   # latest reference speed
                "cmd": []       # list of (v_cmd, w_cmd, v_ref)
            }
            for r in ROBOTS
        }

        # -------------------- Subscriptions --------------------
        for r in ROBOTS:

            # VS provides reference SPEED (shared topic)
            self.create_subscription(
                Float32,
                "/vs/reference_speed",
                lambda msg, name=r: self.cb_vref(msg, name),
                10
            )

            # MPC publishes: msg.data = [v_cmd, w_cmd]
            self.create_subscription(
                Float32MultiArray,
                f"/{r}/mpc_cmd",
                lambda msg, name=r: self.cb_cmd(msg, name),
                50
            )

        self.get_logger().info("📡 Logging v_cmd, w_cmd, and v_ref for all robots.")

        # CSV filenames
        self.files = {r: f"{r}_cmd.csv" for r in ROBOTS}

    # -------------------- Callbacks --------------------

    def cb_vref(self, msg, name):
        """Store latest reference speed."""
        self.data[name]["v_ref"] = float(msg.data)

    def cb_cmd(self, msg, name):
        """Store MPC outputs and reference speed."""
        v_cmd = float(msg.data[0])
        w_cmd = float(msg.data[1])
        v_ref = self.data[name]["v_ref"]

        self.data[name]["cmd"].append((v_cmd, w_cmd, v_ref))

    # -------------------- Save to CSV --------------------

    def save_all(self):
        self.get_logger().info("💾 Saving command logs (v_cmd, w_cmd, v_ref)...")

        for r in ROBOTS:
            with open(self.files[r], "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["v_cmd", "w_cmd", "v_ref"])
                w.writerows(self.data[r]["cmd"])

        self.get_logger().info("✅ Saved!")

# -------------------- MAIN --------------------

def main():
    rclpy.init()
    node = MPCLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_all()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("MPC logging finished.")

if __name__ == "__main__":
    main()
