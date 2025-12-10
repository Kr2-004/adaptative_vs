#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import time
import csv

ROBOTS = ["puzzlebot1", "puzzlebot2", "puzzlebot3", "puzzlebot4"]

class WheelLogger(Node):
    def __init__(self):
        super().__init__("wheel_logger")

        # Store (t, wl, wr)
        self.data = {
            r: {
                "wl": 0.0,
                "wr": 0.0,
                "log": []
            }
            for r in ROBOTS
        }

        self.start_time = time.time()

        # -------------------- Subscriptions --------------------
        for r in ROBOTS:

            # Left wheel
            self.create_subscription(
                Float32,
                f"/{r}/VelocitySetL",
                lambda msg, name=r: self.cb_left(msg, name),
                20
            )

            # Right wheel
            self.create_subscription(
                Float32,
                f"/{r}/VelocitySetR",
                lambda msg, name=r: self.cb_right(msg, name),
                20
            )

        self.get_logger().info("📡 Logging WL and WR for all robots.")

        # Filenames
        self.files = {
            r: f"{r}_wheels.csv" for r in ROBOTS
        }

    # -------------------- Callbacks --------------------

    def cb_left(self, msg, name):
        self.data[name]["wl"] = float(msg.data)
        self._store(name)

    def cb_right(self, msg, name):
        self.data[name]["wr"] = float(msg.data)
        self._store(name)

    # Store after any wheel update
    def _store(self, name):
        t = time.time() - self.start_time
        wl = self.data[name]["wl"]
        wr = self.data[name]["wr"]
        self.data[name]["log"].append((t, wl, wr))

    # -------------------- SAVE --------------------

    def save_all(self):
        self.get_logger().info("💾 Saving wheel logs...")

        for r in ROBOTS:
            with open(self.files[r], "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "wl", "wr"])
                w.writerows(self.data[r]["log"])

        self.get_logger().info("✅ Wheel logs saved!")

# -------------------- MAIN --------------------

def main():
    rclpy.init()
    node = WheelLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_all()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("Wheel logging finished.")

if __name__ == "__main__":
    main()
