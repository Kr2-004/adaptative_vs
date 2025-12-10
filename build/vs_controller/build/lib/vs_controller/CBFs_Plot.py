#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import csv
import time
from datetime import datetime


class CBFLogger(Node):
    def __init__(self):
        super().__init__('cbf_logger')

        # Timestamped CSV file
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = "cbf_log.csv"   # always same filename, always overwritten

        self.csv = open(self.filename, "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow(["t", "slot1", "slot2", "slot3", "slot4"])

        self.last = {"slot1": None, "slot2": None, "slot3": None, "slot4": None}
        self.t0 = time.time()

        # Subscribe to 4 topics
        for i in range(1, 5):
            topic = f"/vs/cbf/slot{i}"
            slot_name = f"slot{i}"
            self.create_subscription(
                Float32, topic,
                lambda msg, name=slot_name: self.cb(msg, name), 10
            )

        # Timer writes CSV at 100Hz
        self.create_timer(0.01, self.flush_row)

        self.get_logger().info(f"Logging CBF values to: {self.filename}")

    def cb(self, msg, name):
        self.last[name] = msg.data

    def flush_row(self):
        if any(v is None for v in self.last.values()):
            return

        t = time.time() - self.t0
        row = [t, self.last["slot1"], self.last["slot2"],
               self.last["slot3"], self.last["slot4"]]
        self.writer.writerow(row)


def main(args=None):
    rclpy.init(args=args)
    node = CBFLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.csv.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
