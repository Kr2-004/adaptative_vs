#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import csv, time, math


class MultiTrackingLogger(Node):
    def __init__(self):
        super().__init__('multi_tracking_logger')

        self.robot_names = ["puzzlebot1", "puzzlebot2", "puzzlebot3", "puzzlebot4"]

        # CSV (fixed name → overwritten every run)
        self.filename = "tracking_log.csv"
        self.csv = open(self.filename, "w", newline="")
        self.writer = csv.writer(self.csv)

        header = ["t"]
        for i in range(1, 5):
            header += [
                f"x_r{i}", f"y_r{i}",
                f"x_ref{i}", f"y_ref{i}",
                f"e_dist{i}"
            ]
        self.writer.writerow(header)

        self.t0 = time.time()

        # robot and reference poses
        self.robot_pose = {name: None for name in self.robot_names}
        self.ref_pose = {name: None for name in self.robot_names}

        # Subscribers
        for name in self.robot_names:
            self.create_subscription(
                PoseStamped,
                f"/vicon/{name}/{name}/pose",
                lambda msg, n=name: self.robot_cb(msg, n),
                20
            )
            self.create_subscription(
                PoseStamped,
                f"/vs/reference/{name}",
                lambda msg, n=name: self.ref_cb(msg, n),
                20
            )

        # Timer 100 Hz
        self.create_timer(0.01, self.log_row)

        print(f"[LOGGER] Writing to {self.filename}")

    def robot_cb(self, msg, name):
        self.robot_pose[name] = (
            msg.pose.position.x,
            msg.pose.position.y
        )

    def ref_cb(self, msg, name):
        self.ref_pose[name] = (
            msg.pose.position.x,
            msg.pose.position.y
        )

    def log_row(self):
        if not all(self.robot_pose[n] for n in self.robot_names):
            return
        if not all(self.ref_pose[n] for n in self.robot_names):
            return

        t = time.time() - self.t0
        row = [t]

        for name in self.robot_names:
            xr, yr = self.robot_pose[name]
            xref, yref = self.ref_pose[name]
            e_dist = math.sqrt((xr - xref)**2 + (yr - yref)**2)
            row += [xr, yr, xref, yref, e_dist]

        self.writer.writerow(row)

    def close(self):
        self.csv.close()


def main(args=None):
    rclpy.init(args=args)
    node = MultiTrackingLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        print("[LOGGER] Tracking log saved as:", node.filename)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
