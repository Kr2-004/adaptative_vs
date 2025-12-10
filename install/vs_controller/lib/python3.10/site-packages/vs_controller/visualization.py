#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math


class RvizVisualizer(Node):
    def __init__(self):
        super().__init__('rviz_visualizer')

        # --- Parameters ---
        self.robot_radius = 0.12

        # --- Individual obstacle radii (EDIT HERE) ---
        self.obstacle1_radius = 0.15
        self.obstacle2_radius = 0.20
        self.obstacle3_radius = 0.42

        # --- Individual safe buffers (EDIT HERE) ---
        self.buffer_radius_1 = 0.10
        self.buffer_radius_2 = 0.10
        self.buffer_radius_3 = 0.10

        # Derived safe distances
        self.d_safe_1 = self.obstacle1_radius + self.buffer_radius_1
        self.d_safe_2 = self.obstacle2_radius + self.buffer_radius_2
        self.d_safe_3 = self.obstacle3_radius + self.buffer_radius_3

        self.vs_z = 0.11

        # --- State ---
        self.robot1_pose = None
        self.robot2_pose = None
        self.robot3_pose = None
        self.robot4_pose = None
        self.obstacle1_pose = None
        self.obstacle2_pose = None
        self.obstacle3_pose = None

        self.vs_slots = {'puzzlebot1': None, 'puzzlebot2': None,'puzzlebot3': None, 'puzzlebot4': None}
        self.vs_center = None

        # Publisher
        self.pub = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)

        # Subscriptions (robots)
        self.create_subscription(PoseStamped, '/vicon/puzzlebot1/puzzlebot1/pose', self.cb_robot1, 10)
        self.create_subscription(PoseStamped, '/vicon/puzzlebot2/puzzlebot2/pose', self.cb_robot2, 10)
        self.create_subscription(PoseStamped, '/vicon/puzzlebot3/puzzlebot3/pose', self.cb_robot3, 10)
        self.create_subscription(PoseStamped, '/vicon/puzzlebot4/puzzlebot4/pose', self.cb_robot4, 10)
        self.create_subscription(PoseStamped, '/vicon/Obstacle/Obstacle/pose', self.cb_obstacle1, 10)
        self.create_subscription(PoseStamped, '/vicon/Obstacle2/Obstacle2/pose', self.cb_obstacle2, 10)
        self.create_subscription(PoseStamped, '/vicon/Obstacle3/Obstacle3/pose', self.cb_obstacle3, 10)
        self.create_subscription(PoseStamped, '/vs/reference/puzzlebot1', self.cb_vs_p1, 10)
        self.create_subscription(PoseStamped, '/vs/reference/puzzlebot2', self.cb_vs_p2, 10)
        self.create_subscription(PoseStamped, '/vs/reference/puzzlebot3', self.cb_vs_p3, 10)
        self.create_subscription(PoseStamped, '/vs/reference/puzzlebot4', self.cb_vs_p4, 10)

        # --- Build markers ONCE ---
        self.marker_array = MarkerArray()
        self.markers = {}  # dict

        self.build_markers_once()

        # Timer (30 Hz)
        self.create_timer(0.033, self.update_markers)

# Markers
    def build_markers_once(self):

        def add_marker(marker_id, mtype, scale_xyz, color_rgba):
            m = Marker()
            m.header.frame_id = "vicon/world"
            m.ns = "viz"
            m.id = marker_id
            m.type = mtype
            m.action = Marker.ADD
            m.scale.x, m.scale.y, m.scale.z = scale_xyz
            m.color.r, m.color.g, m.color.b, m.color.a = color_rgba
            m.pose.orientation.w = 1.0
            self.marker_array.markers.append(m)
            self.markers[marker_id] = m

        add_marker(0, Marker.SPHERE,
                   (self.obstacle1_radius*2, self.obstacle1_radius*2, 0.01),
                   (1.0, 0.0, 0.0, 1.0)) # Obstacle 1

        add_marker(1, Marker.CYLINDER,
                   (self.d_safe_1*2, self.d_safe_1*2, 0.01),
                   (1.0, 0.3, 0.3, 0.3))
        add_marker(2, Marker.SPHERE,
                   (self.obstacle2_radius*2, self.obstacle2_radius*2, 0.01),
                   (1.0, 0.0, 0.0, 1.0)) # Obstacle 2

        add_marker(3, Marker.CYLINDER,
                   (self.d_safe_2*2, self.d_safe_2*2, 0.01),
                   (1.0, 0.3, 0.3, 0.3))

        add_marker(4, Marker.SPHERE,
                   (self.obstacle3_radius*2, self.obstacle3_radius*2, 0.01),
                   (1.0, 0.0, 0.0, 1.0)) # Obstacle 3

        add_marker(5, Marker.CYLINDER,
                   (self.d_safe_3*2, self.d_safe_3*2, 0.01),
                   (1.0, 0.3, 0.3, 0.3))

        add_marker(6, Marker.SPHERE,
                   (self.robot_radius*2, self.robot_radius*2, 0.01),
                   (0.0, 0.8, 1.0, 1.0))  # robot1

        add_marker(7, Marker.SPHERE,
                   (self.robot_radius*2, self.robot_radius*2, 0.01),
                   (1.0, 0.8, 0.0, 1.0))  # robot2
        add_marker(8, Marker.SPHERE,
                   (self.robot_radius*2, self.robot_radius*2, 0.01),
                   (0.0, 0.8, 1.0, 1.0))  # robot3

        add_marker(9, Marker.SPHERE,
                   (self.robot_radius*2, self.robot_radius*2, 0.01),
                   (1.0, 0.8, 0.0, 1.0))  # robot4
        
        add_marker(10, Marker.SPHERE,
                   (0.11, 0.11, 0.11),
                   (0.0, 1.0, 1.0, 1.0))  # center

        add_marker(11, Marker.SPHERE,
                   (0.09, 0.09, 0.09),
                   (1.0, 0.9, 0.0, 1.0))  # slot 1

        add_marker(12, Marker.SPHERE,
                   (0.09, 0.09, 0.09),
                   (1.0, 0.9, 0.0, 1.0))  # slot 2

        add_marker(13, Marker.SPHERE,
                   (0.09, 0.09, 0.09),
                   (1.0, 0.9, 0.0, 1.0))  # slot 3
        
        add_marker(14, Marker.SPHERE,
                   (0.09, 0.09, 0.09),
                   (1.0, 0.9, 0.0, 1.0))  # slot 4

    def cb_robot1(self, msg):
        self.robot1_pose = np.array([msg.pose.position.x, msg.pose.position.y, 0.11])

    def cb_robot2(self, msg):
        self.robot2_pose = np.array([msg.pose.position.x, msg.pose.position.y, 0.11])

    def cb_robot3(self, msg):
        self.robot3_pose = np.array([msg.pose.position.x, msg.pose.position.y, 0.11])

    def cb_robot4(self, msg):
        self.robot4_pose = np.array([msg.pose.position.x, msg.pose.position.y, 0.11])

    def cb_obstacle1(self, msg):
        self.obstacle1_pose = np.array([msg.pose.position.x, msg.pose.position.y, 0.11])

    def cb_obstacle2(self, msg):
        self.obstacle2_pose = np.array([msg.pose.position.x, msg.pose.position.y, 0.11])

    def cb_obstacle3(self, msg):
        self.obstacle3_pose = np.array([msg.pose.position.x, msg.pose.position.y, 0.11])

    def cb_vs_p1(self, msg):
        self.vs_z = msg.pose.position.z
        self.vs_slots["puzzlebot1"] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def cb_vs_p2(self, msg):
        self.vs_z = msg.pose.position.z
        self.vs_slots["puzzlebot2"] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def cb_vs_p3(self, msg):
        self.vs_z = msg.pose.position.z
        self.vs_slots["puzzlebot3"] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def cb_vs_p4(self, msg):
        self.vs_z = msg.pose.position.z
        self.vs_slots["puzzlebot4"] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    # UPDATE MARKERS
    def update_markers(self):

        # Helper function for obstacles
        def update_obstacle(obs_pos, sphere_id, cyl_id):
            if obs_pos is not None:
                m = self.markers[sphere_id]
                m.pose.position.x, m.pose.position.y, m.pose.position.z = obs_pos

                m2 = self.markers[cyl_id]
                m2.pose.position.x, m2.pose.position.y, m2.pose.position.z = obs_pos

        update_obstacle(self.obstacle1_pose, 0, 1)
        update_obstacle(self.obstacle2_pose, 2, 3)
        update_obstacle(self.obstacle3_pose, 4, 5)

        # Robots
        if self.robot1_pose is not None:
            m = self.markers[6]
            m.pose.position.x, m.pose.position.y, m.pose.position.z = self.robot1_pose

        if self.robot2_pose is not None:
            m = self.markers[7]
            m.pose.position.x, m.pose.position.y, m.pose.position.z = self.robot2_pose

        if self.robot3_pose is not None:
            m = self.markers[8]
            m.pose.position.x, m.pose.position.y, m.pose.position.z = self.robot3_pose

        if self.robot4_pose is not None:
            m = self.markers[9]
            m.pose.position.x, m.pose.position.y, m.pose.position.z = self.robot4_pose

        # VS Center + Slots
        if all(v is not None for v in self.vs_slots.values()):
            p1 = self.vs_slots["puzzlebot1"]
            p2 = self.vs_slots["puzzlebot2"]
            p3 = self.vs_slots["puzzlebot3"]
            p4 = self.vs_slots["puzzlebot4"]

            center = np.array([(p1[0] + p2[0] + p3[0] + p4[0]) * 0.25,
                               (p1[1] + p2[1] + p3[1] + p4[1]) * 0.25,
                               self.vs_z])
            self.vs_center = center

            # Center
            m = self.markers[10]
            m.pose.position.x, m.pose.position.y, m.pose.position.z = center

            # Slot 1
            m = self.markers[11]
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p1

            # Slot 2
            m = self.markers[12]
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p2

            # Slot 3
            m = self.markers[13]
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p3
    
            # Slot 4
            m = self.markers[14]
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p4

        # Publish whole array
        now = self.get_clock().now().to_msg()
        for m in self.marker_array.markers:
            m.header.stamp = now

        self.pub.publish(self.marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = RvizVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
