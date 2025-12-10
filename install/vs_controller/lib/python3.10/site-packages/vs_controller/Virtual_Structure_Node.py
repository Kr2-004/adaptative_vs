#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
import numpy as np, math, time

from qpsolvers import solve_qp, available_solvers


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi


class VirtualStructureNode(Node):
    def __init__(self):
        super().__init__('virtual_structure_node')

        # Number of robots
        self.N_robots = 4  # Change this to 2, 3, 5, ... as needed

        # --- Slot / robot names: puzzlebot1, puzzlebot2, ... ---
        self.slot_names = [f"puzzlebot{i+1}" for i in range(self.N_robots)]

        # --- Robot states (pose per robot) ---
        # pose: np.array([x, y, theta])
        self.robot_poses = {name: None for name in self.slot_names}

        # Obstacle positions (3 independent obstacles)
        self.obs1 = None
        self.obs2 = None
        self.obs3 = None

        # VS position [x_vs, y_vs]
        self.p_vs = None

        # Timing 
        self.dt = 0.01

        # Nominal motion & limits
        self.v_nom = 0.20
        self.v_min = 0.05
        self.v_max = 0.20
        self.dy_dot_max = 0.10
        self.dy_max = 1.0

        # Safety rad
        self.R_robot = 0.12
        self.r_buf   = 0.25

        # Each obstacle has its own radius
        self.R_obs1 = 0.15
        self.R_obs2 = 0.20
        self.R_obs3 = 0.42

        # Precompute d_safe for each obstacle (center-to-center)
        self.d_safe1 = self.R_robot + self.R_obs1 + self.r_buf
        self.d_safe2 = self.R_robot + self.R_obs2 + self.r_buf
        self.d_safe3 = self.R_robot + self.R_obs3 + self.r_buf

        # If slots are closer than this in lateral distance, we push them apart symmetrically.
        self.d_slot_nom = 0.45  

        # Gain for pairwise repulsion when too close
        self.k_pair = 3.0

        # CBF & decay 
        self.alpha_cbf = 3.0
        self.k_decay   = 0.10

        self.hard_stop = False

        # Formation nominal offsets (symmetric with spacing 0.25) 
        d_lat = 0.7
        center_index = 0.5 * (self.N_robots - 1)  # e.g., N=4 → 1.5
        self.formation_offsets = {
            name: (i - center_index) * d_lat
            for i, name in enumerate(self.slot_names)
        }

        # Slot deformation states 
        self.delta_y = {name: 0.0 for name in self.slot_names}
        self.delta_x = {name: 0.0 for name in self.slot_names}
        self.kx_decay = 0.15
        self.dx_max = 1.0

        # Per-slot heading computation 
        self.prev_slot_positions = {name: None for name in self.slot_names}
        self.slot_headings = {name: 0.0 for name in self.slot_names}

        # Slowdown logic 
        self.spring_thresh = 0.10
        self.k_spring = 2.0

        # Stop point 
        self.x_stop = 3.0

        # ROS I/O 
        self.pub_vref = self.create_publisher(Float32, '/vs/reference_speed', 10)
        self.pub_refs = {
            name: self.create_publisher(PoseStamped, f'/vs/reference/{name}', 10)
            for name in self.slot_names
        }
        self.pub_cbf = {
            name: self.create_publisher(Float32, f'/vs/cbf/slot{idx+1}', 10)
            for idx, name in enumerate(self.slot_names)
        }

        # Robot subscriptions (auto for each puzzlebotN)
        for name in self.slot_names:
            topic = f'/vicon/{name}/{name}/pose'
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, n=name: self.robot_cb(msg, n),
                20
            )

        # Obstacles 
        self.create_subscription(
            PoseStamped, '/vicon/Obstacle/Obstacle/pose',
            self.obs1_cb, 10)
        self.create_subscription(
            PoseStamped, '/vicon/Obstacle2/Obstacle2/pose',
            self.obs2_cb, 10)
        self.create_subscription(
            PoseStamped, '/vicon/Obstacle3/Obstacle3/pose',
            self.obs3_cb, 10)

        # Pick solver
        pref = ['proxqp', 'highs', 'osqp', 'scs', 'clarabel', 'ecos']
        av = set(available_solvers)
        self.qp_solver = next((s for s in pref if s in av), None)
        if self.qp_solver:
            self.get_logger().info(f"Using QP solver: {self.qp_solver}")
        else:
            self.get_logger().warn("No QP solver found.")

        # Timer
        self.timer = self.create_timer(self.dt, self.loop)

    # ROS Callbacks
    def robot_cb(self, msg, name: str):
        x, y = msg.pose.position.x, msg.pose.position.y
        th = 2 * math.atan2(msg.pose.orientation.z, msg.pose.orientation.w)
        self.robot_poses[name] = np.array([x, y, th], float)
        self.init_vs_if_ready()

    # Obstacles callbacks
    def obs1_cb(self, msg):
        self.obs1 = np.array([msg.pose.position.x, msg.pose.position.y], float)

    def obs2_cb(self, msg):
        self.obs2 = np.array([msg.pose.position.x, msg.pose.position.y], float)

    def obs3_cb(self, msg):
        self.obs3 = np.array([msg.pose.position.x, msg.pose.position.y], float)

    # VS Initialization
    def init_vs_if_ready(self):
        if self.p_vs is not None:
            return
        # Require all robots to have a pose
        if not all(self.robot_poses[name] is not None for name in self.slot_names):
            return

        xs = [self.robot_poses[name][0] for name in self.slot_names]
        ys = [self.robot_poses[name][1] for name in self.slot_names]

        x_vs = max(xs)               # VS at the front-most robot in x
        y_vs = float(np.mean(ys))    # VS y at the average of robots
        self.p_vs = np.array([x_vs, y_vs])

        # Initialize delta_x, delta_y for each slot/robot
        for name in self.slot_names:
            p_robot = self.robot_poses[name][:2]
            rel = p_robot - self.p_vs
            self.delta_x[name] = rel[0]
            self.delta_y[name] = rel[1] - self.formation_offsets[name]

        self.get_logger().info("VS initialized for N robots.")

    # QP Solver (multi-obstacle CBF only, N robots)
    def solve_qp(self, v_nom_eff, slot_positions):

        if self.qp_solver is None:
            v = v_nom_eff
            dy_dot = {}
            for s in self.slot_names:
                dy = self.delta_y[s]
                dy_dot[s] = float(np.clip(-self.k_decay * dy,
                                          -self.dy_dot_max, self.dy_dot_max))
            return v, dy_dot

        e_th = np.array([1.0, 0.0])
        R = np.eye(2)

        # Variables: [v, dy_<robot1>, ..., dy_<robotN>, xi_<robot1>, ..., xi_<robotN>]
        idx = {'v': 0}
        k = 1
        # dy indices
        for name in self.slot_names:
            idx[f'dy_{name}'] = k
            k += 1
        # slack indices
        for name in self.slot_names:
            idx[f'xi_{name}'] = k
            k += 1
        nvar = k

        u_nom = np.zeros(nvar)
        u_nom[idx['v']] = v_nom_eff

        P = np.zeros((nvar, nvar))
        # Weight on v
        P[idx['v'], idx['v']] = 10.0
        # Weights on dy_dot
        for name in self.slot_names:
            P[idx[f'dy_{name}'], idx[f'dy_{name}']] = 0.5
        # Weights on slack 
        for name in self.slot_names:
            P[idx[f'xi_{name}'], idx[f'xi_{name}']] = 1e4

        q = -P @ u_nom

        G_rows = []
        h_rows = []

        # Build obstacle list: (position, d_safe)
        obs_list = []
        if self.obs1 is not None: obs_list.append((self.obs1, self.d_safe1))
        if self.obs2 is not None: obs_list.append((self.obs2, self.d_safe2))
        if self.obs3 is not None: obs_list.append((self.obs3, self.d_safe3))

        # Obstacle CBF constraints for ALL obstacles + ALL slots 
        for obs, d_safe in obs_list:
            for name in self.slot_names:

                dy_tot = self.formation_offsets[name] + self.delta_y[name]
                offs = np.array([self.delta_x[name], dy_tot])

                p_slot = slot_positions[name]
                r = p_slot - obs
                h_val = float(r @ r - d_safe**2)
                grad_h = 2.0 * r

                c_v = grad_h @ e_th
                c_dy = grad_h @ (R @ np.array([0, 1]))

                row = np.zeros(nvar)
                row[idx['v']] = c_v
                row[idx[f'dy_{name}']] = c_dy
                row[idx[f'xi_{name}']] = 1.0  # slack for this slot

                G_rows.append(-row)
                h_rows.append(self.alpha_cbf * h_val)

        # Stack constraints
        G = np.array(G_rows) if G_rows else None
        h = np.array(h_rows) if h_rows else None

        lb = -np.inf * np.ones(nvar)
        ub =  np.inf * np.ones(nvar)

        # Bounds on v
        lb[idx['v']], ub[idx['v']] = self.v_min, self.v_max

        # Bounds on dy_dot
        for name in self.slot_names:
            lb[idx[f'dy_{name}']] = -self.dy_dot_max
            ub[idx[f'dy_{name}']] =  self.dy_dot_max

        # Bounds on slack (>= 0)
        for name in self.slot_names:
            lb[idx[f'xi_{name}']] = 0.0
            ub[idx[f'xi_{name}']] = np.inf

        try:
            x = solve_qp(P, q, G, h, lb=lb, ub=ub, solver=self.qp_solver)
            if x is None:
                raise RuntimeError("QP returned None")
        except Exception as e:
            self.get_logger().warn(f"QP failed, fallback used: {e}")
            v = v_nom_eff
            dy_dot = {}
            for s in self.slot_names:
                dy = self.delta_y[s]
                dy_dot[s] = float(np.clip(-self.k_decay * dy,
                                          -self.dy_dot_max, self.dy_dot_max))
            return v, dy_dot

        v = float(x[idx['v']])
        dy_dot = {
            name: float(x[idx[f'dy_{name}']])
            for name in self.slot_names
        }
        return v, dy_dot

    # MAIN LOOP
    def loop(self):
        if self.p_vs is None:
            return
        
        if self.p_vs[0] >= self.x_stop:
            self.hard_stop = True

        dt = self.dt

        # Slowdown using all robots
        lags = []
        for name in self.slot_names:
            pose = self.robot_poses[name]
            if pose is not None:
                lags.append(self.p_vs[0] - pose[0])

        if len(lags) == 0:
            v_nom_eff = self.v_nom
        else:
            d_err = max(lags)
            v_nom_eff = self.v_nom
            if d_err > self.spring_thresh:
                v_nom_eff = max(
                    0.05,
                    self.v_nom - self.k_spring * (d_err - self.spring_thresh)
                )
            v_nom_eff = float(np.clip(v_nom_eff, 0.05, self.v_max))

        # Compute slot positions
        slot_positions = {}
        for name in self.slot_names:
            dy_tot = self.formation_offsets[name] + self.delta_y[name]
            p_loc = np.array([self.delta_x[name], dy_tot])
            slot_positions[name] = self.p_vs + p_loc

        # Solve QP (obstacle CBF only)
        v_vs, dy_dot = self.solve_qp(v_nom_eff, slot_positions)
        if self.hard_stop:
            v_vs = 0.0

        obs_list = []
        if self.obs1 is not None: obs_list.append((self.obs1, self.d_safe1))
        if self.obs2 is not None: obs_list.append((self.obs2, self.d_safe2))
        if self.obs3 is not None: obs_list.append((self.obs3, self.d_safe3))
        for idx, name in enumerate(self.slot_names):

            if len(obs_list) == 0:
                h_min = 1.0
            else:
                h_vals = []
                for obs, d_safe in obs_list:
                    d2 = np.sum((slot_positions[name] - obs)**2)
                    h_vals.append(d2 - d_safe**2)
                h_min = min(h_vals)

            # Publish as Float32
            self.pub_cbf[name].publish(Float32(data=float(h_min)))
        slot_cbf_active = {s: False for s in self.slot_names}
        for obs, d_safe in obs_list:
            for name in self.slot_names:
                d2 = np.sum((slot_positions[name] - obs)**2)
                if d2 - d_safe**2 <= 0.0:
                    slot_cbf_active[name] = True

        dx_dot = {}

        for name in self.slot_names:
            dx = self.delta_x[name]

            if slot_cbf_active[name]:
                # slow decay when CBF active
                dx_dot[name] = -0.3 * self.kx_decay * dx
            else:
                # full decay when safe
                dx_dot[name] = -self.kx_decay * dx

        for name in self.slot_names:
            if not slot_cbf_active[name]:
                # Decay delta_y back to nominal when no CBF is active
                dy = self.delta_y[name]
                dy_dot[name] = float(np.clip(-self.k_decay * dy,
                                             -self.dy_dot_max, self.dy_dot_max))
 
        # If |y_i - y_j| < d_slot_nom, push them apart symmetrically.
        for i in range(len(self.slot_names)):
            name_i = self.slot_names[i]
            y_i = self.formation_offsets[name_i] + self.delta_y[name_i]

            for j in range(i + 1, len(self.slot_names)):
                name_j = self.slot_names[j]
                y_j = self.formation_offsets[name_j] + self.delta_y[name_j]

                dist_y = abs(y_i - y_j)
                if dist_y < self.d_slot_nom:
                    err = self.d_slot_nom - dist_y    
                    if err <= 0.0:
                        continue

                    if y_i <= y_j:
                        sign = 1.0
                    else:
                        sign = -1.0

                    # Symmetric correction on dy_dot
                    dy_dot[name_i] -= self.k_pair * err * sign
                    dy_dot[name_j] += self.k_pair * err * sign

        # Clip dy_dot after applying pairwise separation
        for name in self.slot_names:
            dy_dot[name] = float(np.clip(dy_dot[name],
                                         -self.dy_dot_max, self.dy_dot_max))

        # Move VS forward
        self.p_vs += np.array([1.0, 0.0]) * v_vs * dt

        # Update offsets
        for name in self.slot_names:
            self.delta_x[name] += dx_dot[name] * dt
            self.delta_x[name] = float(np.clip(self.delta_x[name],
                                               -self.dx_max, self.dx_max))

            self.delta_y[name] += dy_dot[name] * dt
            self.delta_y[name] = float(np.clip(self.delta_y[name],
                                               -self.dy_max, self.dy_max))

        # Stop condition
        if self.p_vs[0] >= self.x_stop:
            v_vs = 0.0

        # Publish references
        for name in self.slot_names:
            dy_tot = self.formation_offsets[name] + self.delta_y[name]
            p_loc = np.array([self.delta_x[name], dy_tot])
            p_ref = self.p_vs + p_loc

            prev = self.prev_slot_positions[name]
            speed = 0.0
            th_slot = self.slot_headings[name]

            if prev is not None:
                v_slot = (p_ref - prev) / dt
                speed = np.linalg.norm(v_slot)

            if speed > 1e-4:
                th_raw = math.atan2(v_slot[1], v_slot[0])
                th_old = self.slot_headings[name]

                dy_cur = self.delta_y[name]
                if abs(dy_cur) < 0.02:
                    th_slot = 0.7 * th_old + 0.3 * 0.0
                else:
                    th_slot = 0.8 * th_old + 0.2 * th_raw

                self.slot_headings[name] = th_slot

            self.prev_slot_positions[name] = p_ref.copy()

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "vicon/world"
            msg.pose.position.x = float(p_ref[0])
            msg.pose.position.y = float(p_ref[1])
            msg.pose.position.z = 0.11
            msg.pose.orientation.z = math.sin(self.slot_headings[name] / 2)
            msg.pose.orientation.w = math.cos(self.slot_headings[name] / 2)

            self.pub_refs[name].publish(msg)

        self.pub_vref.publish(Float32(data=float(v_vs)))


def main(args=None):
    rclpy.init(args=args)
    node = VirtualStructureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_vref.publish(Float32(data=0.0))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
