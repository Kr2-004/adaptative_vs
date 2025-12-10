#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROBOTS = ["puzzlebot1", "puzzlebot2", "puzzlebot3", "puzzlebot4"]

COLORS = {
    "puzzlebot1": "tab:blue",
    "puzzlebot2": "tab:orange",
    "puzzlebot3": "tab:green",
    "puzzlebot4": "tab:red",
}

def main():

    # ===============================
    # LOAD ALL CSV FILES
    # ===============================
    cmd_data = {}
    for r in ROBOTS:
        df = pd.read_csv(f"{r}_cmd.csv")
        cmd_data[r] = df

    # ===============================
    # FIGURE 1 — v_cmd for all robots + v_ref
    # ===============================
    plt.figure(figsize=(12, 6))
    plt.title("Linear Velocity Tracking: v_cmd for All Robots vs v_ref")

    # Use the time index for x-axis
    # (all CSVs should have same length; if not, we handle it)
    max_len = max(len(cmd_data[r]) for r in ROBOTS)
    t = np.arange(max_len)

    # Plot v_cmd for each robot
    for r in ROBOTS:
        df = cmd_data[r]
        n = len(df)

        plt.plot(
            t[:n],
            df["v_cmd"].to_numpy(),
            color=COLORS[r],
            linewidth=2,
            label=f"{r} v_cmd"
        )

    # Plot v_ref (use any robot’s v_ref — they are identical)
    v_ref = cmd_data[ROBOTS[0]]["v_ref"].to_numpy()
    plt.plot(
        np.arange(len(v_ref)),
        v_ref,
        "--",
        color="black",
        linewidth=2,
        label="v_ref (reference)"
    )

    plt.xlabel("Time step (0.01 s per step)")
    plt.ylabel("Linear velocity v [m/s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ===============================
    # FIGURE 2 — w_cmd for all robots
    # ===============================
    plt.figure(figsize=(12, 6))
    plt.title("Angular Velocity Commands: w_cmd for All Robots")

    for r in ROBOTS:
        df = cmd_data[r]
        n = len(df)

        plt.plot(
            t[:n],
            df["w_cmd"].to_numpy(),
            color=COLORS[r],
            linewidth=2,
            label=f"{r} w_cmd"
        )

    plt.xlabel("Time step (0.01 s per step)")
    plt.ylabel("Angular velocity w [rad/s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ===============================
    # SHOW BOTH PLOTS
    # ===============================
    plt.show()


if __name__ == "__main__":
    main()
