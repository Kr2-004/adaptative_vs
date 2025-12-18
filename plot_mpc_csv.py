#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams.update({
    "figure.subplot.left":   0.07,
    "figure.subplot.right":  0.97,
    "figure.subplot.bottom": 0.09,
    "figure.subplot.top":    0.97,
    "figure.subplot.wspace": 0.20,
    "figure.subplot.hspace": 0.20,
})

ROBOTS = ["puzzlebot1", "puzzlebot2", "puzzlebot3", "puzzlebot4"]

COLORS = {
    "puzzlebot1": "#FF0000",
    "puzzlebot2": "#0000FF",
    "puzzlebot3": "#008800",
    "puzzlebot4": "#000000",
}

T_TOTAL = 65.0  # experiment duration [s]

# Main
def main():

    # Load CSV data
    cmd_data = {
        r: pd.read_csv(f"{r}_cmd.csv")
        for r in ROBOTS
    }

    max_len = max(len(cmd_data[r]) for r in ROBOTS)
    t = np.linspace(0.0, T_TOTAL, max_len)

    # Figure 1: Linear velocity
    plt.figure(figsize=(12, 6))

    for r in ROBOTS:
        df = cmd_data[r]
        n = len(df)

        plt.plot(
            t[:n],
            df["v_cmd"].to_numpy(),
            color=COLORS[r],
            linewidth=3,
            label=f"{r} v_cmd"
        )

    v_ref = cmd_data[ROBOTS[0]]["v_ref"].to_numpy()
    plt.plot(
        t[:len(v_ref)],
        v_ref,
        "--",
        color="black",
        linewidth=3,
        label="v_ref"
    )

    plt.xlabel("Time [s]", fontsize=30)
    plt.ylabel("Linear velocity v [m/s]", fontsize=30)
    plt.xlim(0.0, 65.1)
    plt.ylim(0.0, 0.220)
    plt.grid(True)
    plt.legend(loc="upper right",ncol = 5, fontsize=17)
    plt.tick_params(axis="both", labelsize=14)

    # Figure 2: Angular velocity
    plt.figure(figsize=(12, 6))

    for r in ROBOTS:
        df = cmd_data[r]
        n = len(df)

        plt.plot(
            t[:n],
            df["w_cmd"].to_numpy(),
            color=COLORS[r],
            linewidth=3,
            label=f"{r} w_cmd"
        )

    plt.xlabel("Time [s]", fontsize=30)
    plt.ylabel("Angular velocity w [rad/s]", fontsize=30)
    plt.xlim(0.0, 65.1)
    plt.ylim(-1.7, 1.5)
    plt.grid(True)
    plt.legend(loc="upper left", fontsize=17)
    plt.tick_params(axis="both", labelsize=14)

    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
