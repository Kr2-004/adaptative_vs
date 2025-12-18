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

def load_data(robot):
    df = pd.read_csv(f"{robot}_wheels.csv")
    return df["t"].to_numpy(), df["wl"].to_numpy(), df["wr"].to_numpy()


def plot_all_wr():
    plt.figure(figsize=(12, 5))
    plt.xlabel("Time [s]", fontsize=30)
    plt.ylabel("ωR [rad/s]", fontsize=30)
    plt.xlim(0.0, 60.5)
    plt.ylim(-1.6, 5)

    for r in ROBOTS:
        t, wl, wr = load_data(r)
        plt.plot(t, wr, label=r, linewidth=3, color=COLORS[r])

    plt.grid(True)
    plt.legend(loc = "lower right", fontsize = 17)
    plt.tick_params(axis="both", labelsize=14)


def plot_all_wl():
    plt.figure(figsize=(12, 5))
    plt.xlabel("Time [s]", fontsize=30)
    plt.ylabel("ωL [rad/s]", fontsize=30)
    plt.xlim(0.0, 60.5)
    plt.ylim(-1.6, 5)

    for r in ROBOTS:
        t, wl, wr = load_data(r)
        plt.plot(t, wl, label=r, linewidth=3, color=COLORS[r])

    plt.grid(True)
    plt.legend(loc = "lower right", fontsize = 17)
    plt.tick_params(axis="both", labelsize=14)


def main():
    print("📊 Plotting wheel speeds for all robots...")

    plot_all_wr()
    plot_all_wl()

    plt.show()


if __name__ == "__main__":
    main()
