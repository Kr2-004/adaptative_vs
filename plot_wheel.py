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

def load_data(robot):
    df = pd.read_csv(f"{robot}_wheels.csv")
    return df["t"].to_numpy(), df["wl"].to_numpy(), df["wr"].to_numpy()


def plot_all_wr():
    plt.figure(figsize=(12, 5))
    plt.title("Right Wheel Speed ωR for All Robots")
    plt.xlabel("Time [s]")
    plt.ylabel("ωR [rad/s]")

    for r in ROBOTS:
        t, wl, wr = load_data(r)
        plt.plot(t, wr, label=r, linewidth=2, color=COLORS[r])

    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_all_wl():
    plt.figure(figsize=(12, 5))
    plt.title("Left Wheel Speed ωL for All Robots")
    plt.xlabel("Time [s]")
    plt.ylabel("ωL [rad/s]")

    for r in ROBOTS:
        t, wl, wr = load_data(r)
        plt.plot(t, wl, label=r, linewidth=2, color=COLORS[r])

    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def main():
    print("📊 Plotting wheel speeds for all robots...")

    plot_all_wr()
    plot_all_wl()

    plt.show()


if __name__ == "__main__":
    main()
