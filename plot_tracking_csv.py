#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import pandas as pd

def plot_tracking(filename):
    print(f"Plotting: {filename}")

    df = pd.read_csv(filename)

    t = df["t"].to_numpy()

    # ------------ TRAJECTORY PLOT ------------
    fig1 = plt.figure(figsize=(10,7))
    ax1 = fig1.add_subplot(111)
    pzbt_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    vs_ref_colors   = ["#FFB3B3", "#B3B3FF", "#B3FFB3", "#FFD8A8"]  
    for i in range(1, 5):
        xr = df[f"x_r{i}"].to_numpy()
        yr = df[f"y_r{i}"].to_numpy()
        xref = df[f"x_ref{i}"].to_numpy()
        yref = df[f"y_ref{i}"].to_numpy()

        ax1.plot(xr, yr, linewidth=2, color=pzbt_colors[i-1], label=f"Puzzlebot {i}")
        ax1.plot(xref, yref, '--', linewidth=2, color=vs_ref_colors[i-1], label=f"Ref {i}")

    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_title("Trajectory Tracking")
    ax1.legend()
    ax1.grid(True)

    # ------------ DISTANCE ERRORS PLOT ------------
    fig2 = plt.figure(figsize=(10,6))
    ax2 = fig2.add_subplot(111)

    for i in range(1, 5):
        e = df[f"e_dist{i}"].to_numpy()
        ax2.plot(t, e, linewidth=2, label=f"Puzzlebot {i}", color=pzbt_colors[i-1])

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Distance Error [m]")
    ax2.set_title("Tracking Distance Error")
    ax2.legend()
    ax2.grid(True)

    # ------------ SHOW BOTH NON-BLOCKING ------------
    plt.show(block=False)

    # Keep program alive until user closes windows
    print("\nClose the plot windows to exit.")
    plt.pause(0.1)
    input("\nPress ENTER to exit after closing plots...\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_tracking_csv.py <csvfile>")
        sys.exit(1)

    plot_tracking(sys.argv[1])
