#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# -------------------- Matplotlib global config --------------------
mpl.rcParams.update({
    "figure.subplot.left":   0.07,
    "figure.subplot.right":  0.97,
    "figure.subplot.bottom": 0.09,
    "figure.subplot.top":    0.97,
    "figure.subplot.wspace": 0.20,
    "figure.subplot.hspace": 0.20,
})

# -------------------- Main plotting function --------------------
def plot_tracking(filename="tracking_log.csv"):
    print(f"Plotting: {filename}")

    df = pd.read_csv(filename)
    t = df["t"].to_numpy()

    pzbt_colors = ["#FF0000", "#0000FF", "#008800", "#000000"]
    vs_ref_colors = ["#A50000", "#0000A1", "#00C900", "#6B6868"]

    # ==========================================================
    # TRAJECTORY PLOT
    # ==========================================================
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)

    for i in range(1, 5):
        xr = df[f"x_r{i}"].to_numpy()
        yr = df[f"y_r{i}"].to_numpy()
        xref = df[f"x_ref{i}"].to_numpy()
        yref = df[f"y_ref{i}"].to_numpy()

        ax1.plot(xr, yr, linewidth=3.5, color=pzbt_colors[i-1], label=f"Puzzlebot {i}")
        ax1.plot(xref, yref, '--', linewidth=3, color=vs_ref_colors[i-1], label=f"Ref {i}")

    ax1.set_xlabel("X [m]", fontsize=30)
    ax1.set_ylabel("Y [m]", fontsize=30)
    ax1.set_xlim(-4.5, 3.1)
    ax1.set_ylim(-1.4, 2)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.grid(True)
    ax1.legend(loc="lower left", ncol=4, frameon=True, fontsize=17)

    # -------------------- CENTERED START / END MARKERS --------------------
    # Start (left-center)
    ax1.plot(
        0.01, 0.5,
        marker=">",
        markersize=14,
        markeredgewidth=2.5,
        markeredgecolor="black",
        markerfacecolor="none",
        transform=ax1.transAxes,
        zorder=10
    )
    ax1.text(
        0.025, 0.55,
        "Start",
        transform=ax1.transAxes,
        fontsize=22,
        ha="center",
        va="top"
    )

    # End (right-center)
    ax1.plot(
        0.98, 0.5,
        marker="X",
        markersize=16,
        markeredgewidth=3,
        color="black",
        transform=ax1.transAxes,
        zorder=10
    )
    ax1.text(
        0.98, 0.55,
        "End",
        transform=ax1.transAxes,
        fontsize=22,
        ha="center",
        va="top"
    )

    # ==========================================================
    # DISTANCE ERROR PLOT
    # ==========================================================
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)

    for i in range(1, 5):
        e = df[f"e_dist{i}"].to_numpy()
        ax2.plot(t, e, linewidth=2.5, color=pzbt_colors[i-1], label=f"Puzzlebot {i}")

    ax2.set_xlabel("Time [s]", fontsize=30)
    ax2.set_ylabel("Distance Error [m]", fontsize=30)
    ax2.set_xlim(0, 65.3)
    ax2.set_ylim(0, 0.35)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=17)

    # ==========================================================
    # SHOW
    # ==========================================================
    plt.show(block=False)

    print("\nClose the plot windows to exit.")
    plt.pause(0.1)
    input("\nPress ENTER to exit after closing plots...\n")

# -------------------- Entry point --------------------
if __name__ == "__main__":
    plot_tracking()
