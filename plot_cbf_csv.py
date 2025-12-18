#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

mpl.rcParams.update({
    "figure.subplot.left":   0.07,
    "figure.subplot.right":  0.97,
    "figure.subplot.bottom": 0.09,
    "figure.subplot.top":    0.97,
    "figure.subplot.wspace": 0.20,
    "figure.subplot.hspace": 0.20,
})
pzbt_colors = ["#FF0000", "#0000FF", "#008800", "#000000"]
def plot_csv(filename="cbf_log.csv"):
    print(f"Plotting {filename} ...")

    df = pd.read_csv(filename)

    t = df["t"].to_numpy()
    s1 = df["slot1"].to_numpy()
    s2 = df["slot2"].to_numpy()
    s3 = df["slot3"].to_numpy()
    s4 = df["slot4"].to_numpy()

    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(t, s1, label="Slot 1", linewidth=3, color=pzbt_colors[0])
    ax.plot(t, s2, label="Slot 2", linewidth=3, color=pzbt_colors[1])
    ax.plot(t, s3, label="Slot 3", linewidth=3, color=pzbt_colors[2])
    ax.plot(t, s4, label="Slot 4", linewidth=3, color=pzbt_colors[3])

    ax.axhline(0, color="black", linewidth=2, linestyle="--", label="CBF=0")
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("Time [s]", fontsize =30)
    ax.set_ylabel("CBF h(x)", fontsize =30)
    ax.set_xlim(0, 67.2)
    ax.set_ylim(-0.2, 15)
    ax.legend(loc = "upper right", fontsize = 17)
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_csv()
