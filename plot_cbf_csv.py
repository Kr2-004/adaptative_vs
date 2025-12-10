#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import pandas as pd

pzbt_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
def plot_csv(filename):
    print(f"Plotting {filename} ...")

    df = pd.read_csv(filename)

    t = df["t"].to_numpy()
    s1 = df["slot1"].to_numpy()
    s2 = df["slot2"].to_numpy()
    s3 = df["slot3"].to_numpy()
    s4 = df["slot4"].to_numpy()

    plt.figure(figsize=(10,6))
    plt.plot(t, s1, label="Slot 1", linewidth=2, color = pzbt_colors[0])
    plt.plot(t, s2, label="Slot 2", linewidth=2, color = pzbt_colors[1])
    plt.plot(t, s3, label="Slot 3", linewidth=2, color = pzbt_colors[2])
    plt.plot(t, s4, label="Slot 4", linewidth=2, color = pzbt_colors[3])

    plt.axhline(0, color="black", linestyle="--", label="CBF=0")
    plt.xlabel("Time [s]")
    plt.ylabel("CBF h(x)")
    plt.title("CBF Values Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_cbf_csv.py <filename>")
        sys.exit(1)

    plot_csv(sys.argv[1])
