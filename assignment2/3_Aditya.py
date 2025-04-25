import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


def get_mmse(data, mean, variance):
    points = np.zeros((data.shape[0], 2))
    for n in range(data.shape[0]):
        var = 1 / (1/variance + np.sum(1/data[0:n, 1]))
        mu = (mean / variance + np.sum(data[:n, 0]/data[:n, 1])) * var

        points[n, 0] = n
        points[n, 1] = mu

    return points


def plot_mmse(points):
    plt.scatter(points[:, 0], points[:, 1],
                label="MMSE estimate")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Error\nUSAGE python3 3_Aditya.py [MEAN] [VARIANCE] [FILEPATH]")
        exit(1)

    try:
        mean = float(sys.argv[1])
        variance = float(sys.argv[2])
    except ValueError:
        print("Error: [MEAN] and [VARIANCE] accept floating point values only")
        exit(1)
    try:
        filepath = sys.argv[3]
        samples = pd.read_csv(filepath).iloc[1:].values
    except FileNotFoundError:
        print("CSV file not found.")
        exit(1)

    mmse_vals = get_mmse(samples, mean, variance)
    plot_mmse(mmse_vals)
    plt.title("Estimate for X vs. Number of observations")
    plt.legend(loc="best")
    plt.show()
