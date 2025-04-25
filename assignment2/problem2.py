import numpy as np
import matplotlib.pyplot as plt
import sys
import random as rand
import math


def density_1(x):
    factor = math.factorial(6)/(math.factorial(3) * math.factorial(2))
    return factor * x ** 3 * (1 - x) ** 2


def density_2(x):
    factor = math.factorial(6)/(math.factorial(4) * math.factorial(1))
    return factor * x ** 4 * (1 - x)


def plot_density(density, lab):
    x_range = np.linspace(0, 1, 1000)
    plt.plot(x_range, density(x_range), label=lab)


def simulate_X(N):
    arrival_matrix = np.zeros((N, 6))
    for j in range(N):
        for k in range(6):
            arrival_matrix[j, k] = rand.uniform(0, 1)

    for j in range(N):
        arrival_matrix[j] = np.sort(arrival_matrix[j])

    fourth_latest_values = arrival_matrix[:, 3].T

    counts, bins, _ = plt.hist(
        fourth_latest_values, bins=100, density=True, label="histogram")

    bin_centers = (bins[:-1] + bins[1:])/2

    error_density_1 = np.linalg.norm((np.abs(density_1(bin_centers) -
                                             counts))/(bin_centers[-1] - bin_centers[0]))
    error_density_2 = np.linalg.norm((np.abs(density_2(bin_centers) -
                                             counts))/(bin_centers[-1] - bin_centers[0]))

    if (error_density_1 < error_density_2):
        print("f_a")
    else:
        print("f_b")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error\nUSAGE: python3 problem2.py [N]")
        exit(1)

    N = sys.argv[1]

    if (N.isnumeric()):
        simulate_X(int(N))
        plot_density(density_1, r"$f_a \left(x\right)$")
        plot_density(density_2, r"$f_b \left(x\right)$")
    else:
        print("Error: N needs to be a natural number")
        exit(1)

    plt.legend(loc='best')
    plt.show()
