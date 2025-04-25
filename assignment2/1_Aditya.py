import numpy as np
import matplotlib.pyplot as plt
import sys
import random as rand
from scipy.stats import norm

# rand.seed(123)


def bernoulli(p=0.5):
    return (1 if rand.uniform(0, 1) >= 0.5 else 0)


def uniform():
    return rand.uniform(0, 1)


def exponential(lam):
    x = rand.uniform(0, 1)
    return -(1 / lam) * np.log(1 - x) if x != 1 else 0


def plot_gaussian(mean, variance):
    x_range = np.linspace(mean - 4*np.sqrt(variance),
                          mean + 4*np.sqrt(variance), 10000)

    plt.plot(x_range, norm.pdf(x_range, mean, np.sqrt(variance)),
             label=f"normal({mean}, {variance})")


def solve_mode_0(N, n, p):
    mean = p
    variance = p * (1 - p) / n
    sample_matrix = np.zeros((N, n))
    for i in range(N):
        for j in range(n):
            sample_matrix[i, j] = bernoulli(p)

    averages_vector = np.mean(sample_matrix, 1)
    plt.hist(averages_vector, bins=int(np.sqrt(N)),
             label="histogram", density=True)
    return mean, variance


def solve_mode_1(N, n):
    mean = 0.5
    variance = (1/12)/n
    sample_matrix = np.zeros((N, n))
    for i in range(N):
        for j in range(n):
            sample_matrix[i, j] = uniform()

    averages_vector = np.mean(sample_matrix, 1)
    plt.hist(averages_vector, bins=int(np.sqrt(N)),
             label="histogram", density=True)
    return mean, variance


def solve_mode_2(N, n, lam):
    mean = 1/(lam)
    variance = 1/((lam)**2 * n)
    sample_matrix = np.zeros((N, n))
    for i in range(N):
        for j in range(n):
            sample_matrix[i, j] = exponential(lam)

    averages_vector = np.mean(sample_matrix, 1)
    plt.hist(averages_vector, bins=int(np.sqrt(N)),
             label="histogram", density=True)
    return mean, variance


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3 or len(args) > 4:
        print("USAGE: python3 1_Aditya.py [MODE] [N] [n] [PARAM]")
        sys.exit(1)

    try:
        mode = int(args[0])
        N = int(args[1])
        n = int(args[2])
        param = float(args[3]) if len(args) == 4 else 0.0
    except ValueError:
        print(
            "Error: [MODE], [N], [n] must be integers and [PARAM] must be a float if provided.")
        sys.exit(1)

    if mode == 0:
        mean, variance = solve_mode_0(N, n, param)
    elif mode == 1:
        mean, variance = solve_mode_1(N, n)
    elif mode == 2:
        mean, variance = solve_mode_2(N, n, param)
    else:
        print("Error: Allowed values for [MODE] are 0, 1 and 2.")
        sys.exit(1)

    plot_gaussian(mean, variance)
    plt.legend(loc='best')
    plt.show()
