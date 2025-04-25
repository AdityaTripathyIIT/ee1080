import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats


def overlay_contour(mean, cov, ax):
    x1 = np.arange(-2.5, 2.5, 0.001)
    x2 = np.arange(-3.5, 3.5, 0.001)

    X1, X2 = np.meshgrid(x1, x2)
    Xpos = np.empty(X1.shape + (2,))
    Xpos[:, :, 0] = X1
    Xpos[:, :, 1] = X2

    F = stats.multivariate_normal.pdf(Xpos, mean, cov)
    ax.contour(x1, x2, F)


def generate_samples_lib(N, mean, cov):
    samples_matrix = stats.multivariate_normal.rvs(mean, cov, size=N)
    return samples_matrix


def scatter_plot(samples_matrix, ax, title):
    ax.scatter(samples_matrix[:, 0], samples_matrix[:, 1], s=5)
    ax.set_title(title)


def generate_samples_manual(N, mean, cov):
    U, D = diagonalize(cov)
    sqrt_D = np.sqrt(D)
    A = U @ sqrt_D
    S = np.random.normal(size=(N, 2)).T
    return (A @ S).T + mean


def diagonalize(cov):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # sorted_indices = np.argsort(eigenvalues)[::-1]
    # eigenvalues = eigenvalues[sorted_indices]
    # eigenvectors = eigenvectors[:, sorted_indices]
    D = np.diag(eigenvalues)
    U = eigenvectors
    return U, D


if __name__ == "__main__":
    mean, cov, N = np.zeros((1, 2)), np.zeros((2, 2)), 0
    if len(sys.argv) != 8:
        print(
            "Error\nUSAGE python3 4_Aditya.py [mean 0] [mean 1] [cov 0,0] [cov 0,1] [cov 1,0] [cov 1,1] [N]")
        exit(1)
    else:
        mean = np.array([float(x) for x in sys.argv[1:3]])
        cov[0] = np.array([float(x) for x in sys.argv[3:5]])
        cov[1] = np.array([float(x) for x in sys.argv[5:7]])
        N = int(sys.argv[7])

    print(f"Mean Vector : {mean.T}")
    print(f"Covariance Matrix: {cov}")

    if not np.array_equal(cov, cov.T):
        print("Error: Covariance Matrix is not symmetric.")
        exit(1)

    if not np.all(np.linalg.eigvals(cov) >= 0):
        print("Error: Covariance Matrix is not positive semidefinite.")
        exit(1)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Library samples
    points_lib = generate_samples_lib(N, mean, cov)
    scatter_plot(points_lib, axs[0], "Library Generated Samples")
    overlay_contour(mean, cov, axs[0])

    # Manual samples
    points_manual = generate_samples_manual(N, mean, cov)
    scatter_plot(points_manual, axs[1], "Manually Generated Samples")
    overlay_contour(mean, cov, axs[1])

    plt.tight_layout()
    plt.show()
