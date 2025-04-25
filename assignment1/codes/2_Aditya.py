import matplotlib.pyplot as plt
import numpy as np
import random as rand
import sys
import pandas as pd
rand.seed(42)

def read_samples(filepath):
    try:
        # Read CSV file and convert every number to float, skipping the header
        return pd.read_csv(filepath, header=None).iloc[1:].astype(float)
    except Exception as e:
        print(f"Error reading samples file: \n{e}")
        exit(1)

def save_samples(samples, filename):
    try:
        samples.to_csv(filename, index=False)
        print(f"{filename} successfully generated.")
    except Exception as e:
        print(f"Error in saving output CSV file.\nError: \n{e}")

def plot_histogram(samples):
    # Plot histogram of the sample data
    hist, bins = np.histogram(samples[0].to_list(), bins=int(np.sqrt(len(samples[0].to_list()))))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.bar(bin_centers, hist, width=bin_centers[1] - bin_centers[0])
    plt.show()

def solve_mode_0(filepath, p):
    uniform_samples = read_samples(filepath)
    # Generate Bernoulli RV samples from given CSV
    bernoulli_samples = uniform_samples.map(lambda x: 1 if x < p else 0)
    # Compute Sample Mean
    sample_mean = float(bernoulli_samples.mean().iloc[0])
    print(f"Sample Mean: {sample_mean: .3f}")

    st = str(p).replace(".", "p")
    output_filename = f"Bernoulli_{st}.csv"
    save_samples(bernoulli_samples, output_filename)

def solve_mode_1(filepath, lam):
    uniform_samples = read_samples(filepath)
    # Generate Exponential RV samples using uniform samples CSV 
    exponential_samples = uniform_samples.map(lambda x: -(1 / lam) * np.log(1 - x) if x != 1 else 0)
    # Compute Sample Mean
    sample_mean = float(exponential_samples.mean().iloc[0])
    print(f"Sample Mean: {sample_mean}")

    st = str(lam).replace(".", "p")
    st = st[1:] if lam < 1 else st

    output_filename = f"Exponential_{st}.csv"
    save_samples(exponential_samples, output_filename)
    plot_histogram(exponential_samples)

def solve_mode_2(filepath):
    count_2 = 0
    # Define the function determining the law to generate RV samples from uniform samples CSV
    def give_sample(x):
        nonlocal count_2
        if 0 <= x < 1/3:
            return np.sqrt(3 * x)
        elif 1/3 < x <= 2/3:
            count_2 += 1
            return 2
        elif 2/3 <= x <= 1:
            return 6 * x - 2
        else:
            return 0

    uniform_samples = read_samples(filepath)
    samples = uniform_samples.map(lambda x: give_sample(x))

    output_filename = "CDFX.csv"
    save_samples(samples, output_filename)
    print(f"Number of 2's = {count_2}")
    plot_histogram(samples)

if __name__ == "__main__":
    if int(sys.argv[1]) == 0:
        solve_mode_0(sys.argv[2], float(sys.argv[3]))
    elif int(sys.argv[1]) == 1:
        solve_mode_1(sys.argv[2], float(sys.argv[3]))
    elif int(sys.argv[1]) == 2 and len(sys.argv) == 3:
        solve_mode_2(sys.argv[2])
    else:
        print("Incorrect usage.\nUSAGE: python3 2_Aditya.py [MODE] [SAMPLES CSV FILEPATH] [PARAMETER]\nNote: [PARAMETER] arguement is valid only for modes 0 and 1.")
        exit(1)
