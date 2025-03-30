import random as rand 
import math 
import sys 
import matplotlib.pyplot as plt
import numpy as np 

rand.seed(42)

def compute_histogram_values(n, k, N):
    frequency_array = np.zeros(int(math.pow(2, n))) 
    for _ in range(N):
        num_to_choose = k # this variable corresponds to the number of elements still left to choose after certain number of iterations
        index_of_subset = 0 # just an initialization value
        uniform_samples_array = np.array([rand.uniform(0, 1) for _ in range(n)])
        for j in range(n):
            if uniform_samples_array[j] <= (num_to_choose)/(n - j):
                index_of_subset <<= 1
                index_of_subset += 1
                num_to_choose -= 1
            else :
                index_of_subset <<= 1
        frequency_array[index_of_subset] += 1
    
    return frequency_array/N

def plot_histogram(frequency_array):
    indices = np.arange(len(frequency_array))
    plt.bar(indices, frequency_array)
    plt.show()

if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print("Incorrect usage.\nUSAGE: python3 3_AdityaTripathy.py [n] [k] [N]")
        exit(1)
    _, n, k, N = sys.argv
    n, k, N = int(n), int(k), int(N)
    if (k > n):
        print("Incorrect usage.\nk cannot be greater than n")
        exit(1)

    plot_histogram(compute_histogram_values(n, k, N))
