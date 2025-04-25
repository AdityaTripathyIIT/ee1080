import numpy as np 
import sys 
import random as rand 
import matplotlib.pyplot as plt 

rand.seed(42)

def solve_mode_0(num_samples):
    desired_chords = 0
    chord_lengths = list()

    for _ in range(num_samples):
        angle = rand.uniform(0, np.pi)
        chord_len = 2 * np.abs(np.sin(angle))
        
        chord_lengths.append(chord_len)

        if chord_len > np.sqrt(3):
            desired_chords += 1
    
    probability = desired_chords / num_samples
    
    print(f"Probability for Mode 0: {probability}")
    chord_lengths = np.array(chord_lengths)

    hist, bins = np.histogram(chord_lengths, bins = int(np.sqrt(num_samples)))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.bar(bin_centers, hist, width=bin_centers[1] - bin_centers[0])
    plt.show()


def solve_mode_1(num_samples):
    desired_chords = 0
    chord_lengths = list()

    for _ in range(num_samples):
        center_distance = rand.uniform(0, 1)
        chord_len = 2 * np.sqrt((1 - center_distance ** 2))
        
        if chord_len > np.sqrt(3):
            desired_chords += 1
        
        chord_lengths.append(chord_len)
    
    probability = desired_chords / num_samples
    print(probability)
    chord_lengths = np.array(chord_lengths)

    hist, bins = np.histogram(chord_lengths, bins = int(np.sqrt(num_samples)))

    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.bar(bin_centers, hist, width=bin_centers[1] - bin_centers[0])
    plt.show()


def solve_mode_2(num_samples):
    desired_chords = 0 
    chord_lengths = list()

    for _ in range(num_samples):
        # points are generated from polar coordinates
        # cdf(r, theta) = (pi * r^2 * theta / 2pi) / pi = r^2 * theta / 2 * pi
        # pdf(r, theta) = r * /pi 
        # pdf(r) = 2 * r 
        # cdf(r) = r^2
        # since cdf(r) = uniform -> r = sqrt(uniform)
        chord_center_r = np.sqrt(rand.uniform(0, 1))
        chord_center_theta = rand.uniform(0, 2 * np.pi)
        
        chord_len = 2 * np.sqrt(1 - chord_center_r ** 2)

        if chord_len > np.sqrt(3):
            desired_chords += 1 
        
        chord_lengths.append(chord_len)

    probability = desired_chords / num_samples        
    print(probability)
    chord_lengths = np.array(chord_lengths)
    
    hist, bins = np.histogram(chord_lengths, bins = int(np.sqrt(num_samples)))

    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.bar(bin_centers, hist, width=bin_centers[1] - bin_centers[0])
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3 :
        print("Invalid usage.\nUSAGE python3 3_Aditya.py [MODE] [NUM_SAMPLES]")
        exit(1)

    if sys.argv[1] not in "012":
        print("Invalid usage.\n MODE can only take values 0, 1 or 2")
        exit(1)

    if not (sys.argv[1].isnumeric()):
        print("Invalid usage.\n MODE can only take values 0, 1 or 2")
        exit(1)

    mode = int(sys.argv[1])
    num_samples = int(sys.argv[2])

    if mode == 0:
        solve_mode_0(num_samples)

    if mode == 1:
        solve_mode_1(num_samples)

    if mode == 2:
        solve_mode_2(num_samples)
