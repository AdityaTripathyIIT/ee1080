#St. Petersburg Paradox

import random as rand 
import math

rand.seed(42)

def simulate_coin_toss(probability_of_heads = 0.5):
    sample_of_uniform = rand.uniform(0, 1)

    return 1 if sample_of_uniform <= probability_of_heads else 0

def average_payout(number_of_games = 100, probability_of_heads = 0.5):
    total_payout = 0
    for _ in range(number_of_games):
        number_of_tosses = 0
        while True:
            number_of_tosses += 1 
            if (simulate_coin_toss(probability_of_heads) == 0):
                break
        total_payout += math.pow(2, number_of_tosses)

    return total_payout/number_of_games

if __name__ == "__main__":
    print(f"m = 100 games -> average payout = {average_payout(100):0.3f}", f"m = 10000 games -> average payout = {average_payout(10000):0.3f}",\
            f"m = 1000000 games -> average payout = {average_payout(1000000):.3f}", sep = "\n", end = "\n")

