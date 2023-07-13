from itertools import combinations
import pandas as pd
import math

def distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def combination(points):
    # Find all combinations of sequences of points
    combinations = list(combinations(points, 2))
    # Compute inter-distances for each combination
    for combination in combinations:
        total_distance = 0
        for i in range(len(combination) - 1):
            total_distance += distance(combination[i], combination[i + 1])
        print(f"Combination: {combination}, Total Distance: {total_distance}")

def main():
    # Sample input points
    heli_cmm = pd.read_csv('heli_cmm.txt', sep=',', header=None).to_numpy()
    ct_cmm = pd.read_csv('ct_cmm.txt', sep=',', header=None).to_numpy()

    source = combination(heli_cmm)
    target = combination(ct_cmm) 




















