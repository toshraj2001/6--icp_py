import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from numpy.linalg import norm
from itertools import combinations
from itertools import permutations

x = pd.read_csv('heli_cmm.txt', sep=',', header=None).to_numpy()
y = pd.read_csv('ct_cmm.txt', sep=',', header=None).to_numpy()

lst1 = np.reshape(x, (6,3))
lst2 = np.reshape(y, (6,3))

def compute_intermarker(f,fiducial_count):
    perm = combinations(np.arange(fiducial_count),2)
    imd = []
    for indices in perm:
        dist = norm(f[indices[0]]-f[indices[1]])
        imd.append([indices[0],indices[1],dist])

    imd_array = np.array(imd)
    return imd_array

imd_X = compute_intermarker(lst1, len(lst1))
imd_Y = compute_intermarker(lst2, len(lst2))


def compute_corresponding_rows(imd_X, imd_Y, threshold):
    combinations_array = combinations(imd_Y, 15)
    corresponding_rows = []  
    a = 0

    for comb in combinations_array:
        for rowx in range(len(imd_X[:,2])):
            min_error = float("inf")
            x_index = -1
            y_index = -1
            for rowy in range(len(imd_Y[:,2])):
                relative_difference = np.abs(imd_X[rowx][2] - comb[rowy][2])
                if np.all(relative_difference < threshold):
                    lstsq_error = norm(relative_difference)
                    if lstsq_error < min_error:
                        min_error = lstsq_error
                        x_index = rowx
                        y_index = rowy
            if x_index != -1 and y_index != -1:
                corresponding_rows.append((min_error, x_index, y_index, imd_X[x_index], comb[y_index]))
                print(corresponding_rows)
            a = a + 1

    return corresponding_rows, a

threshold = 2
corresponding, a = compute_corresponding_rows(imd_X, imd_Y, threshold)

print(corresponding)

#######################################################################
# heli_points = []
# ct_points = []

# for corr in corresponding:
#     x_index = corr[3][0]  # Index of the point in heli_cmm.txt
#     y_index = corr[4]  # Index of the point in ct_cmm.txt
    
#     heli_point = lst1[x_index]
#     ct_point = lst2[y_index]
    
#     heli_points.append(heli_point)
#     ct_points.append(ct_point)

# # Printing the extracted points
# print("Heli CMM Points:")
# for point in heli_points:
#     print(point)

# print("\nCT CMM Points:")
# for point in ct_points:
#     print(point)

# print("Value of a:", a)