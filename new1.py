import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from numpy.linalg import norm

x = pd.read_csv('heli_cmm.txt', sep=',', header=None).to_numpy()
y = pd.read_csv('ct_cmm.txt', sep=',', header=None).to_numpy()

lst1 = np.reshape(x, (6,3))
lst2 = np.reshape(y, (6,3))

dist_X = distance_matrix(lst1,lst1,p=2)
dist_Xc = dist_X.copy()
dist_Y = distance_matrix(lst2,lst2,p=2)
dist_Yc = dist_Y.copy()

def sortRowWise(m):   
    # loop for rows of matrix
    for i in range(len(m)): 
        # loop for column of matrix
        for j in range(len(m[i])):    
            # loop for comparison and swapping` `
            for k in range(len(m[i]) - j - 1):   
                if (m[i][k] > m[i][k + 1]):    
                    # swapping of elements
                    t = m[i][k]
                    m[i][k] = m[i][k + 1]
                    m[i][k + 1] = t

sorted_dist_X = sortRowWise(dist_Xc)
sorted_dist_Y = sortRowWise(dist_Yc)

def compute_corresponding_rows(imd_dist_X, imd_dist_Y, threshold):
    corresponding_rows = []
    for rowx in range(len(dist_Xc)):
        min_error = float("inf")
        x_index = -1
        y_index = -1
        for rowy in range(len(dist_Yc)):
            valid_indices = min(len(dist_Xc[rowx]), len(dist_Yc[rowy]))
            relative_difference = np.abs(dist_Xc[rowx][:valid_indices] - dist_Yc[rowy][:valid_indices])
            if np.all(relative_difference < threshold):
                lstsq_error = norm(dist_Xc[rowx][:valid_indices] - dist_Yc[rowy][:valid_indices])
                if lstsq_error < min_error:
                    min_error = lstsq_error
                    x_index = rowx
                    y_index = rowy
        if x_index != -1 and y_index != -1:
            corresponding_rows.append((min_error, x_index, y_index, dist_Xc[x_index], dist_Yc[y_index]))
    return corresponding_rows


threshold = 1
corresponding_rows = compute_corresponding_rows(sorted_dist_X, sorted_dist_Y, threshold)

print(corresponding_rows)