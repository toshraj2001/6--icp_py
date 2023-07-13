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

# dist_X = distance_matrix(lst1,lst1,p=2)
# dist_Xc = dist_X.copy()
# dist_Y = distance_matrix(lst2,lst2,p=2)
# dist_Yc = dist_Y.copy()

# def compute_intermarker(f,fiducial_count):
#     comb = combinations(np.arange(fiducial_count),2)
#     imd = []
#     for indices in comb:
#         # print(indices)
#         dist =norm(f[indices[0]]-f[indices[1]])
#         imd.append([indices[0],indices[1],dist])

#     imd_array = np.array(imd)
#     return imd_array

def compute_intermarker(f,fiducial_count):
    perm = permutations(np.arange(fiducial_count),2)
    imd = []
    for indices in perm:
        dist =norm(f[indices[0]]-f[indices[1]])
        imd.append([indices[0],indices[1],dist])

    imd_array = np.array(imd)
    return imd_array

imd_X = compute_intermarker(lst1, len(lst1))
imd_Y = compute_intermarker(lst2, len(lst2))

sort_imd_X = imd_X[imd_X[:,2].argsort()]
sort_imd_Y = imd_Y[imd_Y[:,2].argsort()]

def compute_corresponding_rows(sort_imd_X, sort_imd_Y, threshold):
    corresponding_rows = []
    for rowx in range(len(sort_imd_X[:,2])):
        min_error = float("inf")
        x_index = -1
        y_index = -1
        for rowy in range(len(sort_imd_Y[:,2])):
            #valid_indices = min(len(dist_Xc[rowx]), len(dist_Yc[rowy]))
            distances = np.abs(sort_imd_X[rowx][2] - sort_imd_Y[rowy][2])
            if np.all(distances < threshold):
                lstsq_error = norm(sort_imd_X[rowx][2] - sort_imd_Y[rowy][2])
                if lstsq_error < min_error:
                    min_error = lstsq_error
                    x_index = rowx
                    y_index = rowy
        if x_index != -1 and y_index != -1:
            corresponding_rows.append((min_error, x_index, y_index, sort_imd_X[x_index], sort_imd_Y[y_index]))
    return corresponding_rows

threshold = 1
corresponding_rows = compute_corresponding_rows(sort_imd_X, sort_imd_Y, threshold)

print(corresponding_rows)

# def sortRowWise(m):   
#     # loop for rows of matrix
#     for i in range(len(m)): 
#         # loop for column of matrix
#         for j in range(len(m[i])):    
#             # loop for comparison and swapping` `
#             for k in range(len(m[i]) - j - 1):   
#                 if (m[i][k] > m[i][k + 1]):    
#                     # swapping of elements
#                     t = m[i][k]
#                     m[i][k] = m[i][k + 1]
#                     m[i][k + 1] = t

# sorted_dist_X = sortRowWise(dist_Xc)
# sorted_dist_Y = sortRowWise(dist_Yc)

# def compute_corresponding_rows(imd_dist_X, imd_dist_Y, threshold):
#     corresponding_rows = []
#     for rowx in range(len(dist_Xc)):
#         min_error = float("inf")
#         x_index = -1
#         y_index = -1
#         for rowy in range(len(dist_Yc)):
#             valid_indices = min(len(dist_Xc[rowx]), len(dist_Yc[rowy]))
#             distances = np.abs(dist_Xc[rowx][:valid_indices] - dist_Yc[rowy][:valid_indices])
#             if np.all(distances < threshold):
#                 lstsq_error = norm(dist_Xc[rowx][:valid_indices] - dist_Yc[rowy][:valid_indices])
#                 if lstsq_error < min_error:
#                     min_error = lstsq_error
#                     x_index = rowx
#                     y_index = rowy
#         if x_index != -1 and y_index != -1:
#             corresponding_rows.append((min_error, x_index, y_index, dist_Xc[x_index], dist_Yc[y_index]))
#     return corresponding_rows

# def compute_corresponding_rows(dist_Xc, dist_Yc, threshold):
#     corresponding_rows = []
#     for rowx in range(len(dist_Xc)):
#         min_error = float("inf")
#         for rowy in range(len(dist_Yc)):
#             for k in range(min(len(dist_Xc),len(dist_Yc))):
#                 distance = np.abs(dist_Xc[rowx][k] - dist_Yc[rowy][k])
#                 if distance > threshold:
#                     a=0
#                     break
#             if a != 0:
#                 lstsq_error = norm((dist_Xc[rowx]-dist_Yc[rowy]))
#                 print(rowx,rowy,lstsq_error)
#                 if lstsq_error < min_error:
#                     min_error = lstsq_error
#                     x_index = rowx
#                     y_index = rowy
#             if a==0:
#                 a=a+1
#     if min_error <= threshold:
#         corresponding_rows.append((x_index, y_index, dist_Xc[x_index], dist_Yc[y_index]))

#     return corresponding_rows