import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from numpy.linalg import norm

x = pd.read_csv('heli_cmm.txt', sep=',', header=None).to_numpy()
y = pd.read_csv('ct_cmm.txt', sep=',', header=None).to_numpy()

lst1 = np.reshape(x, (6,3))
lst2 = np.reshape(y, (6,3))

#dst = distance_matrix(lst1, lst2, p=2)

dist_X = distance_matrix(lst1,lst1,p=2)
dist_Y = distance_matrix(lst2,lst2,p=2)
dist_X[dist_X==0] = 1000
dist_Y[dist_Y==0] = 1005

rowx   = 1
min_dist = 1000

Rowx = list(np.arange(6))
Rowy = list(np.arange(6))
Colx = list(np.arange(6))
Coly = list(np.arange(6))

for k in range(len(dist_X)):
    for i in range(min(len(dist_X), len(dist_Y))):
        for j in range(len(dist_X[i])):
            diff = np.abs(dist_X[rowx][k] - dist_Y[i][j])
            if  diff <= 2 and diff < min_dist:
                min_dist = diff
                row_y_sel = i
                row_x_sel = rowx 
                print(diff,rowx,k,i,j)

Rowy.remove(row_x_sel)

def find_correspondace(row_x_sel,row_y_sel):
    print(row_x_sel,row_y_sel)

    