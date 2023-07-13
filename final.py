import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def rotate_vec(r,p_vec):
    rotated_vec = []
    for i in range (len(p_vec)):
        rot_vec =r.apply(p_vec[i])
        rotated_vec.append(rot_vec)  
    return np.array(rotated_vec)

r = R.from_quat([0, 0, np.sin(np.pi/6), np.cos(np.pi/6)])

x = np.array([[10,0,0],[30,0,0],[0,40,0],[0,10,10],[10,10,300],[0,50,60]])
y = rotate_vec(r,x)

# x = pd.read_csv('heli_cmm.txt', sep=',', header=None).to_numpy()
# y = pd.read_csv('ct_cmm.txt', sep=',', header=None).to_numpy()

m = 6
n = 3

# lst1 = np.reshape(x, (m,n))
# lst2 = np.reshape(y, (m,n))

lst1 = x
lst2 = y[[2,1,3,0,5,4],:]

dist_X = distance_matrix(lst1,lst1,p=2)
dist_Y = distance_matrix(lst2,lst2,p=2)
dist_XY = distance_matrix(lst1,lst2,p=2)

ord_lst1 = []
ord_lst2 = []
XY_indices = []

corres = {}
intersec_ip = {}

iterr_x = 0
flag = True

for i in range(m):
    corres[i] = []
    index = np.unravel_index(np.argmin(dist_XY), shape=dist_XY.shape)
    ord_lst1.append(lst1[index[0], :])
    ord_lst2.append(lst2[index[1], :])
    XY_indices.append((index[0], index[1]))

    dist_XY[index[0], :] = np.inf
    dist_XY[:, index[1]] = np.inf
    
    for j in range(m):
        err = np.linalg.norm(np.sort(dist_X,axis=1)[i] - np.sort(dist_Y,axis=1)[j])
        if err < 1 and flag:
            for iterr_x in range(m):
                for iterr_y in range(m):
                    element_err = dist_X[i,iterr_x] - dist_Y[j,iterr_y]
                    # print(dist_X[i] , dist_Y[j],i,j)
                    if np.abs(element_err) < 1:
                        if iterr_y in corres.keys():
                            corres[iterr_y].append(iterr_x)
                        else:
                            corres[iterr_y] = [iterr_x]
            # flag = False
            intersec_ip[i] = corres 
            corres = {}

# Get the common keys from all rows
common_keys = set(intersec_ip[0].keys())
for row in intersec_ip.values():
    common_keys.intersection_update(row.keys())

# Compute the intersection of values for each common key
result = {}
for key in common_keys:
    pair_values = [set(row[key]) for row in intersec_ip.values()]
    intersection = set.intersection(*pair_values)
    result[key] = list(intersection)

# Separate keys and values into lists
keys_list = list(result.keys())
values_list = list(result.values())

# Print the keys and values lists
print("Keys List:", keys_list)
print("Values List:", values_list)
val = [x[0] for x in values_list]

x_points = np.array(lst1)
# y_points = np.array([[values_list][k][0] for k in range(len(values_list))])
Y_in = np.array(lst2)
y_points = Y_in

x_points = x_points[val,:]
# Print correspondence indices
for i, (idx1, idx2) in enumerate(XY_indices):
    value1 = x_points[idx1]
    value2 = y_points[idx2]
    print(f"\nCorrespondence {i+1}: X Point {idx1}: {value1} <-> Y Point {idx2}: {value2}")

# Plot X and Y points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_points[:,0], x_points[:,1], x_points[:,2], c='blue', label='X')
ax.scatter(y_points[:,0], y_points[:,1], y_points[:,2], c='red', label='Y')

# Connect corresponding points
for i, (idx1, idx2) in enumerate(XY_indices):
    x_connect = x_points[idx1]
    y_connect = y_points[idx2]
    ax.plot([x_connect[0], y_connect[0]], [x_connect[1], y_connect[1]], [x_connect[2], y_connect[2]], c='green', linestyle='dashed')
    ax.text(x_connect[0], x_connect[1], x_connect[2], str(x_points[idx1]), color='black', fontsize=8, horizontalalignment='left')
    ax.text(y_connect[0], y_connect[1], y_connect[2], str(y_points[idx2]), color='black', fontsize=8, horizontalalignment='left')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Corresponding X and Y Points')
ax.legend()
plt.grid(True)
plt.show()


def fun(a,b,c):
    return a, b,c


inn = [1,2,3]
fun(*inn)

print(fun(*inn))