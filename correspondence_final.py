import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R

def rotate_vec(r,p_vec):
    rotated_vec = []
    for i in range (len(p_vec)):
        rot_vec =r.apply(p_vec[i])
        rotated_vec.append(rot_vec)  
    return np.array(rotated_vec)

r = R.from_quat([0, 0, np.sin(np.pi/6), np.cos(np.pi/6)])

# x = pd.read_csv('heli_cmm.txt', sep=',', header=None).to_numpy()
# y = pd.read_csv('ct_cmm.txt', sep=',', header=None).to_numpy()

x = np.array([[10,0,0],[30,0,0],[0,40,0],[0,10,10],[10,10,300],[0,50,60]])
y = rotate_vec(r,x)

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

# Print correspondence indices
for i, (idx1, idx2) in enumerate(XY_indices):
    value1 = lst1[idx1]
    value2 = lst2[idx2]
    print(f"\nCorrespondence {i+1}: Heli_CMM Point {idx1}: {value1} <-> CT_CMM Point {idx2}: {value2}")

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

x_points = [lst1[keys_list][k] for k in range(len(keys_list))]
y_points = [y[values_list][k][0] for k in range(len(values_list))]

import matplotlib.pyplot as plt

# Print correspondence indices
for i, (idx1, idx2) in enumerate(result.items()):
    value1 = lst1[idx1]
    value2 = lst2[idx2]
    print(f"\nCorrespondence {i+1}: Heli_CMM Point {idx1}: {value1} <-> CT_CMM Point {idx2}: {value2}")

    plt.scatter(value1, value2, label=f"Key: {key}")

# Plot the points
for key, value in result.items():
    x_points = [lst1[index][0] for index in value]
    y_points = [lst2[index][1] for index in value]
    plt.scatter(x_points, y_points, label=f"Key: {key}")
    print(x_points,y_points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_points[:, 0], x_points[:, 1], x_points[:, 2], label="Heli_CMM")
ax.scatter(y_points[:, 0], y_points[:, 1], y_points[:, 2], label="CT_CMM")

for idx, (p1, p2) in enumerate(zip(x_points, y_points)):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "--", color="black")
    ax.text(p1[0], p1[1], p1[2], str(x_points[idx]), color="red")
    ax.text(p2[0], p2[1], p2[2], str(y_points[idx]), color="blue")

ax.legend()
plt.show()

# Set plot labels and legend
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# Show the plot
plt.show()

# # Get the indices of the points
# point_indices = []
# for indices in result.keys():
#     point_indices.extend(indices)

# # Get the coordinates of the points
# points_x = lst1[point_indices]
# points_y = lst2[point_indices]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points_x[:, 0], points_x[:, 1], points_x[:, 2], label="Heli_CMM")
# ax.scatter(points_y[:, 0], points_y[:, 1], points_y[:, 2], label="CT_CMM")

# for idx, (p1, p2) in enumerate(zip(points_x, points_y)):
#     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "--", color="black")
#     ax.text(p1[0], p1[1], p1[2], str(point_indices[idx]), color="red")
#     ax.text(p2[0], p2[1], p2[2], str(point_indices[idx]), color="blue")

# ax.legend()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points_x[:, 0], points_x[:, 1], points_x[:, 2], label="Heli_CMM")
# ax.scatter(points_y[:, 0], points_y[:, 1], points_y[:, 2], label="CT_CMM")

# for idx, (p1, p2) in enumerate(zip(points_x, points_y)):
#     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "--", color="black")
#     ax.text(p1[0], p1[1], p1[2], str(point_indices[idx]), color="red")
#     ax.text(p2[0], p2[1], p2[2], str(point_indices[idx]), color="blue")

# ax.legend()
# plt.show()


# # # Plot the points
# # plt.scatter(points_x[:, 0], points_x[:, 1], color='red', label='Original')
# # plt.scatter(points_y[:, 0], points_y[:, 1], color='blue', label='Rotated')

# # # Add labels and legend
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # plt.legend()

# # # Show the plot
# # plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points_x[:, 0], points_x[:, 1], points_x[:, 2], label="Heli_CMM")
# ax.scatter(points_y[:, 0], points_y[:, 1], points_y[:, 2], label="CT_CMM")

# for p1, p2 in zip(points_x, points_y):
#     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "--", color="black")
    
# ax.legend()
# plt.show()


                        
             
                       