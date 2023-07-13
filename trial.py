import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

x = pd.read_csv('heli_cmm.txt', sep=',', header=None).to_numpy()
y = pd.read_csv('ct_cmm.txt', sep=',', header=None).to_numpy()

lst1 = np.reshape(x, (6,3))
lst2 = np.reshape(y, (6,3))

num_rows = len(lst2)

# Create a random permutation of the row indices.
row_indices = np.arange(num_rows)
row_shuffle = np.random.shuffle(row_indices)

# Shuffle the rows in the array.
shuffle_lst2 = lst2[row_indices, :]

dst = distance_matrix(lst1, shuffle_lst2, p=2)

dist_X = distance_matrix(lst1,lst1,p=2)
dist_Y = distance_matrix(shuffle_lst2,shuffle_lst2,p=2)

ord_lst1 = []
ord_shuffle_lst2 = []
correspondence_indices = []

for i in range(min(len(lst1), len(shuffle_lst2))):
    index = np.unravel_index(np.argmin(dst), shape=dst.shape)
    ord_lst1.append(lst1[index[0], :])
    ord_shuffle_lst2.append(shuffle_lst2[index[1], :])
    correspondence_indices.append((index[0], index[1]))

    dst[index[0], :] = np.inf
    dst[:, index[1]] = np.inf

# Print correspondence indices
for i, (idx1, idx2) in enumerate(correspondence_indices):
    value1 = lst1[idx1]
    value2 = shuffle_lst2[idx2]
    print(f"\nCorrespondence {i+1}: Heli_CMM Point {idx1}: {value1} <-> CT_CMM Point {idx2}: {value2}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(lst1[:, 0], lst1[:, 1], lst1[:, 2], label="Heli_CMM")
ax.scatter(shuffle_lst2[:, 0], shuffle_lst2[:, 1], shuffle_lst2[:, 2], label="CT_CMM")

for p1, p2 in zip(ord_lst1, ord_shuffle_lst2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "--", color="black")
    
ax.legend()
plt.show()