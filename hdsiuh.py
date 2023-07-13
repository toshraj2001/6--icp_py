import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

x = pd.read_csv('heli_cmm.txt', sep=',', header=None).to_numpy()
y = pd.read_csv('ct_cmm.txt', sep=',', header=None).to_numpy()

lst1 = np.reshape(x, (6,3))
lst2 = np.reshape(y, (6,3))

dist_X = distance_matrix(lst1,lst1,p=2)
dist_Xc = dist_X.copy()
dist_Y = distance_matrix(lst2,lst2,p=2)
dist_Yc = dist_Y.copy()


# Set the deviation factor
deviation = 0.01

# Initialize a list to store the corresponding rows
corresponding_rows = []

# Iterate through each row in matrix A
for i in range(dist_Xc.shape[0]):
    row_A = dist_Xc[i]
    
    # Iterate through each row in matrix B
    for j in range(dist_Yc.shape[0]):
        row_B = dist_Yc[j]
        
        # Check if the rows correspond based on small deviations
        if np.allclose(row_A, row_B, atol=deviation):
            corresponding_rows.append((i, j))
            break  # Move to the next row in matrix A
            
# Print the corresponding row indices
for correspondence in corresponding_rows:
    print("Row in matrix A: {}, Row in matrix B: {}".format(correspondence[0], correspondence[1]))
    
    print("tosh")
