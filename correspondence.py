import pandas as pd
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
np.set_printoptions(suppress=True)

def shuffle_array_row_wise(spherical_marker_pos):
    # Get the number of rows in the array.
    num_rows = len(spherical_marker_pos)

    # Create a random permutation of the row indices.
    row_indices = np.arange(num_rows)
    np.random.shuffle(row_indices)

    # Shuffle the rows in the array.
    return spherical_marker_pos[row_indices, :]

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)

    return distances.ravel(), indices.ravel()

def point_to_point_registration(source, target):
    # Compute centroids
    C_source = np.mean(source, axis=0)
    C_target = np.mean(target, axis=0)

    # Center the points
    source_centered = source - C_source
    target_centered = target - C_target

    # Compute covariance matrix
    H = np.dot(source_centered.T, target_centered)

    # Perform SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)

    # Compute translation vector
    t = C_target - np.dot(R, C_source)

    # Assemble transformation matrix
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def main():
    x1 = pd.read_csv('spherical_marker_detection_final.txt', sep=' ', header=None).to_numpy()
    y1 = pd.read_csv('ct_cmm.txt', sep=',', header=None).to_numpy()

    spherical_marker_pos = np.reshape(x1, (6,3))
    print("\nSpherical marker detected points", spherical_marker_pos)

    shuffle_spherical_marker_pos = shuffle_array_row_wise(spherical_marker_pos)
    print("\nShuffled spherical marker detected points", shuffle_spherical_marker_pos)

    cmm2phs_pos = np.reshape(y1, (6,3))
    print("\nCT_CMM points", shuffle_spherical_marker_pos)

    correspondence = nearest_neighbor(shuffle_spherical_marker_pos,cmm2phs_pos)

    print ("\nDistance:", correspondence[0], "\nIndex:",correspondence[1])

    transformation = point_to_point_registration(shuffle_spherical_marker_pos,cmm2phs_pos)

    print ("\nTransformation of CT_CMM points in Spherical Marker Detected space:\n", transformation)

    return correspondence, transformation
    
if __name__ == "__main__":
    main()