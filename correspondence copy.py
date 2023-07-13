import pandas as pd
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from svd import point_to_point_registration_sphere2ctcmm
from functions import CamRoboRegistration
from numpy.linalg import inv,norm
from itertools import combinations

np.set_printoptions(suppress=True)

#Shuffle the spherical marker detected points from WIF
def shuffle_array_row_wise(spherical_marker_pos):
    # Get the number of rows in the array.
    num_rows = len(spherical_marker_pos)

    # Create a random permutation of the row indices.
    row_indices = np.arange(num_rows)
    np.random.shuffle(row_indices)

    # Shuffle the rows in the array.
    return spherical_marker_pos[row_indices, :]

#Compute the correspondence between the given set of points
def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: shuffled spherical marker pos
        dst: cmm2phs_pos
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1,algorithm='kd_tree')
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)

    return distances.ravel(), indices.ravel()

def nearest_neighbor_test(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: shuffled spherical marker pos
        dst: cmm2phs_pos
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    assert src.shape == dst.shape
    indexes = [m for m in range(len(src))] 
    ind = indexes.copy()
    p1 = indexes.pop(1)
    p2 = indexes.pop(1)
    dist = norm(src[p1]-src[p2])
    print(dist)
    comb = combinations(ind, 2)
    
    for i in list(comb):
        print(norm(src[i[0]]-src[i[1]]),i)
        



    

  



    neigh = NearestNeighbors(n_neighbors=1,algorithm='kd_tree')
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)

    return distances.ravel(), indices.ravel()

#Transformation of Shuffled_pts_2_ctcmm_pts
def point_to_point_registration_shuffle2ct(source, target):
    '''
    Find the Transformation of target to source
    Input:
        point_to_point_registration_sphere2ctcmm function from svd.py file
    Output:
        T_new: transformation matrix
    '''
    T_new = point_to_point_registration_sphere2ctcmm(source, target)
    return T_new

def main():
    #load the text files
    x1 = pd.read_csv('heli_cmm.txt', sep=',', header=None).to_numpy()
    y1 = pd.read_csv('ct_cmm.txt', sep=',', header=None).to_numpy()

    spherical_marker_pos = np.reshape(y1, (6,3))# ct_cmm
    shuffle_spherical_marker_pos = shuffle_array_row_wise(spherical_marker_pos) # shuffled points
    cmm2phs_pos = np.reshape(x1, (6,3)) #heli_cmm points

    correspondence = nearest_neighbor_test(shuffle_spherical_marker_pos,cmm2phs_pos) #to compute the distance and index 
    transformation,error_shuff = CamRoboRegistration(shuffle_spherical_marker_pos,cmm2phs_pos[correspondence[1]])# transformation shuffle2ct

    final_trans,error = CamRoboRegistration(spherical_marker_pos,cmm2phs_pos)# transformation shuffle2original
    # @np.linalg.inv(transformation) 
    print ("\nTransformation of shuffled points in original spherical marker detected space:\n", final_trans)

    return final_trans
    
if __name__ == "__main__":
    main()