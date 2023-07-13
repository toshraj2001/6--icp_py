import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)

#Transformation of original_pts_2_ctcmm_pts
def point_to_point_registration_sphere2ctcmm(source, target):
    '''
    Find the Transformation of target to source
    Input:
        source: spherical marker pos
        target: cmm2phs_pos
    Output:
        T: transformation matrix
    '''
        
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
    x = pd.read_csv('D:\Sphere_detect\spherical_marker_detection_final.txt', sep=' ', header=None).to_numpy()
    y = pd.read_csv('D:\Sphere_detect\Dataset\Dataset-2\ct_cmm.txt', sep=',', header=None).to_numpy()

    # Compute transformation matrix
    T = point_to_point_registration_sphere2ctcmm(x, y)

    #print("Transformation matrix:")
    #print(T)

if __name__ == "__main__":
    main()
