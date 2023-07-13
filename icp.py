import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
np.set_printoptions(suppress=True)

from icp_functions import icp

def main():
    x = pd.read_csv('D:\Sphere_detect\spherical_marker_detection_final.txt', sep=' ', header=None).to_numpy()
    y = pd.read_csv('D:\Sphere_detect\Trial\cmm2phan_space_pts_copy.txt', sep=' ', header=None).to_numpy()

    spherical_marker_pos = np.reshape(x, (6,3))
    cmm2phs_pos = np.reshape(y, (6,3))

    spherical_marker_pos.tofile('D:\Sphere_detect\Trial\s1.xyz', sep=' ', format='%f')
    cmm2phs_pos.tofile('D:\Sphere_detect\Trial\c1.xyz', sep=' ', format='%f')

    ICP = icp(spherical_marker_pos, cmm2phs_pos)

    print(ICP)

    return ICP

if __name__ == "__main__":
    main()