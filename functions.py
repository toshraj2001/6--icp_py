
import SimpleITK as sitk
import numpy as np
# import json
from numpy.linalg import inv
import configparser
import os
import open3d as o3d


def rot2tf(rot,pos):
        
    pos_s = np.array(pos)
    rot_matrix = rot
    temp= np.column_stack((rot_matrix,pos_s))
    tf= np.vstack((temp,[0,0,0,1]))
    return tf


# read the geometry file
def read_ini(path,geometry_path):
    config = configparser.ConfigParser()

    file_name = os.path.join(path,geometry_path)
    config.read(file_name)

    marker_id = path[8:12]
    keys = config.sections()
    marker_count = len(keys)-1
    # inlineTop = 1
    # inlineBottom = 3
    xCoord = []
    yCoord = []
    zCoord = []
    for k in keys:
        if k != 'geometry':
            xCoord.append (float(config[k]['x']))
            yCoord.append (float(config[k]['y']))
            zCoord.append (float(config[k]['z']))
            


    xx= np.array(xCoord)
    yy= np.array(yCoord)
    zz= np.array(zCoord)
    fids = np.vstack((xx,yy,zz)).transpose()
    return fids
# register between two points
def CamRoboRegistration(CT_Space,Robot_Space):
    CT_Space_pointCloud = o3d.geometry.PointCloud()
    CT_Space_pointCloud.points = o3d.utility.Vector3dVector(CT_Space)
    Robot_Space_pointCloud = o3d.geometry.PointCloud()
    Robot_Space_pointCloud.points = o3d.utility.Vector3dVector(Robot_Space)
    corres_mat = np.asarray([[0,0],[1,1],[2,2],[3,3]])
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    transformation_mat = p2p.compute_transformation(CT_Space_pointCloud, Robot_Space_pointCloud,o3d.utility.Vector2iVector(corres_mat))
    ErrorCalc_vec = []
    Src_CtPts = np.asarray(CT_Space_pointCloud.points)
    for i in range(len(Src_CtPts)):
        errCal_vec = list(Src_CtPts[i])
        errCal_vec.append(1)
        errCal_vec2 = np.dot(transformation_mat,errCal_vec)
        errCal_vec3 = np.asarray(errCal_vec2[0:3])
        ErrorCalc_vec.append(errCal_vec3)
    ErrorCalc = o3d.geometry.PointCloud()
    ErrorCalc.points = o3d.utility.Vector3dVector(ErrorCalc_vec)
    error = p2p.compute_rmse(ErrorCalc,Robot_Space_pointCloud,o3d.utility.Vector2iVector(corres_mat))
    print("Registration Error : {}".format(error))
    return transformation_mat,error
    