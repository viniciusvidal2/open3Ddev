import open3d as o3d
import os, sys, shutil
import re
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

def remove_existing_points(src, tgt, radius):
    s2 = o3d.geometry.PointCloud()
    dists = src.compute_point_cloud_distance(tgt)
    for i, d in enumerate(dists):
        if d > radius:
            s2.points.append(src.points[i])
            s2.colors.append(src.colors[i])
            s2.normals.append(src.normals[i])
    return s2

# Path to whole space acquisition
path = "C:/Users/vinic/Desktop/transformadores_sd_alinhados"

# Read relative transformation matrices file
file = open(os.path.join(path, "poses_scan.txt"), 'r')
lines = file.readlines()

# Read base space
l = lines[0].split()
base_space_name = os.path.join(path, l[1], "3DData/acumulada.ply")
space = o3d.io.read_point_cloud(base_space_name)

# For every other line in the file, read child space
for i, line in enumerate(lines):
    print(f"Working on cloud {i+1:d} of {len(lines):d} ...")
    next_space_name = os.path.join(path, l[0], "3DData/acumulada.ply")
    next_space = o3d.io.read_point_cloud(next_space_name)
    l = line.split()
    pose = np.array([[l[ 2], l[ 3], l[ 4], l[ 5]],
                     [l[ 6], l[ 7], l[ 8], l[ 9]],
                     [l[10], l[11], l[12], l[13]],
                     [0    , 0    , 0    , 1   ]], dtype=np.float)
    # Transform cloud to pose
    #next_space.transform(pose)
    # Approximate through ICP even further
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=100)
    icp_fine = o3d.pipelines.registration.registration_icp(next_space, space, 0.3, np.identity(4, float), 
                                                           o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
    #next_space.transform(icp_fine.transformation)
    # Remove existing points
    next_space_new = remove_existing_points(next_space, space, 0.25)
    #next_space_new = copy.deepcopy(next_space)
    next_space_new.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)
    # Add to space
    space += next_space_new
    #o3d.visualization.draw_geometries([space])

# Save final space
o3d.io.write_point_cloud(os.path.join(path, "space.ply"), space)
# Display result
o3d.visualization.draw_geometries([space])