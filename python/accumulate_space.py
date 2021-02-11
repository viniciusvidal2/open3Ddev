import open3d as o3d
import os, sys, shutil
import re
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

def pairwise_registration(source, target, dist_lim):
    max_correspondence_distance_fine = dist_lim
    #r = 0.1
    # FPFH features
    #src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=100))
    #tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=100))
    ## FAST parameters
    #option = o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=max_correspondence_distance_coarse)
    ## Estimate FAST transform
    #result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(source, target, src_fpfh, tgt_fpfh, option)
    # Refine colored ICP
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=100)
    icp_fine = o3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance_fine, np.identity(4, float), 
                                                           o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, max_correspondence_distance_fine, icp_fine.transformation)
    return transformation_icp, information_icp

# Function to return the absolute angle difference between 2 vectors in Degrees
def check_angle(T1, T2):    
    z = np.array([[0],[0],[1]], dtype=np.float)
    v1 = T1[0:3, 0:3].dot(z)/np.linalg.norm(z)
    v2 = T2[0:3, 0:3].dot(z)/np.linalg.norm(z)
    v1_ = np.squeeze(np.asarray(v1))
    v2_ = np.squeeze(np.asarray(v2))
    a = np.arccos( v1_.dot(v2_) )
    b = abs(np.rad2deg(a))
    return b

def filter_depth(cloud, dmax):
    cloud2 = o3d.geometry.PointCloud()
    for i, point in enumerate(cloud.points):
        d = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        if d < dmax and cloud.colors[i][2] < 0.85:
            cloud2.points.append(cloud.points[i])
            cloud2.colors.append(cloud.colors[i])
    return cloud2

def filter_color(cloud):
    cloud2 = o3d.geometry.PointCloud()
    for i, point in enumerate(cloud.colors):
        if cloud.colors[i][0] > 0.1 and cloud.colors[i][1] > 0.1 and cloud.colors[i][2] < 0.9:
            cloud2.points.append(cloud.points[i])
            cloud2.colors.append(cloud.colors[i])
    return cloud2

## Retirar pontos ja existentes por vizinhanca
def remove_existing_points(src, tgt, radius):
    s2 = o3d.geometry.PointCloud()
    dists = src.compute_point_cloud_distance(tgt)
    for i, d in enumerate(dists):
        if d > radius:
            s2.points.append(src.points[i])
            s2.colors.append(src.colors[i])
            s2.normals.append(src.normals[i])
    return s2

def create_plane(a, b, c, d, lim, res):
    plane = o3d.geometry.PointCloud()
    # Get plane points
    p1 = np.array([   0, -d/b,    0])
    p2 = np.array([   0,    0, -d/c])
    p3 = np.array([-d/a,    0,    0])
    # Get the vectors in the plane
    v1 = p2 - p1
    v2 = p3 - p1
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    for x in np.arange(-lim, lim, res):
        for z in np.arange(-lim, lim, res):
            point = p1 + x*v1 + z*v2
            plane.points.append(point)
            plane.colors.append(np.array([0, 1, 0]))
    plane.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=200))

    return plane

##############################################################################################################################
##############################################################################################################################

### Read corresponding folder
###
path = "C:/Users/vinic/Desktop/transformadores_sd/scan5"
sfm = os.path.join(path,"cameras.sfm")
vs = 0.04
depth_max = 30

### Read SFM file, mainly the orientations
###
file = open(sfm, 'r')
lines = file.readlines()[2:]

### Check which clouds really exist, point to corresponding orientation
###
path2clouds = []
cloud_orientations = []
reference_cloud_index = 0

for i in range(3, len(lines)+1, 5):
    # Find if the cloud was captured or not
    name = "0";
    if i < 10:
        name = "c_00"+str(i)+".ply"
    elif i < 100:
        name = "c_0" +str(i)+".ply"
    name = os.path.join(path, name)
    # If the file exists, save data
    try:
        f = open(name)
        path2clouds.append(name) # Save cloud path
        l = lines[i-1].split() # Get odometry and record it to corresponding cloud
        T = np.array([[l[1], l[2], l[3], l[10]],
                      [l[4], l[5], l[6], l[11]],
                      [l[7], l[8], l[9], l[12]],
                      [0   , 0   , 0   , 1   ]], dtype=np.float)
        #cloud_orientations.append(np.linalg.inv(T))
        cloud_orientations.append(T)
        # Check if this is the reference cloud (looking horizontal towards Z axis)
        if check_angle(T, np.identity(4, dtype=float)) < 10:
            reference_cloud_index = i

    except IOError:
        print("NO CLOUD FOR THIS POINT OF VIEW, BUT THATS OK!")
        continue

### Accumulate horizontal by following the plane model
###
accumulated = o3d.geometry.PointCloud()
plane = o3d.geometry.PointCloud()
plane_model_floor = []
criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=100)
for i, cloud_name in enumerate(path2clouds):
    
    source = o3d.io.read_point_cloud(cloud_name)
    sourcev = source.voxel_down_sample(voxel_size=vs)
    sourcev2 = filter_depth(sourcev, depth_max)
    if len(sourcev2.points) > 100:
        sourcev2 = filter_color(sourcev2)
        sourcev2.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
        sourcev2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=200))
        sourcev2.orient_normals_towards_camera_location()

        # Find floor plane
        plane_model, inliers = sourcev2.segment_plane(distance_threshold=0.05, ransac_n=10, num_iterations=1000)
        [a, b, c, d] = plane_model            
        plane_normal = np.array([a, b, c])
        cloud_plane = sourcev2.select_by_index(inliers)
        not_plane = sourcev2.select_by_index(inliers, invert=True)
        while abs(plane_normal.dot(np.array([0, 1, 0]))/np.linalg.norm(plane_normal)) < 0.9:
            plane_model, inliers = not_plane.segment_plane(distance_threshold=0.05, ransac_n=10, num_iterations=1000)
            [a, b, c, d] = plane_model                
            plane_normal = np.array([a, b, c])
            cloud_plane = not_plane.select_by_index(inliers)
            not_plane = not_plane.select_by_index(inliers, invert=True)      
        cloud_plane_print = copy.deepcopy(cloud_plane)
        cloud_plane_print.paint_uniform_color([1.0, 0, 0])

        print(f"Align cloud number {i+1:d} ...")    

        if i == 0:

            accumulated = copy.deepcopy(sourcev2)
            plane = create_plane(a, b, c, d, depth_max, 3*vs)
            print(f"Equation of reference plane: a={a:.2f} b={b:.2f} c={c:.2f} d={d:.2f} ")

        else:           

            dist_min_icp = 0.70
            for i in range(11):              
                icp_fine = o3d.pipelines.registration.registration_icp(cloud_plane, plane, dist_min_icp, np.identity(4, float), o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
                Tplane = icp_fine.transformation
                if check_angle(Tplane, np.identity(4, float)) < 15:
                    cloud_plane.transform(Tplane)
                    cloud_plane_print.transform(Tplane)
                    not_plane.transform(Tplane)
                    sourcev2.transform(Tplane)
                dist_min_icp -= 0.06

            dist_min_icp = 0.05
            for i in range(3):
                icp_fine = o3d.pipelines.registration.registration_icp(sourcev2, accumulated, dist_min_icp, np.identity(4, float), o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
                Tsa = icp_fine.transformation
                if check_angle(Tsa, np.identity(4, float)) < 15:
                    cloud_plane.transform(Tsa)
                    cloud_plane_print.transform(Tsa)
                    not_plane.transform(Tsa)
                    sourcev2.transform(Tsa)
                dist_min_icp -= 0.01

            sourcev2 = remove_existing_points(sourcev2, accumulated, 1.5*vs)
            accumulated += sourcev2

### Accumulate looking down
### 
path2clouds = []
cloud_indexes = np.append(np.arange(2, len(lines)+1, 10), np.arange(9, len(lines)+1, 10))
for i in cloud_indexes:
    # Find if the cloud was captured or not
    name = "0";
    if i < 10:
        name = "c_00"+str(i)+".ply"
    elif i < 100:
        name = "c_0" +str(i)+".ply"
    name = os.path.join(path, name)
    # If the file exists, save data
    try:
        f = open(name)
        path2clouds.append(name) # Save cloud path
        l = lines[i-1].split() # Get odometry and record it to corresponding cloud
        T = np.array([[l[1], l[2], l[3], l[10]],
                      [l[4], l[5], l[6], l[11]],
                      [l[7], l[8], l[9], l[12]],
                      [0   , 0   , 0   , 1   ]], dtype=np.float)
        cloud_orientations.append(T)

    except IOError:
        print("NO CLOUD FOR THIS POINT OF VIEW, BUT THATS OK!")
        continue

for i, cloud_name in enumerate(path2clouds):
    source = o3d.io.read_point_cloud(cloud_name)
    sourcev = source.voxel_down_sample(voxel_size=vs)
    sourcev2 = filter_depth(sourcev, depth_max)
    if len(sourcev2.points) > 100:
        sourcev2 = filter_color(sourcev2)
        sourcev2.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
        sourcev2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=-1, max_nn=100))
        sourcev2.orient_normals_towards_camera_location()

        # Find floor plane
        plane_model, inliers = sourcev2.segment_plane(distance_threshold=0.05, ransac_n=10, num_iterations=1000)
        [a, b, c, d] = plane_model            
        plane_normal = np.array([a, b, c])
        cloud_plane = sourcev2.select_by_index(inliers)
        not_plane = sourcev2.select_by_index(inliers, invert=True)
        while abs(plane_normal.dot(np.array([0, 1, 0]))/np.linalg.norm(plane_normal)) < 0.9:
            plane_model, inliers = not_plane.segment_plane(distance_threshold=0.05, ransac_n=10, num_iterations=1000)
            [a, b, c, d] = plane_model                
            plane_normal = np.array([a, b, c])
            cloud_plane = not_plane.select_by_index(inliers)
            not_plane = not_plane.select_by_index(inliers, invert=True)      
        cloud_plane_print = copy.deepcopy(cloud_plane)
        cloud_plane_print.paint_uniform_color([1.0, 0, 0])

        print(f"Align down cloud number {i+1:d} ...") 
        
        dist_min_icp = 0.20
        for i in range(15):
            icp_fine = o3d.pipelines.registration.registration_icp(cloud_plane, plane, dist_min_icp, np.identity(4, float), o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
            Tplane = icp_fine.transformation
            if check_angle(Tplane, np.identity(4, float)) < 15:
                cloud_plane.transform(Tplane)
                cloud_plane_print.transform(Tplane)
                not_plane.transform(Tplane)
                sourcev2.transform(Tplane)
            dist_min_icp -= 0.01
        #sourcev2 = remove_existing_points(sourcev2, accumulated, 1.5*vs)
        accumulated += sourcev2

### Accumulate looking up
### 
path2clouds = []
cloud_indexes = np.append(np.arange(4, len(lines)+1, 10), np.arange(7, len(lines)+1, 10))
for i in cloud_indexes:
    # Find if the cloud was captured or not
    name = "0";
    if i < 10:
        name = "c_00"+str(i)+".ply"
    elif i < 100:
        name = "c_0" +str(i)+".ply"
    name = os.path.join(path, name)
    # If the file exists, save data
    try:
        f = open(name)
        path2clouds.append(name) # Save cloud path
        l = lines[i-1].split() # Get odometry and record it to corresponding cloud
        T = np.array([[l[1], l[2], l[3], l[10]],
                      [l[4], l[5], l[6], l[11]],
                      [l[7], l[8], l[9], l[12]],
                      [0   , 0   , 0   , 1   ]], dtype=np.float)
        cloud_orientations.append(T)

    except IOError:
        print("NO CLOUD FOR THIS POINT OF VIEW, BUT THATS OK!")
        continue

for i, cloud_name in enumerate(path2clouds):
    source = o3d.io.read_point_cloud(cloud_name)
    sourcev = source.voxel_down_sample(voxel_size=vs)
    sourcev2 = filter_depth(sourcev, depth_max)
    if len(sourcev2.points) > 100:
        sourcev2 = filter_color(sourcev2)
        sourcev2.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
        sourcev2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=200))
        sourcev2.orient_normals_towards_camera_location()
        
        print(f"Align up cloud number {i+1:d} ...") 

        dist_min_icp = 0.50
        for i in range(9):
            icp_fine = o3d.pipelines.registration.registration_icp(sourcev2, accumulated, dist_min_icp, np.identity(4, float), o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
            Tsa = icp_fine.transformation
            if check_angle(Tsa, np.identity(4, float)) < 15:
                cloud_plane.transform(Tsa)
                cloud_plane_print.transform(Tsa)
                not_plane.transform(Tsa)
                sourcev2.transform(Tsa)
            dist_min_icp -= 0.05

        sourcev2 = remove_existing_points(sourcev2, accumulated, 1.5*vs)
        accumulated += sourcev2


print("Saving final result ...")
accumulated.voxel_down_sample(voxel_size=vs)
o3d.io.write_point_cloud(os.path.join(path, "acumulada.ply"), accumulated)
o3d.visualization.draw_geometries([accumulated, plane])

#### Initialize pose graph
####
#pose_graph = o3d.pipelines.registration.PoseGraph()
##odometry = np.identity(4, dtype=float)
##pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

#### Add each pose as a node
####
#referenced_cloud_orientations = []
#for t in cloud_orientations:
#    # Refer to the origin node
#    t_ = t.dot(np.linalg.inv(cloud_orientations[reference_cloud_index]))
#    referenced_cloud_orientations.append(t_)

#    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode( t ))
#    #pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(t_))
#    #pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(cloud_orientations[reference_cloud_index]).dot(t)))

#### Double for, analyze if neighboring clouds according to orientation, and append new edge if true
####
#for source_id in range(len(path2clouds)):
#    for target_id in range(source_id + 1, len(path2clouds)):
#        # If angle is ok, we can consider them as neighbor acquisitions and perform the process
#        pp = check_angle(cloud_orientations[source_id], cloud_orientations[target_id])
#        if pp < 40:
#            print("Connecting edge from cloud {} to neighbor {} ...".format(source_id, target_id))
#            # Read and filter source cloud
#            source = o3d.io.read_point_cloud(path2clouds[source_id])
#            sourcev = source.voxel_down_sample(voxel_size=vs)
#            sourcev2 = filter_depth(sourcev, depth_max)
#            if len(sourcev2.points) > 100:
#                sourcev2 = filter_color(sourcev2)
#                sourcev2.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#                sourcev2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=-1, max_nn=100))
#                sourcev2.orient_normals_towards_camera_location()
#            # Read and filter target cloud
#            target = o3d.io.read_point_cloud(path2clouds[target_id])
#            targetv = target.voxel_down_sample(voxel_size=vs)
#            targetv2 = filter_depth(targetv, depth_max)
#            if len(targetv2.points) > 100:
#                targetv2 = filter_color(targetv2)
#                targetv2.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#                targetv2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=-1, max_nn=100))
#                targetv2.orient_normals_towards_camera_location()
            
#            if len(targetv2.points) > 100 and len(sourcev2.points) > 100:
#                # Relative transform between them
#                #Tst, information = pairwise_registration(sourcev, targetv)
#                #Tst = np.linalg.inv(cloud_orientations[source_id]).dot(cloud_orientations[target_id])
#                Tst = np.identity(4, dtype=float)
#                information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(sourcev2, targetv2, 2*vs, np.identity(4, dtype=float))

#                #temp = copy.deepcopy(source)
#                #temp.transform(Tst)
#                #o3d.visualization.draw_geometries([target, temp])

#                # Add the edge
#                #if target_id == source_id + 1:
#                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge( source_node_id=source_id, target_node_id=target_id, transformation=Tst, information=information, uncertain=True ))
#                #else:
#                #    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge( source_node_id=source_id, target_node_id=target_id, transformation=Tst, information=information, uncertain=False ))
#                # If its the following one, check precise transform and add to pode graph node
#                #if target_id == source_id + 1:
#                #    odometry = np.dot(Tst, odometry)
#                #    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))


#### Optimize graph
####
#print("Optimizing PoseGraph ...")
#option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=1.5*vs, edge_prune_threshold=10.25, reference_node=reference_cloud_index)
#method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
##method = o3d.pipelines.registration.GlobalOptimizationGaussNewton()
#criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
#criteria.lower_scale_factor = 1e-9
#criteria.max_iteration = 10000
#criteria.max_iteration_lm = 1000
#criteria.min_relative_increment = 1e-10
#criteria.min_relative_residual_increment = 1e-10
#criteria.min_residual = 1e-9

#with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)

#### Accumulate cloud, always looking for existing points - neighbors
####
#acc = o3d.geometry.PointCloud()
#for pc in range(len(path2clouds)):
#    cloud = o3d.io.read_point_cloud(path2clouds[pc])
#    cloudv = cloud.voxel_down_sample(voxel_size=vs)
#    cloudv.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#    cloudv.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=-1, max_nn=100))
#    cloudv.orient_normals_towards_camera_location()

#    #t_ = np.linalg.inv(referenced_cloud_orientations[pc])
#    #cloud.transform( t_ )
#    cloudv.transform( pose_graph.nodes[pc].pose )

#    acc += cloudv

#accv = acc.voxel_down_sample(voxel_size=vs)
#o3d.io.write_point_cloud(os.path.join(path, "acumulada.ply"), accv)
#o3d.visualization.draw_geometries([accv])