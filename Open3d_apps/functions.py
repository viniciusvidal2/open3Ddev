import open3d as o3d
import numpy as np
import sys, os
import re
import math 
import copy
import cv2

#######################################################################################################
def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)
#######################################################################################################
def get_file_list(path, ignored_files, extension=None):
    if extension is None:
        file_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(join(path, f))]
    else:
        file_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) 
                     and os.path.splitext(f)[1] == extension 
                     and os.path.splitext(f)[0] != ignored_files[0]
                     and os.path.splitext(f)[0] != ignored_files[1]
                     and os.path.splitext(f)[0] != ignored_files[2]
                     and os.path.splitext(f)[0] != ignored_files[3]
                     and os.path.splitext(f)[0] != ignored_files[4]]
    file_list = sorted_alphanum(file_list)
    return file_list
#######################################################################################################
def filter_depth(cloud, dmax):
    cloud2 = o3d.geometry.PointCloud()
    for i, point in enumerate(cloud.points):
        d = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        if d < dmax:
            cloud2.points.append(cloud.points[i])
            cloud2.colors.append(cloud.colors[i])
            if len(cloud.normals) == len(cloud.points):
                cloud2.normals.append(cloud.normals[i])

    return cloud2
#######################################################################################################
## Retirar pontos ja existentes por vizinhanca
def remove_existing_points(src, tgt, radius):
    if len(tgt.points) == 0:
        return src

    s2 = o3d.geometry.PointCloud()
    dists = src.compute_point_cloud_distance(tgt)
    for i, d in enumerate(dists):
        if d > radius:
            s2.points.append(src.points[i])
            s2.colors.append(src.colors[i])
            s2.normals.append(src.normals[i])
    return s2
#######################################################################################################
def load_point_clouds(folder, final_name, voxel_size=0.0, depth_max=10):
    pcds = []
    cloud_paths = get_file_list(folder, final_name, extension=".ply")
    for i in range(len(cloud_paths)):
        print(f"Processing cloud {i+1:d} out of {len(cloud_paths):d} in Total ...")
        pcd = o3d.io.read_point_cloud(cloud_paths[i])
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.4)
        pcd_down2 = filter_depth(copy.deepcopy(pcd_down), depth_max)
        #pcd_down2 = raycasting(pcd_down2, 0.5, 22, 88, voxel_size, depth_max)
        if len(pcd_down2.points) > 100:
            pcd_down2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10*voxel_size, max_nn=250))
            pcd_down2.orient_normals_towards_camera_location()
            pcd_down2.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.4)
            pcds.append(pcd_down2)
        else:
            pcds.append(o3d.geometry.PointCloud())

    return pcds
#######################################################################################################
def load_filter_point_cloud(name, voxel_size=0.0, depth_max=10):
    pcd = o3d.io.read_point_cloud(name)
    pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down2 = filter_depth(copy.deepcopy(pcd_down), depth_max)
    #pcd_down2 = raycasting(pcd_down2, 0.5, 22, 88, voxel_size, depth_max)
    if len(pcd_down2.points) > 100:
        pcd_down2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5*voxel_size, max_nn=50))
        #pcd_down2.orient_normals_towards_camera_location()
    return pcd_down2
#######################################################################################################
def pairwise_registration(source, target, voxel_size, intensity=3, repeat=1, use_features=False, initial=np.identity(4, float)):
    
    Tsa = np.identity(4, float)

    if use_features:
        radius_feature = 6*voxel_size
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200))
        result_fast = o3d.pipelines.registration.registration_fast_based_on_feature_matching(source, target, source_fpfh, target_fpfh, 
                                                                                             o3d.pipelines.registration.FastGlobalRegistrationOption(
                                                                                                 maximum_correspondence_distance=0.03))
        #result_fast = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source, target, source_fpfh, target_fpfh, max_correspondence_distance=1.5*voxel_size,
        #                                                                                       checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold=voxel_size)])
        Tsa = result_fast.transformation
    else:
        Tsa = initial

    dist_min_icp = intensity*voxel_size
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-11, relative_rmse=1e-11, max_iteration=100) 
    #o3d.visualization.draw_geometries([source, target], zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172,  2.0475,  1.5320], up=[-0.0694, -0.9768, 0.2024])
    
    for i in range(intensity-1):
        icp_fine = o3d.pipelines.registration.registration_colored_icp(source, target, dist_min_icp, Tsa, o3d.pipelines.registration.TransformationEstimationForColoredICP(), criteria)
        #icp_fine = o3d.pipelines.registration.registration_icp(source, target, dist_min_icp, Tsa, o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
        Tsa = icp_fine.transformation
        dist_min_icp -= voxel_size
        #teste = copy.deepcopy(source)
        #o3d.visualization.draw_geometries([teste.transform(Tsa), target], zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172,  2.0475,  1.5320], up=[-0.0694, -0.9768, 0.2024])
    

    for i in range(repeat):
        icp_fine = o3d.pipelines.registration.registration_colored_icp(source, target, dist_min_icp, Tsa, o3d.pipelines.registration.TransformationEstimationForColoredICP(), criteria)
        #icp_fine = o3d.pipelines.registration.registration_icp(source, target, dist_min_icp, Tsa, o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
        Tsa = icp_fine.transformation
    
    transformation_icp = icp_fine.transformation
   
    return Tsa
#######################################################################################################
def full_registration(pcds, voxel_size):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for target_id in range(n_pcds):
        for source_id in range(target_id + 1, n_pcds):
            #transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id], voxel_size)
            #print("Build o3d.pipelines.registration.PoseGraph")
            if source_id == target_id + 1:  # odometry case
                transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id], voxel_size)
                print("Build o3d.pipelines.registration.PoseGraph")
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append( o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)) )
                pose_graph.edges.append( o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transformation_icp, information_icp, uncertain=False))
            #else:  # loop closure case
            #    pose_graph.edges.append( o3d.pipelines.registration.PoseGraphEdge(source_id,
            #                                                 target_id,
            #                                                 transformation_icp,
            #                                                 information_icp,
            #                                                 uncertain=True))
    return pose_graph
#######################################################################################################
def color_point_clouds(folder_path, clouds, k, t):
    clouds2 = []
    images = get_file_list(folder_path, extension=".png")

    t = t[:, np.newaxis]

    if len(images) != len(clouds):
        raise Exception("Deve haver mesmo numero de imagens e nuvens no diretorio, observe.")
        return 
    
    for im_id, cloud in enumerate(clouds):
        cloud2 = copy.deepcopy(cloud)
        im = cv2.imread(images[im_id])
        for id, p in enumerate(cloud.points):
            pT = p[:, np.newaxis]
            pixel_homogeneous = np.dot(k, (pT+t))
            pixel_homogeneous = (pixel_homogeneous/pixel_homogeneous[2]).astype(int)
            if 0 <= pixel_homogeneous[1][0] < im.shape[0] and 0 <= pixel_homogeneous[0][0] < im.shape[1]:
                pixel_color = (im[pixel_homogeneous[1][0], pixel_homogeneous[0][0]]).astype(float)/255
                cloud2.colors[id] = pixel_color[::-1]
        clouds2.append(cloud2)

    return clouds2
#######################################################################################################
def color_point_cloud(image_path, cloud, k, t):
  
    t = t[:, np.newaxis]
    cloud2 = copy.deepcopy(cloud)
    im = cv2.imread(image_path)
    for id, p in enumerate(cloud.points):
        pT = p[:, np.newaxis]
        pixel_homogeneous = np.dot(k, (pT+t))
        pixel_homogeneous = (pixel_homogeneous/pixel_homogeneous[2]).astype(int)
        if 0 <= pixel_homogeneous[1][0] < im.shape[0] and 0 <= pixel_homogeneous[0][0] < im.shape[1]:
            pixel_color = (im[pixel_homogeneous[1][0], pixel_homogeneous[0][0]]).astype(float)/255
            cloud2.colors[id] = pixel_color[::-1]

    return cloud2
#######################################################################################################
def raycasting(cloud, step, fov_lat, fov_lon, voxel_size, max_depth):
    cloud2 = o3d.geometry.PointCloud()
    # Campo de visao para procurar - angulos
    lats = math.pi/180*np.arange(-fov_lat/2, fov_lat/2, step)
    lons = math.pi/180*np.arange(-fov_lon/2, fov_lon/2, step)
    # Criar os vetores de direcao e buscar com Kdtree
    dists = np.arange(2, max_depth, voxel_size)
    tree = o3d.geometry.KDTreeFlann(cloud)
    for lat in lats:
        for lon in lons:
            vec = np.array([np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)])
            # Para cada direcao, seguir com pontos 3D nessa linha
            for d in dists:
                # Testar se ha vizinhos na nuvem para aquele ponto, se sim adicionar o ponto na nova nuvem                
                [points, indices, _] = tree.search_radius_vector_3d(d*vec, 2*voxel_size)
                if len(indices) >= 1:
                    cloud2.points.append(cloud.points[indices[0]])
                    cloud2.colors.append(cloud.colors[indices[0]])
                    break

    return cloud2
#######################################################################################################
def create_sfm_file(name, images_list, Ts, k=np.identity(3, float), Tcam=np.identity(4, float), only_write=False):
    fx = np.asarray(k)[0][0]
    fy = np.asarray(k)[1][1]
    cx = np.asarray(k)[0][2] 
    cy = np.asarray(k)[1][2]

    sfm = open(name, 'w')
    sfm.write(str(len(Ts))+"\n\n")
    
    pose = np.identity(4, float)
    for i, T in enumerate(Ts):
        if only_write:
            pose = T
        else:
            pose = np.matmul(Tcam, np.linalg.inv(T))

        linha  = images_list[i] + " "
        linha += str(np.asarray(pose)[0][0]) + " " + str(np.asarray(pose)[0][1]) + " " + str(np.asarray(pose)[0][2]) + " "
        linha += str(np.asarray(pose)[1][0]) + " " + str(np.asarray(pose)[1][1]) + " " + str(np.asarray(pose)[1][2]) + " "
        linha += str(np.asarray(pose)[2][0]) + " " + str(np.asarray(pose)[2][1]) + " " + str(np.asarray(pose)[2][2]) + " "
        linha += str(np.asarray(pose)[0][3]) + " " + str(np.asarray(pose)[1][3]) + " " + str(np.asarray(pose)[2][3]) + " "
        linha += str(fx) + " " + str(fy) + " " + str(cx) + " " + str(cy) + "\n"

        sfm.write(linha)

    sfm.close()
#######################################################################################################
def read_sfm_file(name):
    file = open(name, 'r')
    lines = file.readlines()[2:]

    transforms = []
    for n,line in enumerate(lines):
        l = line.split()
        T = np.array([[l[1], l[2], l[3], l[10]],
                      [l[4], l[5], l[6], l[11]],
                      [l[7], l[8], l[9], l[12]],
                      [0, 0, 0, 1]], dtype=np.float)
        transforms.append(T)

    return transforms
#######################################################################################################
def remove_floor(cloud, height=10):
    cloud2 = o3d.geometry.PointCloud()
    for i, p in enumerate(cloud.points):
        if p[1] <= height:
            cloud2.points.append(cloud.points[i])
            cloud2.colors.append(cloud.colors[i])
            cloud2.normals.append(cloud.normals[i])

    return cloud2
#######################################################################################################
def check_angle(T1, T2):    
    z = np.array([[0],[0],[1]], dtype=np.float)
    v1 = T1[0:3, 0:3].dot(z)/np.linalg.norm(z)
    v2 = T2[0:3, 0:3].dot(z)/np.linalg.norm(z)
    v1_ = np.squeeze(np.asarray(v1))
    v2_ = np.squeeze(np.asarray(v2))
    a = np.arccos( v1_.dot(v2_) )
    b = abs(np.rad2deg(a))

    return b
#######################################################################################################