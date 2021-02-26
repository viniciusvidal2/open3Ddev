# examples/Python/Advanced/o3d.color_map.color_map_optimization.py

import open3d as o3d
from trajectory_io import *
from trajectory import *
import matplotlib.pyplot as plt
import os, sys
import re
import cv2

path = "C:/Users/vinic/Desktop/Reconstruction/reconstruction_system/dataset/pessoa"
debug_mode = False

def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)

def get_file_list(path, extension=None):
    if extension is None:
        file_list = [
            path + f for f in os.listdir(path) if os.path.isfile(join(path, f))
        ]
    else:
        file_list = [
            path + f
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and
            os.path.splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list

def read_sfm(name):
    file = open(name, 'r')
    return file.readlines()[2:]

if __name__ == "__main__":
    #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    # Lendo o arquivo sfm
    #lines = read_sfm(os.path.join(path, "cameras_ok2.sfm"))

    ## Nosso parametro intrinseco
    #our_w, our_h = 1920, 1080
    ##our_intrinsic = np.array([[1427.1, -0.063, our_w/2],[0.041, 1449.4, our_h/2],[0, 0, 1]])
    #our_intrinsic = np.array([[1427.1, -0.063, 987.9],[0.041, 1449.4, 579.4],[0, 0, 1]])
    #fx, fy = our_intrinsic[0,0], our_intrinsic[1,1]
    #cx, cy = our_intrinsic[0,2], our_intrinsic[1,2]
    #scale = 1000
    #intrinsic = o3d.camera.PinholeCameraIntrinsic(our_w, our_h, fx, fy, cx, cy)
    # Dados intrinsecos da camera astra
    w, h = 640, 480
    K = np.array([[570.3422241210938, 0, 319.5],[0, 570.3422241210938, 239.5],[0, 0, 1]])
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    scale = 1000
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    intrinsic.intrinsic_matrix = K

    # Read RGBD images
    rgbd_images = []
    depth_image_path = get_file_list(os.path.join(path, "depth2/"), extension=".png")
    color_image_path = get_file_list(os.path.join(path, "image2/"), extension=".png")
    assert (len(depth_image_path) == len(color_image_path))
    print('Lendo imagens ...')
    for i in range(len(depth_image_path)):
        depth = o3d.io.read_image(depth_image_path[i])
        color = o3d.io.read_image(color_image_path[i])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=scale, depth_trunc=50, convert_rgb_to_intensity=False)
        if debug_mode:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            o3d.visualization.draw_geometries([pcd])
        rgbd_images.append(rgbd_image)

    # Read camera pose and mesh
    #camera = o3d.camera.PinholeCameraTrajectory()    
    #parameters = []
    #for i,line in enumerate(lines):
    #    p = o3d.camera.PinholeCameraParameters()
    #    p.intrinsic = intrinsic
    #    l = line.split()
    #    T = np.array([[l[1], l[2], l[3], l[10]],
    #                  [l[4], l[5], l[6], l[11]],
    #                  [l[7], l[8], l[9], l[12]],
    #                  [0, 0, 0, 1]], dtype=np.float)
    #    p.extrinsic = T
    #    parameters.append(p)
    #camera.parameters = parameters
    camera = o3d.io.read_pinhole_camera_trajectory(os.path.join(path, "scene", "trajectory.log"))
    
    ptc = o3d.io.read_point_cloud(os.path.join(path, "scene", "integrated.ply"))
    #ptcv = ptc.voxel_down_sample(voxel_size=0.02)
    #ptcv.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    #ptcv.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=-1, max_nn=100))
    #ptcv.orient_normals_towards_camera_location()
    radii = [0.005, 0.01, 0.02]
    ptc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100))
    ptc.orient_normals_towards_camera_location()
    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(ptc, o3d.utility.DoubleVector(radii))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(ptc, depth=10)
    #mesh.filter_smooth_laplacian(200, 0.5)
    #mesh = o3d.io.read_triangle_mesh(os.path.join(path, "mesh_refined.ply"))
    #o3d.visualization.draw_geometries([mesh])
    #mesh.filter_smooth_laplacian(20, 0.5)

    # Before full optimization, let's just visualize texture map
    # with given geometry, RGBD images, and camera poses.
    option = o3d.pipelines.color_map.ColorMapOptimizationOption() 
    option.depth_threshold_for_discontinuity_check = 0.03
    option.depth_threshold_for_visibility_check = 0.03
    option.maximum_allowable_depth = 2

    option.maximum_iteration = 0    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.color_map.color_map_optimization(mesh, rgbd_images, camera, option)
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh(os.path.join(path, "scene", "color_map_before_optimization.ply"), mesh)

    option.maximum_iteration = 3000
    option.non_rigid_camera_coordinate = True
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.color_map.color_map_optimization(mesh, rgbd_images, camera, option)
    
    o3d.io.write_triangle_mesh(os.path.join(path, "scene", "color_map_after_optimization.ply"), mesh)
    mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([mesh])