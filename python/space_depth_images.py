import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
import multiprocessing
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

## PATH
path  = "C:/Users/vinic/Desktop/SANTOS_DUMONT_2/sala/scan2"

def custom_draw_geometry_with_camera_trajectory(mesh):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory = []
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()

    filec = open(os.path.join(path, 'cameras_ok2.sfm'), 'r')
    linhas = filec.readlines()[2:]
    scale = 1000

    # Trajetoria da camera
    w, h = 1920, 1080
    K = np.array([[1427.1, 0, w/2-0.5],[0, 1449.4, h/2-0.5],[0, 0, 1]])
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    trajectory = o3d.camera.PinholeCameraTrajectory()
    parameters = []
    for n,line in enumerate(linhas):
        parameter = o3d.camera.PinholeCameraParameters()
        l = line.split()
        T = np.array([[l[1], l[2], l[3], l[10]],
                      [l[4], l[5], l[6], l[11]],
                      [l[7], l[8], l[9], l[12]],
                      [0, 0, 0, 1]], dtype=np.float)
        parameter.extrinsic = T
        parameter.intrinsic.set_intrinsics(w,h,fx,fy,cx,cy)
        parameter.intrinsic.intrinsic_matrix = K
        parameters.append(parameter)
    trajectory.parameters = parameters
    custom_draw_geometry_with_camera_trajectory.trajectory = trajectory

    def move_forward(vis):
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            depth = np.array(depth, dtype=np.float)*scale
            cv2.imwrite(path+"/depth/{:05d}.png".format(glb.index), depth.astype(np.uint16))
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(glb.trajectory.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.register_animation_callback(None)
        return False    

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window(window_name='camera', width=w, height=h)
    vis.add_geometry(mesh)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()

print('Lendo mesh ...')
#mesh = o3d.io.read_triangle_mesh(os.path.join(path, "poisson.ply"))
ptc = o3d.io.read_point_cloud(os.path.join(path, "acumulada.ply"))
ptcv = ptc.voxel_down_sample(voxel_size=0.02)
ptcv.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
ptcv.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=-1, max_nn=100))
ptcv.orient_normals_towards_camera_location()
radii = [0.005, 0.01, 0.02, 0.04]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(ptcv, o3d.utility.DoubleVector(radii))
mesh.filter_smooth_laplacian(200, 0.5)
o3d.io.write_triangle_mesh(os.path.join(path, "mesh_refined.ply"), mesh)

custom_draw_geometry_with_camera_trajectory(mesh)