# examples/Python/Advanced/o3d.color_map.color_map_optimization.py

import open3d as o3d
from trajectory_io import *
from trajectory import *
import os, sys
import re

path = "C:/Users/vinic/Desktop/Reconstruction/reconstruction_system/dataset/violao1"
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

if __name__ == "__main__":
    #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    # Read RGBD images
    rgbd_images = []
    depth_image_path = get_file_list(os.path.join(path, "depth/"), extension=".png")
    color_image_path = get_file_list(os.path.join(path, "image/"), extension=".png")
    assert (len(depth_image_path) == len(color_image_path))
    for i in range(len(depth_image_path)):
        depth = o3d.io.read_image(os.path.join(depth_image_path[i]))
        color = o3d.io.read_image(os.path.join(color_image_path[i]))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,convert_rgb_to_intensity=False)
        if debug_mode:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
            o3d.visualization.draw_geometries([pcd])
        rgbd_images.append(rgbd_image)

    # Read camera pose and mesh
    camera = o3d.io.read_pinhole_camera_trajectory(os.path.join(path, "scene", "trajectory.log"))

    #pc = o3d.geometry.PointCloud()
    #lk = o3d.geometry.PointCloud()
    #for n,o in enumerate(camera.parameters):
    #    e = np.linalg.inv(o.extrinsic)
    #    l=np.array([e[0:3, 3]]).T
    #    pc.points.append(l)
    #    lk.points.append(l+e[0:3,0:3].dot(np.array([[0,0,1]]).T))
    #pc.paint_uniform_color(np.array([255, 0, 0]))
    #lk.paint_uniform_color(np.array([255, 0, 230]))
    #o3d.io.write_point_cloud(os.path.join(path, "scene", "path.ply"), pc)
    #o3d.io.write_point_cloud(os.path.join(path, "scene", "look.ply"), lk)

    mesh = o3d.io.read_triangle_mesh(os.path.join(path, "scene", "integrated.ply"))
    #o3d.visualization.draw_geometries([mesh])

    # Before full optimization, let's just visualize texture map
    # with given geometry, RGBD images, and camera poses.
    option = o3d.pipelines.color_map.ColorMapOptimizationOption() 

    #print('Rodando cor inicial sem otimizar ...')
    #option.maximum_iteration = 0    
    ##with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #o3d.pipelines.color_map.color_map_optimization(mesh, rgbd_images, camera, option)
    ##o3d.visualization.draw_geometries([mesh])
    #o3d.io.write_triangle_mesh(os.path.join(path, "scene", "color_map_before_optimization.ply"), mesh)

    option.maximum_iteration = 300
    print('Otimizando com {} ...'.format(option.maximum_iteration))
    option.non_rigid_camera_coordinate = True
    #with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.color_map.color_map_optimization(mesh, rgbd_images, camera, option)
    
    mesh.filter_smooth_laplacian(100, 0.5)
    o3d.io.write_triangle_mesh(os.path.join(path, "scene", "color_map_after_optimization.ply"), mesh)
    mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([mesh])
