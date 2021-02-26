import open3d as o3d
import numpy as np
import sys, os
import re
import math 
import copy
import cv2

from functions import *

folder_path = "C:\\Users\\vinic\\Desktop\\CAPDesktop\\objetos\\gerador\\scan1"
ignored_files = ["acumulada", "mesh", "panoramica", "planta_baixa"]
voxel_size_lr = 0.03
voxel_size_fr = 0.005
dobj = 10
debug = True
preprocess = False

# Detalhes da camera
k = np.array([[978.34, -0.013, 654.28], [0.054, 958.48, 367.49], [0, 0, 1]])
Tcam = np.array([[1, 0, 0, -0.011], [0, 1, 0, 0.029], [0, 0, 1, 0.027], [0, 0, 0, 1]])

print("Getting files list ...")
clouds_list = get_file_list(folder_path, ignored_files, extension=".ply")
images_list = get_file_list(folder_path, ignored_files, extension=".png")
if len(clouds_list) != len(images_list):
    raise Exception("The number of images must match the one of images!")

print("Full registration through odometry ...")

print("Starting with first cloud and beginning registration ...")
acc = o3d.geometry.PointCloud()
if preprocess:
    acc = load_filter_point_cloud(clouds_list[0], voxel_size=voxel_size_fr, depth_max=dobj)
    #acc = color_point_cloud(images_list[0], acc, k, Tcam[0:3, 3])
    o3d.io.write_point_cloud(clouds_list[0], acc)
else:
    acc = o3d.io.read_point_cloud(clouds_list[0])

acc_reg = acc.voxel_down_sample(voxel_size=voxel_size_lr)
acc_reg = filter_depth(acc_reg, dmax=dobj)
acc_reg = remove_floor(acc_reg, height=1)

transforms_list = []
odometry = np.identity(4, float)
last_transf = np.identity(4, float)
transforms_list.append(odometry)

for id in range(1, len(clouds_list)):
    print(f"Working with point cloud {id+1:d} out of {len(clouds_list):d} in Total ...")

    # Ler a nuvem atual e leva-la para a referencia 
    src = o3d.geometry.PointCloud()   
    if preprocess:
        print("Loading and preprocessing ...")
        src = load_filter_point_cloud(clouds_list[id], voxel_size=voxel_size_fr, depth_max=dobj)
        #src = color_point_cloud(images_list[id], src, k, Tcam[0:3, 3])        
        o3d.io.write_point_cloud(clouds_list[id], src)
    else:
        print("Loading ...")
        src = o3d.io.read_point_cloud(clouds_list[id])
    

    # Ver o incremento que resta para encaixar a nuvem
    if not preprocess:

        src_reg = src.voxel_down_sample(voxel_size=voxel_size_lr)
        src_reg = filter_depth(src_reg, dmax=dobj)
        src_no_floor = remove_floor(src_reg, height=1)

        src.transform(odometry)
        src_reg.transform(odometry)
        src_no_floor.transform(odometry)

        print("Registering ...")  
        angle_in_this_iteration = 80
        intensity_base = 6
        repeat_base = 5
        use_features = False
        repetitions_count = 1
        while angle_in_this_iteration > 45 and repetitions_count < 6:
            transf, info = pairwise_registration(src_no_floor, acc_reg, voxel_size_lr, 
                                                 intensity=intensity_base, repeat=repeat_base, 
                                                 use_features=use_features)
            angle_in_this_iteration = check_angle(np.identity(4, float), transf)
            print(f"Rotation angle in this iteration: {angle_in_this_iteration:.2f} .")
            intensity_base += 1
            repeat_base += 1
            #use_features = False if use_features else True
            repetitions_count += 1

        if repetitions_count < 6:
    
            # Transformar a nuvem com nova aproximacao
            src.transform(transf)    
            src_reg.transform(transf)
            src_no_floor.transform(transf)
    
            # Adicionar na acumulada removendo pontos repetidos
            print("Adding by removing repeated points ...")
            if id % 1 == 0:
                acc += remove_existing_points(src, acc, 3*voxel_size_fr)
            acc_reg += remove_existing_points(src_no_floor, acc_reg, 0.5*voxel_size_lr)
            temp = acc_reg.voxel_down_sample(voxel_size=voxel_size_lr)
            acc_reg = copy.deepcopy(temp)

            if debug:
                print("Analyzing result ...")
                o3d.visualization.draw_geometries([acc_reg, src_no_floor], zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172,  2.0475,  1.5320], up=[-0.0694, -0.9768, 0.2024])
    
            o3d.io.write_point_cloud(os.path.join(folder_path, "acumulada.ply"), acc)

            # Adicionar na odometria esse incremento
            print("Updating odometry record and moving on ...")
            last_transf = transf
            odometry = np.dot(transf, odometry)    
            # Salvar a odometria na lista
            transforms_list.append(odometry)

        else:

            clouds_list[id] = []
  
if not preprocess:

    accd = acc.voxel_down_sample(voxel_size=voxel_size_fr)

    print("Saving ...")
    o3d.io.write_point_cloud(os.path.join(folder_path, "acumulada.ply"), accd)

    print("Display!")
    acc_reg.translate(np.array([0, 20, 0]))
    o3d.visualization.draw_geometries([accd, acc_reg], zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172, 2.0475, 1.532], up=[-0.0694, -0.9768, 0.2024])

    print("Creating SFM file ...")
    create_sfm_file(os.path.join(folder_path, "cameras_opt.sfm"), clouds_list, transforms_list, k, Tcam, only_write=False)

    print("Creating mesh ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(accd, depth=12)
    ## Filtrar por densidades
    densities = np.asarray(densities)
    densities = (densities - densities.min()) / (densities.max() - densities.min())
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    o3d.io.write_triangle_mesh(os.path.join(folder_path, "mesh.ply"), mesh)
    o3d.visualization.draw_geometries([mesh], zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172, 2.0475, 1.532], up=[-0.0694, -0.9768, 0.2024])