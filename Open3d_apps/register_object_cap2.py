import open3d as o3d
import numpy as np
import sys, os
import re
import math as m
import copy
import cv2

from functions import *

# Pasta a trabalhar
folder_path = "C:\\Users\\vinic\\Desktop\\CAPDesktop\\objetos\\ugff\\scan4"
ignored_files = ["acumulada", "acumulada_opt", "mesh", "panoramica", "planta_baixa"]
# Parametros gerais
voxel_size_lr = 0.02
voxel_size_fr = 0.005
debug = True
preprocess = False
dobj = 60 if preprocess else 25

# Detalhes da camera
k = np.array([[978.34, -0.013, 654.28], [0.054, 958.48, 367.49], [0, 0, 1]])
Tcam = np.array([[1, 0, 0, -0.011], [0, 1, 0, 0.029], [0, 0, 1, 0.027], [0, 0, 0, 1]])

# Lendo lista de nuvens e imagens
print("Getting files list ...")
clouds_list = get_file_list(folder_path, ignored_files, extension=".ply")
images_list = get_file_list(folder_path, ignored_files, extension=".png")
if len(clouds_list) != len(images_list):
    raise Exception("The number of images must match the one of images!")

# Ler arquivo SFM, e ja inverter matrizes
print("Reading original LOAM sfm file ...")
loam_poses = read_sfm_file(os.path.join(folder_path, "cameras.sfm"))
for i, p in enumerate(loam_poses):
    loam_poses[i] = np.linalg.inv(p)

# Iniciar nuvem acumulada com a primeira leitura
print("Reading first cloud as accumulated ...");
acc = load_filter_point_cloud(clouds_list[0], voxel_size=voxel_size_fr, depth_max=20)
acc.orient_normals_towards_camera_location(np.linalg.inv(loam_poses[0])[0:3, 3])

# Para cada nuvem, performar
clouds = []
sfm_poses_list = []
sfm_poses_list.append(np.linalg.inv(loam_poses[0]))
last_icp_adjust = np.identity(4, float)
for i in range(1, len(clouds_list)):
    print(f"Registering point cloud {i+1:d} out of {len(clouds_list):d} ...")
    # Ler e pre processar nuvem
    src = load_filter_point_cloud(clouds_list[i], voxel_size=voxel_size_fr, depth_max=20)
    ##### TEMPORARIO: consertar orientacao das normais
    src.orient_normals_towards_camera_location(np.linalg.inv(loam_poses[i])[0:3, 3])
    #####
    # Transformar para o ultimo ajuste que fizemos por ICP
    src.transform(last_icp_adjust)
    # Simplificar as nuvens por voxel para aproximar por ICP mais rapido
    target = acc.voxel_down_sample(voxel_size_lr)
    source = src.voxel_down_sample(voxel_size_lr)
    transf = pairwise_registration(source=source, target=target, voxel_size=voxel_size_lr, intensity=2, repeat=2)
    # Ajustar essa pose e as sequentes com esse ICP
    for p in range(i, len(loam_poses)):
        loam_poses[i] = np.dot(transf, loam_poses[i])
    # Atualiza ajustes de ICP para a nuvem seguinte
    last_icp_adjust = np.dot(transf, last_icp_adjust)
    # Transforma a nuvem atual e acumula evitando repeticao
    src.transform(transf) 
    acc += remove_existing_points(src, acc, 5*voxel_size_fr)
    # Grava essa boa pose para o sfm final
    sfm_poses_list.append(np.linalg.inv(loam_poses[i]))
    
    
    #rz = np.array([[ m.cos(m.pi/2), -m.sin(m.pi/2), 0 ],
    #               [ m.sin(m.pi/2), m.cos(m.pi/2) , 0 ],
    #               [ 0           , 0            , 1 ]])
    #ry = np.matrix([[ m.cos(-m.pi/2), 0, m.sin(-m.pi/2)],
    #               [ 0           , 1, 0           ],
    #               [-m.sin(-m.pi/2), 0, m.cos(-m.pi/2)]])
    #rot_cam = np.dot(rz, ry)

    #Tb = np.identity(4, float)
    #Tb[0:3, 0:3] = loam_poses[i][0:3, 0:3]
    #Tb[0:3, 3  ] = np.dot(loam_poses[i][0:3, 0:3], loam_poses[i][0:3, 3])
    #transforms_list.append(Tb)

 
# Criar versao final do SFM otimizada
print("Creating optimized SFM file ...")
create_sfm_file(os.path.join(folder_path, "cameras_opt.sfm"), images_list, sfm_poses_list, k, Tcam, only_write=True)

# Salvar acumulada e mostrar
print("Saving ...")
accd = acc.voxel_down_sample(voxel_size=voxel_size_fr)
o3d.io.write_point_cloud(os.path.join(folder_path, "acumulada_opt.ply"), accd)
print("Display!")
o3d.visualization.draw_geometries([accd])

print("Creating mesh ...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(accd, depth=10)
## Filtrar por densidades
densities = np.asarray(densities)
densities = (densities - densities.min()) / (densities.max() - densities.min())
vertices_to_remove = densities < np.quantile(densities, 0.2)
mesh.remove_vertices_by_mask(vertices_to_remove)
mesh.filter_smooth_laplacian(number_of_iterations=6)

o3d.io.write_triangle_mesh(os.path.join(folder_path, "mesh.ply"), mesh)
o3d.visualization.draw_geometries([mesh])