import open3d as o3d
import numpy as np
import sys, os
import re
import math 
import copy
import cv2

from functions import *

# Ler a pasta com as nuvens e carregar todas num vetor, ja simplificando
folder_path = "C:\\Users\\vinic\\Desktop\\CAPDesktop\\ambientes\\condominio\\scan1"
ignored_files = ["acumulada", "mesh", "panoramica", "planta_baixa"]
voxel_size = 0.05
debug = True

print("Loading and filtering every point cloud ...")
point_clouds = load_point_clouds(folder_path, ignored_files, voxel_size, depth_max=50)

# Somar as nuvens de cada PPV - adicionar em uma lista
ntilts = 7
if len(point_clouds) != 68:
    raise Exception("Some point clouds are missing! Check the synchronism.")

print("Registering for each PPV ...")
ppv_clouds = []
ppv_cloud = o3d.geometry.PointCloud()
for i, cloud in enumerate(point_clouds):
    print(f"Adding cloud {i+1:d} out of {len(point_clouds):d} ...")
    ppv_cloud += remove_existing_points(cloud, ppv_cloud, voxel_size)
    if i > 0 and (i+1) % ntilts == 0:
        ppv_clouds.append(copy.deepcopy(ppv_cloud))
        ppv_cloud.clear()
ppv_clouds.append(copy.deepcopy(ppv_cloud))

if debug:
    print("Visualizing raw result ...")
    o3d.visualization.draw_geometries(ppv_clouds[0:4], zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172,  2.0475,  1.5320], up=[-0.0694, -0.9768, 0.2024])
    
# Por lado a lado cada nuvem, aproximando por ICP
ppv_cloud.clear()
acc = copy.deepcopy(ppv_clouds[0])
transform_list = []
transform_list.append(np.identity(4, float))
for i, cloud in enumerate(ppv_clouds):
    print(f"Optimizing cloud {i+1:d} out of {len(ppv_clouds):d} PPVs ...")
    transf, info = pairwise_registration(cloud, acc, voxel_size, intensity=4, repeat=4, use_features=True)
    cloud.transform(transf)
    acc += remove_existing_points(cloud, acc, voxel_size)
    # Salvar transformacao de cada PPV em uma lista
    transform_list.append(transf)
    
print("Saving result ...")
o3d.io.write_point_cloud(os.path.join(folder_path, "acumulada.ply"), acc)

if debug:
    print("Visualizing optimized result ...")
    o3d.visualization.draw_geometries(ppv_clouds, zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172,  2.0475,  1.5320], up=[-0.0694, -0.9768, 0.2024])

# Criar a mesh e salvar
print("Creating mesh ...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(acc, depth=12)
## Filtrar por densidades
densities = np.asarray(densities)
densities = (densities - densities.min()) / (densities.max() - densities.min())
vertices_to_remove = densities < np.quantile(densities, 0.2)
mesh.remove_vertices_by_mask(vertices_to_remove)

o3d.io.write_triangle_mesh(os.path.join(folder_path, "mesh.ply"), mesh)

if debug:
    print("Checking the final mesh !")
    o3d.visualization.draw_geometries([mesh], zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172, 2.0475, 1.532], up=[-0.0694, -0.9768, 0.2024])

# Ler arquivo SFM e imagens
print("Reading and updating sfm file ...")
raw_poses = read_sfm_file(os.path.join(folder_path, "cameras.sfm"))
images_list = get_file_list(folder_path, ignored_files, extension=".png")
if len(raw_poses) != len(images_list):
    raise Exception("There is a problem with either the images or the SFM file, please check!")

# Multiplicar as poses do arquivo pela otimizada aqui
new_poses = []
for i, p in enumerate(raw_poses):
    Tppv = np.matmul( transform_list[int( (i-1)/(ntilts+1) )], np.linalg.inv(p) )
    new_poses.append(np.linalg.inv(Tppv))

# Salvar o arquivo SFM
print("Saving optimized sfm file ...")
k = np.array([[978.34, -0.013, 654.28], [0.054, 958.48, 367.49], [0, 0, 1]])
create_sfm_file(os.path.join(folder_path, "cameras_opt.sfm"), images_list, new_poses, k=k, only_write=True)

print("All done !!")