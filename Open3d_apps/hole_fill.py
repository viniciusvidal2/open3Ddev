import open3d as o3d
import os, sys, shutil
import re
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

### Criando nuvem de pontos de esfera
#esfera = o3d.geometry.PointCloud()
#for la in np.arange(-90, 90, 1):
#    for lo in np.arange(-180, 180, 1):
#        point = np.array([np.sin(np.rad2deg(la))*np.sin(np.rad2deg(lo)), np.sin(np.rad2deg(la))*np.cos(np.rad2deg(lo)), np.cos(np.rad2deg(la))], dtype=float)
#        color = np.array([1, 0, 1], dtype=float)
#        esfera.points.append(point)
#        esfera.colors.append(color)
#plane_model, inliers = esfera.segment_plane(distance_threshold=0.0001, ransac_n=10, num_iterations=1000)
#[a, b, c, d] = plane_model 
#plane = esfera.select_by_index(inliers)
#plane.paint_uniform_color([1, 0, 0])
#o3d.visualization.draw_geometries([plane])

## Ler nuvem
path = "C:\\Users\\vinic\\Desktop\\CAPDesktop\\CapDesktop\\ambientes\\estacionamento\\saida\\3DData"
cloud = o3d.io.read_point_cloud(os.path.join(path, "acumulada_opt.ply"))
print('outliers')
#cl, ind = cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=2)
#cloud2 = cloud.select_by_index(ind, invert=True)
#cloud2.paint_uniform_color([1.0, 0, 0])
print('voxels...')
cloudv = cloud.voxel_down_sample(voxel_size = 0.03)

#meshes = []

#not_plane = copy.deepcopy(cloudv)
#while len(not_plane.points) > 0.4*len(cloudv.points):
#    pct = float(len(not_plane.points))/float(len(cloudv.points))*100
#    print(f"Passada em busca de novo plano, restam {pct:.2f} pct ...")
#    plane_model, inliers = not_plane.segment_plane(distance_threshold=0.05, ransac_n=10, num_iterations=1000)
#    [a, b, c, d] = plane_model 
#    plane = not_plane.select_by_index(inliers)
#    ## Passar mesh
#    #radii = [0.005, 0.01, 0.02, 0.04, 0.1]
#    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#    #    cloudv, o3d.utility.DoubleVector(radii))
#    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#            plane, depth=10)
#    ## Filtrar por densidades
#    densities = np.asarray(densities)
#    densities = (densities - densities.min()) / (densities.max() - densities.min())
#    vertices_to_remove = densities < np.quantile(densities, 0.1)
#    mesh.remove_vertices_by_mask(vertices_to_remove)
#    meshes.append(mesh)
#    ## Mostrar resultado
#    #o3d.visualization.draw_geometries([mesh])
#    not_plane = not_plane.select_by_index(inliers, invert=True)
print('mesh e densidades')
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloudv, depth=11)
## Filtrar por densidades
densities = np.asarray(densities)
densities = (densities - densities.min()) / (densities.max() - densities.min())
vertices_to_remove = densities < np.quantile(densities, 0.2)
mesh.remove_vertices_by_mask(vertices_to_remove)
#meshes.append(mesh)

#mesh_final = o3d.geometry.TriangleMesh()
#for m in meshes:
#    mesh_final += m
print('ver e salvar')
o3d.io.write_triangle_mesh(os.path.join(path, "mesh.ply"), mesh)
o3d.visualization.draw_geometries([mesh])
#o3d.visualization.draw_geometries([m])

