import open3d as o3d
import numpy as np
import sys, os, shutil
import re
import math 
import copy
import cv2
import argparse

from functions import *

# Ler os parametros passados em linhas de comando
parser = argparse.ArgumentParser()
parser.add_argument('-root_path' , type=str  , required=True , default="C:\\Users\\vinic\\Desktop\\CAPDesktop\\ambientes\\demonstracao_ambiente")
parser.add_argument('-resolution', type=float, required=False, default=0.03)
#args = parser.parse_args(['-root_path=C:\\Users\\vinic\\Desktop\\CAPDesktop\\ambientes\\quintal_gps'])
args = parser.parse_args()
root_path     = args.root_path  
voxel_size    = args.resolution
ignored_files = ["acumulada", "acumulada_opt", "mesh", "panoramica", "planta_baixa"]
debug = False

# Parametros de calibracao da camera
k = np.array([[978.34, -0.013, 654.28], [0.054, 958.48, 367.49], [0, 0, 1]])

# Ler todas as pastas de scan no vetor de pastas, processar para cada pasta
folders_list = [os.path.join(root_path, f) for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

for fo, folder_path in enumerate(folders_list):
    print(f"Comecando SCAN {fo+1:d} de {len(folders_list):d} ...", flush=True)
    print("Carregando e filtrando cada nuvem de pontos ...", flush=True)
    point_clouds = load_point_clouds(folder_path, ignored_files, voxel_size, depth_max=30)

    # Somar as nuvens de cada PPV - adicionar em uma lista
    ntilts = 7
    if len(point_clouds) != 70:
        raise Exception("Numero de nuvens e imagens nao bate, cheque os arquivos!")

    print("Registrando cada PPV ...", flush=True)
    ppv_clouds = []
    ppv_cloud = o3d.geometry.PointCloud()
    for i, cloud in enumerate(point_clouds):
        print(f"Adicionando nuvem {i+1:d} de {len(point_clouds):d} ...", flush=True)
        ppv_cloud += remove_existing_points(cloud, ppv_cloud, voxel_size)
        if i > 0 and (i+1) % ntilts == 0:
            ppv_clouds.append(copy.deepcopy(ppv_cloud))
            ppv_cloud.clear()

    if debug:
        print("Visualizando resultado cru...", flush=True)
        o3d.visualization.draw_geometries(ppv_clouds, zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172,  2.0475,  1.5320], up=[-0.0694, -0.9768, 0.2024])
    
    # Por lado a lado cada nuvem, aproximando por ICP
    ppv_cloud.clear()
    acc = copy.deepcopy(ppv_clouds[0])
    transform_list = []
    transform_list.append(np.identity(4, float))
    for i, cloud in enumerate(ppv_clouds):
        print(f"Otimizando nuvem {i+1:d} de {len(ppv_clouds):d} PPVs ...", flush=True)
        transf, _ = pairwise_registration(cloud, acc, voxel_size, intensity=3, repeat=1, use_features=False)
        cloud.transform(transf)
        acc += remove_existing_points(cloud, acc, voxel_size)
        # Salvar transformacao de cada PPV em uma lista
        transform_list.append(transf)
    
    print(f"Salvando resultado do SCAN {fo+1:d} ...", flush=True)
    o3d.io.write_point_cloud(os.path.join(folder_path, "acumulada_opt.ply"), acc)

    if debug:
        print("Visualizando resultado otimizado ...", flush=True)
        o3d.visualization.draw_geometries(ppv_clouds, zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172,  2.0475,  1.5320], up=[-0.0694, -0.9768, 0.2024])
            
    # Ler arquivo SFM e imagens
    print(f"Lendo e atualizando arquivo SFM do SCAN {fo+1:d} ...", flush=True)
    raw_poses = read_sfm_file(os.path.join(folder_path, "cameras.sfm"))
    images_list = get_file_list(folder_path, ignored_files, extension=".png")
    if len(raw_poses) != len(images_list):
        raise Exception("Ha um problema com as imagens no diretorio deste SCAN, por favor checar.")

    # Multiplicar as poses do arquivo pela otimizada aqui
    new_poses = []
    for i, p in enumerate(raw_poses):
        Tppv = np.matmul( transform_list[int( (i-1)/(ntilts+1) )], np.linalg.inv(p) )
        new_poses.append(np.linalg.inv(Tppv))

    # Salvar o arquivo SFM
    print("Salvando arquivo SFM otimizado ...", flush=True)
    create_sfm_file(os.path.join(folder_path, "cameras_opt.sfm"), images_list, new_poses, k=k, only_write=True)

###################################
# ---------------------------------
###################################
for cont in range(4):
    print("------------------------------------------------------", flush=True)
    
# Se ha mais de um scan, processar e fundir todos
if len(folders_list) > 1:
    print("Iniciando processo de fusao de todos os SCANS ...", flush=True)
    # Ler a primeira nuvem e ajeitar referencias de GPS
    final_cloud = o3d.io.read_point_cloud(os.path.join(folders_list[0], 'acumulada_opt.ply'))
    gps_file = open(os.path.join(folders_list[0], 'gps.txt'), 'r')
    gps_ref  = gps_file.readlines()
    gps_file.close()
    # Ler o primeiro arquivo sfm e comecar a lista de linhas
    temp_poses = read_sfm_file(os.path.join(folders_list[0], "cameras.sfm"))
    images_list = get_file_list(folders_list[0], ignored_files, extension=".png")
    final_sfm_lines = assemble_sfm_lines(images_list, temp_poses, k)

    # Para cada nuvem restante, transformar por coordenadas de GPS, features e ICP
    for i in range(1, len(folders_list)):
        print(f"Lendo SCAN {i+1:d} de {len(folders_list):d} ...", flush=True)
        temp_cloud = o3d.io.read_point_cloud(os.path.join(folders_list[i], 'acumulada_opt.ply'))
        gps_file   = open(os.path.join(folders_list[i], 'gps.txt'), 'r')
        gps_temp   = gps_file.readlines()
        gps_file.close()
        # Lista de imagens e sfm daquela pasta
        temp_poses = read_sfm_file(os.path.join(folders_list[i], "cameras.sfm"))
        images_list = get_file_list(folders_list[i], ignored_files, extension=".png")

        # Encontrar transformada inicial por GPS
        #o3d.visualization.draw_geometries([final_cloud, temp_cloud])
        Tgps = find_transform_from_GPS(gps_temp, gps_ref)
        temp_cloud.transform(Tgps)        
        #o3d.visualization.draw_geometries([final_cloud, temp_cloud])
        # Aproximacao por ICP aqui agora
        print("Otimizando ...", flush=True)
        target = final_cloud.voxel_down_sample(3*voxel_size)
        source = temp_cloud.voxel_down_sample(3*voxel_size)
        transf, _ = pairwise_registration(source, target, voxel_size, intensity=6, repeat=3, use_features=True)
        temp_cloud.transform(transf)
        o3d.visualization.draw_geometries([final_cloud, temp_cloud])
        # Somar a nuvem final
        print("Adicionando resultado na nuvem de pontos global ...", flush=True)
        final_cloud += remove_existing_points(temp_cloud, final_cloud, voxel_size)
        
        # Atualizar as poses para o sfm final
        Ttemp = np.linalg.inv(np.dot(transf, Tgps))
        for i, p in enumerate(temp_poses):
            temp_poses[i] = np.dot(p, Ttemp)
        # Guardar as linhas para escrever tudo ao final
        final_sfm_lines += assemble_sfm_lines(images_list, temp_poses, k)

    # Salvar arquivo final de SFM e nuvem total na raiz
    print("Salvando arquivos finais no diretorio raiz ...", flush=True)
    o3d.io.write_point_cloud(os.path.join(root_path, "acumulada_opt.ply"), final_cloud)
    final_sfm_file = open(os.path.join(root_path, "cameras_opt.sfm"), 'w')
    final_sfm_file.write(str(len(final_sfm_lines))+"\n\n")
    for l in final_sfm_lines:
        final_sfm_file.write(l)
    final_sfm_file.close()

else: # Se e so um, copiar a nuvem acumulada e sfm para a pasta mae
    print("Salvando arquivos finais no diretorio raiz ...", flush=True)
    files = [os.path.join(folders_list[0], 'acumulada_opt.ply'), os.path.join(folders_list[0], 'cameras_opt.sfm')]
    for f in files:
        shutil.copy(f, root_path)

print("Nuvem de pontos de cameras processadas com sucesso !!", flush=True)