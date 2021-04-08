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
parser.add_argument('-resolution', type=float, required=False, default=0.05)
#args = parser.parse_args(['-root_path=C:\\Users\\vinic\\Desktop\\CAPDesktop\\ambientes\\estacionamento'])
args = parser.parse_args()
root_path     = args.root_path  
voxel_size    = args.resolution
ignored_files = ["acumulada", "acumulada_opt", "mesh", "panoramica", "planta_baixa"]
debug = False

print("CAP Space Point Cloud Estimator 1.0.1", flush=True)

# Parametros de calibracao da camera
k = np.array([[978.34, -0.013, 654.28], [0.054, 958.48, 367.49], [0, 0, 1]])

# Ler todas as pastas de scan no vetor de pastas, processar para cada pasta
folders_list_all = [os.path.join(root_path, f) for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
folders_list     = [fo for fo in folders_list_all if fo.split('\\')[-1][0:4] == 'scan']

for fo, folder_path in enumerate(folders_list):
    # Ler arquivo SFM e imagens
    print(f"Lendo arquivo SFM do SCAN {fo+1:d} ...", flush=True)
    raw_poses = read_sfm_file(os.path.join(folder_path, "cameras.sfm"))[:80]
    images_list = get_file_list(folder_path, ignored_files, extension=".png")
    if len(raw_poses) != len(images_list):
        raise Exception("Ha um problema com as imagens no diretorio deste SCAN, por favor checar.")

    # Se o arquivo de nuvem existir correspondente a cada imagem, ler e processar (evita nuvens vazias observando o ceu, por exemplo)
    # Ao completar um PPV, somar ali todas as nuvens
    ppv_clouds = []
    ppv_cloud = o3d.geometry.PointCloud()
    ntilts = 8
    for i in range(len(images_list)):
        cloud_path = os.path.join(folder_path, 'c_'+str(i+1).zfill(3)+'.ply')
        if os.path.exists(cloud_path):
            # Pre processar a nuvem
            print(f'Nuvem {i+1:d} de {len(images_list)} existe, processando ...', flush=True)
            cloud = load_filter_point_cloud(cloud_path, voxel_size, 30, raw_poses[i])
            # Somar na vista do PPV
            ppv_cloud += cloud#remove_existing_points(cloud, ppv_cloud, voxel_size)
            # Mudar de PPV e salvar aquele se for o caso
        if i > 0 and (i+1) % ntilts == 0:
            print('Ajustando este PPV ...', flush=True)
            ppv_clouds.append(copy.deepcopy(ppv_cloud.voxel_down_sample(voxel_size)))
            ppv_cloud.clear()    
    ppv_cloud.clear()

    if debug:
        print("Visualizando resultado cru...", flush=True)
        o3d.visualization.draw_geometries(ppv_clouds, zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172,  2.0475,  1.5320], up=[-0.0694, -0.9768, 0.2024])
    
    # Por lado a lado cada nuvem, aproximando por ICP
    acc = copy.deepcopy(ppv_clouds[0])
    transform_list = []
    transform_list.append(np.identity(4, float))
    for i, cloud in enumerate(ppv_clouds):
        if i > 0:
            print(f"Otimizando nuvem {i+1:d} de {len(ppv_clouds):d} PPVs ...", flush=True)
            target = acc.voxel_down_sample(3*voxel_size)
            source = cloud.voxel_down_sample(3*voxel_size)
            transf, _ = pairwise_registration(source, target, voxel_size, intensity=4, repeat=1, use_features=False)            
            cloud.transform(transf)
            acc += copy.deepcopy(cloud)#remove_existing_points(cloud, acc, voxel_size)
            # Salvar transformacao de cada PPV em uma lista
            transform_list.append(transf)
    ppv_clouds.clear()

    print(f"Salvando resultado do SCAN {fo+1:d} ...", flush=True)
    o3d.io.write_point_cloud(os.path.join(folder_path, "acumulada_opt.ply"), filter_depth(acc.voxel_down_sample(voxel_size), 40))

    if debug:
        print("Visualizando resultado otimizado ...", flush=True)
        o3d.visualization.draw_geometries(ppv_clouds, zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172,  2.0475,  1.5320], up=[-0.0694, -0.9768, 0.2024])

    # Multiplicar as poses do arquivo pela otimizada aqui
    new_poses = []
    for i, p in enumerate(raw_poses):
        Tppv = np.dot( transform_list[int(i/ntilts)], np.linalg.inv(p) )
        new_poses.append(np.linalg.inv(Tppv))

    # Salvar o arquivo SFM
    print("Salvando arquivo SFM otimizado ...", flush=True)
    create_sfm_file(os.path.join(folder_path, "cameras_opt.sfm"), images_list, new_poses, k=k, only_write=True)

#####################################
# --------------------------------- #
#####################################
for cont in range(4):
    print("------------------------------------------------------", flush=True)
    
# Se ha mais de um scan, processar e fundir todos
if len(folders_list) > 1:
    print("Iniciando processo de fusao de todos os SCANS ...", flush=True)
    # Varrer todos avaliando coordenadas de GPS, definir referencia ou se e ambiente interno
    print("Definindo referencia por GPS ou ambiente interno ...", flush=True)
    gps_ref_ind = 0
    ambiente_interno = True
    for i in range(len(folders_list)):
        gps_file = open(os.path.join(folders_list[i], 'gps.txt'), 'r')
        gps  = gps_file.readlines()
        gps_file.close()
        if float(gps[0]) != 0:
            gps_ref_ind = i
            ambiente_interno = False
            break
    # Definir a ordem para a otimizacao dependendo de quem e a referencia
    register_order = []
    if gps_ref_ind == 0:
        register_order = list(range(1, len(folders_list)))
    else:
        register_order = list(range(gps_ref_ind-1, -1, -1)) + list(range(gps_ref_ind+1, len(folders_list), 1))
    # Ler a primeira nuvem e ajeitar referencias de GPS
    print('Definindo SCAN de referencia ...', flush=True)
    final_cloud = o3d.io.read_point_cloud(os.path.join(folders_list[gps_ref_ind], 'acumulada_opt.ply'))
    gps_file = open(os.path.join(folders_list[gps_ref_ind], 'gps.txt'), 'r')
    gps_ref  = gps_file.readlines()
    gps_file.close()
    # Ler o primeiro arquivo sfm e comecar a lista de linhas
    temp_poses = read_sfm_file(os.path.join(folders_list[gps_ref_ind], "cameras_opt.sfm"))
    images_list = get_file_list(folders_list[gps_ref_ind], ignored_files, extension=".png")
    images_list_reduced = [os.path.join( il.split('\\')[-2], il.split('\\')[-1]) for il in images_list]
    final_sfm_lines_texture = assemble_sfm_lines(images_list_reduced, temp_poses, k)
    final_sfm_lines_360     = assemble_sfm_lines(["images/"+folders_list[gps_ref_ind].split("\\")[-1]+"_panoramica.png"], [np.identity(4, float)], k)

    # Para cada nuvem que nao seja a referencia, transformar por coordenadas de GPS, features e ICP
    Tlast = np.identity(4, float)
    for ind, i in enumerate(register_order):
        print(f"Lendo SCAN {ind+1:d} de {len(register_order):d} ...", flush=True)
        temp_cloud = o3d.io.read_point_cloud(os.path.join(folders_list[i], 'acumulada_opt.ply'))
        gps_file   = open(os.path.join(folders_list[i], 'gps.txt'), 'r')
        gps_temp   = gps_file.readlines()
        gps_file.close()
        # Lista de imagens e sfm daquela pasta
        temp_poses = read_sfm_file(os.path.join(folders_list[i], "cameras_opt.sfm"))
        images_list = get_file_list(folders_list[i], ignored_files, extension=".png")
        images_list_reduced = [os.path.join( il.split('\\')[-2], il.split('\\')[-1]) for il in images_list]

        # Encontrar transformada inicial por GPS
        #o3d.visualization.draw_geometries([final_cloud, temp_cloud])
        Tgps = find_transform_from_GPS(gps_temp, gps_ref, ambiente_interno)
        Tcoarse = Tlast if (Tgps == np.identity(4, float)).all() else Tgps
        temp_cloud.transform(Tcoarse)
        #o3d.visualization.draw_geometries([final_cloud, temp_cloud])
        # Aproximacao por ICP aqui agora
        print("Otimizando ...", flush=True)
        target = final_cloud.voxel_down_sample(5*voxel_size)
        source = temp_cloud.voxel_down_sample(5*voxel_size)
        if not ambiente_interno:
            transf, _ = pairwise_registration(source, target, 0.01, intensity=50, repeat=1, use_features=False)
        else:
            transf, _ = pairwise_registration(source, target, voxel_size, intensity=20, repeat=1, use_features=False)
        temp_cloud.transform(transf)
        #o3d.visualization.draw_geometries([final_cloud, temp_cloud])
        # Somar a nuvem final
        print("Adicionando resultado na nuvem de pontos global ...", flush=True)
        final_cloud += remove_existing_points(temp_cloud, final_cloud, voxel_size)
        
        # Atualizar as poses para o sfm final
        Ttemp = np.linalg.inv(np.dot(transf, Tcoarse))
        for t, p in enumerate(temp_poses):
            temp_poses[t] = np.dot(p, Ttemp)
        # Guardar as linhas para escrever tudo ao final
        final_sfm_lines_texture += assemble_sfm_lines(images_list_reduced, temp_poses, k)
        final_sfm_lines_360     += assemble_sfm_lines(["images/"+folders_list[i].split("\\")[-1]+"_panoramica.png"], [Ttemp], k)
        # Salvar ultima transformacao para ajudar se a proxima nao tiver coordenadas
        Tlast = np.dot(Tcoarse, transf)

    # Salvar arquivos finais de SFM e nuvem total na raiz
    print("Salvando arquivos finais no diretorio raiz ...", flush=True)
    o3d.io.write_point_cloud(os.path.join(root_path, "acumulada_opt.ply"), final_cloud)
    final_sfm_file_texture = open(os.path.join(root_path, "cameras_opt.sfm"), 'w')
    final_sfm_file_texture.write(str(len(final_sfm_lines_texture))+"\n\n")
    for l in final_sfm_lines_texture:
        final_sfm_file_texture.write(l)
    final_sfm_file_texture.close()
    final_sfm_file_360 = open(os.path.join(root_path, "cameras.sfm"), 'w')
    final_sfm_file_360.write(str(len(final_sfm_lines_360))+"\n\n")
    for l in final_sfm_lines_360:
        final_sfm_file_360.write(l)
    final_sfm_file_360.close()

else: # Se e so um, copiar a nuvem acumulada e sfm para a pasta mae
    print("Salvando arquivos finais no diretorio raiz ...", flush=True)
    files = [os.path.join(folders_list[0], 'acumulada_opt.ply'), os.path.join(folders_list[0], 'cameras_opt.sfm')]
    for f in files:
        shutil.copy(f, root_path)

print("Nuvem de pontos e poses cameras processadas com sucesso !!", flush=True)