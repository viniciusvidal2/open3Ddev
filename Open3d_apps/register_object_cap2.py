import open3d as o3d
import numpy as np
import sys, os, shutil
import re
import math as m
import copy
import cv2
import argparse

from functions import *

# Versao atual do executavel
version = '1.3.5'
# Parametros recebidos pela linha de comando
parser = argparse.ArgumentParser(description='This is the CAP Object Point Cloud Estimator - v'+version+
                                 '. It processes the final object point cloud, from the data acquired '
                                 'by the CAP scanner.', epilog='Fill the parameters accordingly.')
parser.add_argument('-root_path'    , type=str  , required=True, default="C:\\Users\\vinic\\Desktop\\CAPDesktop\\objetos\\demonstracao_objeto",
                    help='REQUIRED. Path for the project root. All \"scanX\" folders should be in here, fully synchronized with CAP.')
parser.add_argument('-dobj'         , type=float, required=False, default=20,
                    help='Maximum distance from the scanner that the object was acquired. Points with further distances will be removed')
parser.add_argument('-fov_hor'      , type=float, required=False, default=50,
                    help='Horizontal Field of View accepted for every acquisition. Points out of this limit will be removed.')
parser.add_argument('-fov_ver'      , type=float, required=False, default=30,
                    help='Vertical Field of View accepted for every acquisition. Points out of this limit will be removed.')
parser.add_argument('-intensity_icp', type=int  , required=False, default=5,
                    help='The number of repetitions for the ICP optimization operation. Increasing this parameter value generally improves point cloud optimization,'
                    ' demanding more processing time.')
parser.add_argument('-manual_registration', type=str2bool, required=False, 
                    default=True,
                    help='Flag to set if each scan registration will be manually aided by the user. Recommended in large outdoor environments.')
#args = parser.parse_args(['-root_path=C:\\CapDesktop\\objetos\\tr'])
args = parser.parse_args()
root_path     = args.root_path
dobj          = args.dobj
fov_hor       = args.fov_hor
fov_ver       = args.fov_ver
intensity_icp = args.intensity_icp
manual_registration = args.manual_registration

print("CAP Object Point Cloud Estimator - v"+version, flush=True)

# Parametros gerais
ignored_files = ["acumulada", "acumulada_opt", "mesh", "panoramica", "planta_baixa"]
voxel_size_lr = 0.06
voxel_size_fr = 0.01

# Detalhes da camera
k = np.array([[978.34, -0.013, 654.28], [0.054, 958.48, 367.49], [0, 0, 1]])
Tcam = np.array([[1, 0, 0, -0.011], [0, 1, 0, 0.029], [0, 0, 1, 0.027], [0, 0, 0, 1]])

# Ler todas as pastas de scan no vetor de pastas, processar para cada pasta
folders_list = [os.path.join(root_path, f) for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
folders_list = [fo for fo in folders_list if fo.split('\\')[-1][0:4] == 'scan']

for fo, folder_path in enumerate(folders_list):
    print(f"Processando SCAN {fo+1:d} de {len(folders_list):d} ...", flush=True)
    # Lendo lista de nuvens e imagens
    print("Lendo lista de arquivos ...", flush=True)
    clouds_list = get_file_list(folder_path, ignored_files, extension=".ply")
    images_list = get_file_list(folder_path, ignored_files, extension=".png")
    if len(clouds_list) != len(images_list):
        raise Exception("O numero de nuvens deve corresponder ao numero de imagens no diretorio!")

    # Ler arquivo SFM
    print("Lendo arquivo original de poses das cameras ...", flush=True)
    loam_poses_original = read_sfm_file(os.path.join(folder_path, "cameras.sfm"))
    loam_poses_adjusted = copy.deepcopy(loam_poses_original)

    # Iniciar nuvem acumulada com a primeira leitura
    print("Lendo primeira nuvem e registrando ...", flush=True);
    acc = load_filter_point_cloud(clouds_list[0], voxel_size=voxel_size_fr, depth_max=dobj, T=loam_poses_original[0])
    acc = enclose_fov(acc, loam_poses_original[0], hor=fov_hor, ver=fov_ver)
    # acc = no_plane(acc)

    # Para cada nuvem, performar
    sfm_poses_list = []
    sfm_poses_list.append(loam_poses_adjusted[0])
    last_icp_adjust = np.identity(4, float)
    for i in range(1, len(clouds_list)):
        print(f"Registrando nuvem {i+1:d} de {len(clouds_list):d} ...", flush=True)
        # Ler e pre processar nuvem
        src = load_filter_point_cloud(clouds_list[i], voxel_size=voxel_size_fr, depth_max=dobj, T=loam_poses_original[i])
        src = enclose_fov(src, loam_poses_original[i], hor=fov_hor, ver=fov_ver)
        # Transformar para o ultimo ajuste que fizemos por ICP
        src.transform(last_icp_adjust)
        # Retirar o chao por aproximacao de plano
        # src = no_plane(src)
        # Aproximar as nuvens por ICP com ajuda manual se quiser
        transf, _ = pairwise_registration(src=src, tgt=acc, manual=manual_registration)
        # Transforma a nuvem atual e acumula evitando repeticao
        src.transform(transf) 
        # Atualiza aproximacao do ICP sequencial
        last_icp_adjust = np.dot(transf, last_icp_adjust)    
        # Ajustar essa pose e as sequentes com esse ICP
        for j in range(i, len(loam_poses_adjusted)):
            loam_poses_adjusted[j] = np.dot(loam_poses_adjusted[j], np.linalg.inv(transf))
        # Grava essa boa pose para o sfm final
        sfm_poses_list.append(loam_poses_adjusted[i])

        # Acumula nuvem final   
        print('Acumulando novos pontos na nuvem final ...', flush=True)
        acc += remove_existing_points(acc, src, 3*voxel_size_fr)

    # Criar versao final do SFM otimizada
    print("Criando arquivo de cameras .sfm otimizado ...", flush=True)
    create_sfm_file(os.path.join(folder_path, "cameras_opt.sfm"), images_list, sfm_poses_list, k, Tcam, only_write=True)

    # Salvar acumulada e mostrar
    print(f"Salvando resultados do SCAN {fo+1:d} ...", flush=True)
    accd = acc.voxel_down_sample(voxel_size=voxel_size_fr)
    o3d.io.write_point_cloud(os.path.join(folder_path, "acumulada_opt.ply"), accd)
    print(f"SCAN {fo+1:d} finalizado.", flush=True)

###################################
# ---------------------------------
###################################
for cont in range(4):
    print("------------------------------------------------------", flush=True)

# Se ha mais de um scan, processar e fundir todos
if len(folders_list) > 1:
    print("Iniciando processo de fusao de todos os SCANS ...", flush=True)
    print("Comecando resultado final com o SCAN 1 ...", flush=True)
    # Ler a primeira nuvem 
    final_cloud = o3d.io.read_point_cloud(os.path.join(folders_list[0], 'acumulada_opt.ply'))
    # Ler o primeiro arquivo sfm e comecar a lista de linhas
    temp_poses = read_sfm_file(os.path.join(folders_list[0], "cameras_opt.sfm"))
    images_list = get_file_list(folders_list[0], ignored_files, extension=".png")
    final_sfm_lines = assemble_sfm_lines(images_list, temp_poses, k)

    # Para cada nuvem restante, transformar por features e ICP
    for i in range(1, len(folders_list)):
        print(f"Lendo SCAN {i+1:d} de {len(folders_list):d} ...", flush=True)
        temp_cloud = o3d.io.read_point_cloud(os.path.join(folders_list[i], 'acumulada_opt.ply'))
        
        # Lista de imagens e sfm daquela pasta
        temp_poses = read_sfm_file(os.path.join(folders_list[i], "cameras_opt.sfm"))
        images_list = get_file_list(folders_list[i], ignored_files, extension=".png")

        # Transformar nuvem pela ultima pose encontrada no scan anterior - esse scan comeca aproximadamente dali
        l = [float(v) for v in final_sfm_lines[-1].split()[1:]]
        Tlast = np.linalg.inv(np.array([[l[0], l[1], l[2], l[ 9]],
                                        [l[3], l[4], l[5], l[10]],
                                        [l[6], l[7], l[8], l[11]],
                                        [0, 0, 0, 1]], dtype=np.float))
        temp_cloud.transform(Tlast)

        # Encontrar transformada inicial features e ICP
        print("Otimizando ...", flush=True)
        target = final_cloud.voxel_down_sample(voxel_size_lr)
        source = temp_cloud.voxel_down_sample(voxel_size_lr)
        transf, _ = pairwise_registration(source, target, voxel_size=voxel_size_lr, intensity=intensity_icp, repeat=3, use_features=False)
        temp_cloud.transform(transf)
        # Somar a nuvem final
        print("Adicionando resultado na nuvem de pontos global ...", flush=True)
        final_cloud += remove_existing_points(temp_cloud, final_cloud, 2*voxel_size_fr)
        
        # Atualizar as poses para o sfm final
        Ttemp = np.linalg.inv(np.dot(transf, Tlast))
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