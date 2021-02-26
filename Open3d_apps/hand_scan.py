import open3d as o3d
import os, sys, shutil
import re
import numpy as np
import matplotlib.pyplot as plt
import copy

from probreg import cpd
from pycpd import RigidRegistration


# Path geral de busca
path = "C:/Users/vinic/Desktop/Reconstruction/reconstruction_system/dataset/pessoa"

### FUNCOES - Lendo as imagens de forma organizada
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

## Escrevendo os arquivos finais
def save_trajectory(a, dig, cig, ps):
    depth_folder = os.path.join(path, 'depth2')
    image_folder = os.path.join(path, 'image2')
    scene_folder = os.path.join(path, 'scene' )
    if not os.path.isdir(scene_folder):
        os.makedirs(scene_folder)
    if not os.path.isdir(depth_folder):
        os.makedirs(depth_folder)
    if not os.path.isdir(image_folder):
        os.makedirs(image_folder)
    for d in dig:
        shutil.copy(d, depth_folder)
    for c in cig:
        shutil.copy(c, image_folder)
    o3d.io.write_point_cloud(os.path.join(scene_folder, "integrated.ply"), a)
    f = open(os.path.join(scene_folder, 'trajectory.log'), "w+")
    for i, p in enumerate(ps):
        f.write("{}\t{}\t{}\n".format(i, i, i+1))
        f.write("{} {} {} {}\n".format(p[0,0], p[0,1], p[0,2], p[0,3]))
        f.write("{} {} {} {}\n".format(p[1,0], p[1,1], p[1,2], p[1,3]))
        f.write("{} {} {} {}\n".format(p[2,0], p[2,1], p[2,2], p[2,3]))
        f.write("{} {} {} {}\n".format(p[3,0], p[3,1], p[3,2], p[3,3]))
    f.close()

## Retirar pontos ja existentes por vizinhanca
def remove_existing_points(s, a, max_n, radius):
    s2 = o3d.geometry.PointCloud()
    dists = s.compute_point_cloud_distance(a)
    for i, d in enumerate(dists):
        if d > radius:
            s2.points.append(s.points[i])
            s2.colors.append(s.colors[i])
            s2.normals.append(s.normals[i])
    return s2

## Checar se a odometria percorreu um threshold de distancia ou angulo
def check_new_scene(tnow, tref, dist_thresh, ang_thresh):
    # Vetor olhando para frente
    z = np.array([0,0,1,0], dtype=np.double)
    # Vetor multiplicado pela odometria ref e atual
    vref = np.matmul(tref, z)[0:3]
    vnow = np.matmul(tnow, z)[0:3]
    # Angulo entre as duas medidas de odometria
    angle = 180/3.1415*abs(np.arccos(np.clip(np.dot(vref/np.linalg.norm(vref), vnow/np.linalg.norm(vnow)), -1.0, 1.0)))
    # Distancia entre as duas odometrias
    dist = np.linalg.norm((tref[0:-1, 3]-tnow[0:-1, 3]))
    # Checando contra limites e retornando
    if angle > ang_thresh or dist > dist_thresh:
        return True
    else:
        return False
    
## Encontrar transformacao por features e RANSAC
def global_transform(src, tgt, vs):
    r  = 7*vs
    dt = 3*vs
    # Calcular features FPFH
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(src, o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=100))
    ## Acertar parametros do calculo RANSAC    
    #checkers = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dt)]
    ## Estimar transformacao por RANSAC
    #result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(src, tgt, src_fpfh, tgt_fpfh, dt, o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
    #                                                                                  ransac_n=30, checkers=checkers, criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500))
    # Acertar parametros FAST
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=dt)
    # Estimar transformacao por FAST
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(src, tgt, src_fpfh, tgt_fpfh, option)

    return result

### Main
if __name__ == "__main__":
    # Dados intrinsecos da camera astra
    w, h = 640, 480
    K = np.array([[570.3422241210938, 0, 319.5],[0, 570.3422241210938, 239.5],[0, 0, 1]])
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    scale = 1000
    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    intrinsics.intrinsic_matrix = K
    # Inicia nuvem de pontos
    tgt = o3d.geometry.PointCloud()
    src = o3d.geometry.PointCloud()    
    acc = o3d.geometry.PointCloud()
    rgbd_tgt = o3d.geometry.RGBDImage()
    # Inicia transformacoes que vao rolar
    Tref = np.identity(4, dtype=np.double)
    Tref_odo = Tref
    Tnow = np.identity(4, dtype=np.double)
    Tnow_odo = Tref
    # Vetores com imagens boas e suas poses de camera respectivas
    depth_images_good = []
    color_images_good = []
    poses = []
    # Ler images RGB e D das pastas devidas
    depth_image_path = get_file_list(os.path.join(path, "depth/"), extension=".png")
    color_image_path = get_file_list(os.path.join(path, "image/"), extension=".png")
    assert (len(depth_image_path) == len(color_image_path))

    n = 300
    vs = 0.01
    dt = 3*vs

    ## Pegar estimativas primeiros por odometria RGBD
    print('Computando odometria RGBD ...')
    #for i in range(0, len(depth_image_path), 3):
    for i in range(0, n, 3):
        print("Lendo imagens {} ...".format(i+1))
        depth = o3d.io.read_image(os.path.join(depth_image_path[i]))
        color = o3d.io.read_image(os.path.join(color_image_path[i]))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=scale, depth_trunc=1.5, convert_rgb_to_intensity=False)
        
        # Iniciando dados de nuvens e de imagem RGBD de referencia
        if i == 0:
            rgbd_tgt = rgbd_image

            src = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
            srcv = src.voxel_down_sample(voxel_size=vs)
            srcv.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
            srcv.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5*vs, max_nn=100))
            srcv.orient_normals_towards_camera_location()
            tgt = copy.deepcopy(srcv)
            acc = copy.deepcopy(srcv)

            # Salva imagens e nuvens
            depth_images_good.append(depth_image_path[i])
            color_images_good.append(color_image_path[i])
            poses.append(Tref)

        else:
            # Calcula a odometria aproximada por RGBD
            option = o3d.pipelines.odometry.OdometryOption(max_depth_diff=dt)
            [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(rgbd_image, rgbd_tgt, intrinsics, np.identity(4), o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
            # Atualiza a odometria atual, antes de aceitar um frame com novidades
            Tnow_odo = np.matmul(trans, Tnow_odo, dtype=np.double)
            # Analiza se o frame pode passar para a nuvem atual - a cena e nova o suficiente
            new_scene = check_new_scene(Tnow_odo, np.identity(4, dtype=np.double), dist_thresh=0.2, ang_thresh=10)
            if new_scene:

                print('Performando ICP e adicionando pontos inexistentes ...')
                # Ler a nuvem
                src = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
                srcv = src.voxel_down_sample(voxel_size=vs)
                srcv.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
                srcv.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5*vs, max_nn=100))
                srcv.orient_normals_towards_camera_location()

                # Transformada da cena de referencia acrescida da odometria atual entre as cenas
                Tref = np.matmul(Tnow_odo, Tref)
                # Traz nuvem da cena atual para proximo da nuvem acumulada
                srcv.transform(Tref)

                # Passar o alinhamento global por RANSAC e ajustar trasnformacao de referencia
                #ransac = global_transform(srcv, acc, vs)
                #srcv.transform(ransac.transformation)
                #Tref = np.matmul(ransac.transformation, Tref)

                ### Alinhar com CPD
                #tf_param, _, _ = cpd.registration_cpd(srcv, acc, update_scale=True, w=0, maxiter=100, tol=1e-5)
                #Tpcd = np.identity(4, np.double)
                #Tpcd[0:3, 0:3] = tf_param.rot
                #Tpcd[0:3, 3]   = tf_param.t
                #srcv.points = tf_param.transform(srcv.points)
                #Tref = np.matmul(Tpcd, Tref)
                # Testando 
                #reg = RigidRegistration(**{'X': np.asarray(acc.points), 'Y': np.asarray(srcv.points)})
                #reg.max_iterations = 100
                #reg.register()
                #for i, y in enumerate(reg.Y):
                #    srcv.points[i] = np.array(y)
                #Tpcd = np.identity(4, np.double)
                #Tpcd[0:3, 0:3] = reg.R
                #Tpcd[0:3, 3]   = reg.t
                #Tref = np.matmul(Tpcd, Tref)

                # Aplica ICP com a transformacao atual
                est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
                #est = o3d.pipelines.registration.TransformationEstimationForColoredICP()
                relative_fitness, relative_rmse = 1e-6, 1e-6
                previous_transform = np.identity(4, np.double)
                for j in range(0,1,1):
                    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=relative_fitness, relative_rmse=relative_rmse, max_iteration=50)
                    reg = o3d.pipelines.registration.registration_icp(srcv, acc, dt, previous_transform, est, criteria)
                    relative_fitness, relative_rmse = relative_fitness/10, relative_rmse/10
                    previous_transform = reg.transformation
                eval = o3d.pipelines.registration.evaluate_registration(srcv, acc, dt, previous_transform)
                print(eval)

                # Transforma a nuvem com o resultado do ICP e pega o resultado do ICP para a nova referencia
                srcv.transform(previous_transform)
                Tref = np.matmul(previous_transform, Tref)

                # Corrige pontos ja existentes
                srcv2 = remove_existing_points(srcv, acc, 20, vs)
                # Soma na acumulada os pontos novos
                acc = copy.deepcopy(acc) + srcv2
                accv = acc.voxel_down_sample(voxel_size=vs)
                acc = copy.deepcopy(accv)

                # Salva imagens e nuvens
                depth_images_good.append(depth_image_path[i])
                color_images_good.append(color_image_path[i])
                poses.append(Tref)

                ## Visualizando
                src3 = copy.deepcopy(srcv2)
                #src3.paint_uniform_color(np.array([0,0,1]))
                acc2 = copy.deepcopy(acc)
                #acc2.paint_uniform_color(np.array([0,1,0]))
                #acc2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                #src3.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                o3d.visualization.draw_geometries([acc2, src3])
                
                # Corrigindo a transformacao por odometria RGBD para a origem novamente
                Tnow_odo = np.identity(4, dtype=np.double)

            # Renova a imagem RGBD de referencia
            rgbd_tgt = rgbd_image
    
    # Salvando dados finais
    save_trajectory(acc, depth_images_good, color_images_good, poses)
    acc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([acc])