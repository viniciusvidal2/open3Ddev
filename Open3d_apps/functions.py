import open3d as o3d
import numpy as np
import sys, os
import re
import math 
import copy
import cv2
import json
import colorsys
#######################################################################################################
def no_plane(cloud):
    plane_model, inliers = cloud.segment_plane(distance_threshold=0.1, ransac_n=10, num_iterations=1000)
    [a, b, c, d] = plane_model 
    if -1.8 < d < -1.1 and abs(b) > abs(a) and abs(b) > abs(c):
        cloud2 = cloud.select_by_index(inliers, invert=True)
        ys = np.asarray(cloud2.points)[:, 1]
        return cloud2.select_by_index(np.argwhere(ys < abs(d)-0.2))
    else:
        ys = np.asarray(cloud.points)[:, 1]
        return cloud.select_by_index(np.argwhere(ys < 1))
#######################################################################################################
def select_best_region(src, tgt):
    dists = src.compute_point_cloud_distance(tgt)
    srct = src.select_by_index(np.argwhere(np.asarray(dists) < 1))
    mean, _ = srct.compute_mean_and_covariance()
    b = 50
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.asarray(mean)-b, max_bound=np.asarray(mean)+b)
    return copy.deepcopy(src).crop(box), copy.deepcopy(tgt).crop(box)
#######################################################################################################
def filter_colors_hsv(cloud, lim1, lim2):
    # Varrer os valores rgb, converter hsv e retirar os pontos desejados que estao dentro dos limites
    hsvs = [colorsys.rgb_to_hsv(c[0], c[1], c[2]) for c in np.asarray(cloud.colors)]
    indices = np.argwhere(np.array([1 if sum(hsv > lim1) + sum(hsv < lim2) == 6 else 0 for hsv in hsvs]))

    return cloud.select_by_index(indices, invert=True)
#######################################################################################################
def filter_sun(cloud):
    indices = []
    # Olhando so parte superior da nuvem para ir mais rapido
    zs = np.asarray(cloud.points)[:, 1]
    cup = cloud.select_by_index(np.argwhere(zs < 0))
    cdown = cloud.select_by_index(np.argwhere(zs >= 0))
    # Criar direcoes apontando para cima
    lats = math.pi/180*np.arange(-150, -30, 5)
    lons = math.pi/180*np.arange(-179, 179, 5)
    # Ver o ponto mais proximo e mais distante para limitar a busca
    point_norms = [np.linalg.norm(p) for p in np.asarray(cup.points)]
    closest = np.min(np.asarray(point_norms))
    further = np.max(np.asarray(point_norms))
    dists = np.arange(closest, further, 0.2)
    # Setar um vetor de nuvens de pontos para medir a distancia em cada direcao
    rays = []
    for lat in lats:
        for lon in lons:
            vec = np.array([np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)])
            vec = vec/np.linalg.norm(vec)
            pts = np.array([d*vec for d in dists])
            ray = o3d.geometry.PointCloud()
            ray.points = o3d.utility.Vector3dVector(pts)
            rays.append(ray)
    # Salvar o numero de pontos proximos para cada vetor e os indices desses pontos em duas listas
    n_neighbors = []
    close_indices = []
    for r in rays:
        dists = cup.compute_point_cloud_distance(r)
        closers = np.argwhere(np.asarray(dists) < 2)
        if len(closers) > 0:
            n_neighbors.append(len(closers))
            close_indices.append(closers)
        else:
            n_neighbors.append(0)
            close_indices.append([])
    # Os que tiverem mais pontos proximos sao apagados da nuvem inicial
    max_neighbors = np.asarray(n_neighbors).max()
    indices_remove = np.array([])
    for i, n in enumerate(n_neighbors):
        if n > 0.7*max_neighbors:
            indices_remove = np.append(indices_remove, close_indices[i])
    #off = cup.select_by_index(np.unique(indices_remove).astype(int))
    cup = cup.select_by_index(np.unique(indices_remove).astype(int), invert=True)
    cup, _ = cup.remove_statistical_outlier(nb_neighbors=10, std_ratio=2)
    #o3d.visualization.draw_geometries([cup, off.paint_uniform_color([1, 0, 0])])
    return cup + cdown
#######################################################################################################
def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)
#######################################################################################################
def get_file_list(path, ignored_files, extension=None):
    if extension is None:
        file_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(join(path, f))]
    else:
        file_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) 
                     and os.path.splitext(f)[1] == extension 
                     and os.path.splitext(f)[0] != ignored_files[0]
                     and os.path.splitext(f)[0] != ignored_files[1]
                     and os.path.splitext(f)[0] != ignored_files[2]
                     and os.path.splitext(f)[0] != ignored_files[3]
                     and os.path.splitext(f)[0] != ignored_files[4]]
    file_list = sorted_alphanum(file_list)
    return file_list
#######################################################################################################
def filter_depth(cloud, dmax):
    dists = [np.linalg.norm(point) for point in np.asarray(cloud.points)]
    indices = np.squeeze(np.argwhere(np.asarray(dists) < dmax))
    cloud2 = cloud.select_by_index(indices)

    return cloud2
#######################################################################################################
## Retirar pontos ja existentes por vizinhanca
def remove_existing_points(src, tgt, radius):
    if len(tgt.points) == 0:
        return src
    # Somente somar as nuvens
    out = src + tgt
    # Passar voxel no resultado
    return out.voxel_down_sample(voxel_size=radius)


    #dists = np.squeeze(src.compute_point_cloud_distance(tgt))
    #indices = np.squeeze(np.argwhere(np.asarray(dists, dtype=float) > radius))
    
    #return src.select_by_index(indices) if indices.size > 1 else o3d.geometry.PointCloud()
#######################################################################################################
def load_point_clouds(folder, final_name, voxel_size=0.0, depth_max=10):
    pcds = []
    cloud_paths = get_file_list(folder, final_name, extension=".ply")
    for i in range(len(cloud_paths)):
        print(f"Lendo nuvem {i+1:d} de {len(cloud_paths):d} no total ...", flush=True)
        pcd = o3d.io.read_point_cloud(cloud_paths[i])
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.4)
        pcd_down2 = filter_depth(copy.deepcopy(pcd_down), depth_max)
        #pcd_down2 = raycasting(pcd_down2, 0.5, 22, 88, voxel_size, depth_max)
        if len(pcd_down2.points) > 100:
            pcd_down2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10*voxel_size, max_nn=250))
            pcd_down2.orient_normals_towards_camera_location()
            pcd_down2.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.4)
            pcds.append(pcd_down2)
        else:
            pcds.append(o3d.geometry.PointCloud())

    return pcds
#######################################################################################################
def load_filter_point_cloud(name, voxel_size=0.0, depth_max=10, T=np.identity(4, float), raycast=False):
    pcd = o3d.io.read_point_cloud(name)
    if len(pcd.points) < 100:
        return o3d.geometry.PointCloud()
    pcd_down, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=3)
    if voxel_size != 0:
        pcd_down2 = pcd_down.voxel_down_sample(voxel_size=voxel_size)
    else:
        pcd_down2 = copy.deepcopy(pcd_down)
    #pcd_down2, _ = pcd_down2.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)
    pcd_down2.transform(T)
    pcd_down2 = filter_depth(copy.deepcopy(pcd_down2), depth_max)
    if raycast:
        pcd_down2 = raycasting(pcd_down2, 1, 30, 75, 0.05, depth_max)
    if len(pcd_down2.points) > 10:
        pcd_down2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5*voxel_size, max_nn=100))
        pcd_down2.orient_normals_towards_camera_location()
    else:
        return o3d.geometry.PointCloud()
    pcd_down2.transform(np.linalg.inv(T))

    return pcd_down2
#######################################################################################################
def pick_points(pcd):
    print("", flush=True)
    print("1) Por favor escolha no minimo 3 pontos correspondentes utilizando [shift + botao esquerdo do mouse]", flush=True)
    print("   Pressione [shift + botao direito do mouse] para desfazer um clique, caso necessario", flush=True)
    print("   OBS: Caso entenda que o alinhamento ja atende o requisito necessario, nao ha necessidade de selecionar nenhum ponto", flush=True)
    print("2) Ao final, pressione 'Q' para fechar a janela", flush=True)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        return vis.get_picked_points()
#######################################################################################################
def manual_registration(source, target):
    # Mostrar as nuvens para ter uma ideia
    print("Visualizando nuvens antes do alinhamento ...", flush=True)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries(window_name='Demonstrativo das nuvens', geometry_list=[source_temp, target_temp])

    # pick points from two point clouds and builds correspondences
    print("Escolhendo pontos da nuvem source ...", flush=True)
    picked_id_source = pick_points(source)
    print("Escolhendo pontos da nuvem target ...", flush=True)
    picked_id_target = pick_points(target)
    print("", flush=True)

    # Checar se a pessoa escolheu pontos, pois ja pode ser que esteja bom o alinhamento inicial
    if len(picked_id_source) > 0 and len(picked_id_target) > 0:
        assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
        assert (len(picked_id_source) == len(picked_id_target))
        corr = np.zeros((len(picked_id_source), 2))
        corr[:, 0] = picked_id_source
        corr[:, 1] = picked_id_target

        # estimate rough transformation using correspondences
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))
    else:
        return np.identity(4, float)

    return trans_init
#######################################################################################################
def pairwise_registration(src, tgt, use_features=False, initial=np.identity(4, float), manual=False):    
    Tsa = np.identity(4, float)
    # Separar em conjunto de voxels, intensidades e distancias a serem checadas 
    voxels = [0.2, 0.1, 0.04]
    min_dists = [10, 2, 0.10]
    iterations = [1500, 1000, 700]
    planes_removed = False

    if manual:
        voxel_size = voxels[-1]
        target = tgt.voxel_down_sample(voxel_size)
        source = src.voxel_down_sample(voxel_size)
        source, _ = source.remove_statistical_outlier(nb_neighbors=10, std_ratio=2)
        Tsa = manual_registration(source, target)
        print('Refinando a aproximacao ...', flush=True)
        source, target = select_best_region(source.transform(Tsa), target)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-15, relative_rmse=1e-15, max_iteration=250)
        icp_fine = o3d.pipelines.registration.registration_icp(source, target, 0.06, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria)
        Tsa = np.dot(icp_fine.transformation, Tsa)
        teste = copy.deepcopy(source).transform(icp_fine.transformation)
        o3d.visualization.draw_geometries(geometry_list=[teste.paint_uniform_color([0, 1, 0]), target], window_name='Resultado Final')
        return Tsa, _
    else:
        if use_features:
            voxel_size = voxels[0]
            target = no_plane(tgt.voxel_down_sample(voxel_size)) if planes_removed else tgt.voxel_down_sample(voxel_size)
            source = no_plane(src.voxel_down_sample(voxel_size)) if planes_removed else src.voxel_down_sample(voxel_size)
            #target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size, max_nn=10))
            #target.orient_normals_towards_camera_location()
            #source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size, max_nn=10))
            #source.orient_normals_towards_camera_location()
            source, target = select_best_region(source, target)
            radius_feature = 8*voxel_size
            source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamRadius(radius=radius_feature))
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamRadius(radius=radius_feature))
            result_fast = o3d.pipelines.registration.registration_fast_based_on_feature_matching(source, target, source_fpfh, target_fpfh, 
                                                                                                 o3d.pipelines.registration.FastGlobalRegistrationOption(
                                                                                                     maximum_correspondence_distance=voxel_size/3))
            #result_fast = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source, target, source_fpfh, target_fpfh, max_correspondence_distance=1.5*voxel_size,
            #                                                                                       checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold=voxel_size)])
            Tsa = result_fast.transformation
        else:
            Tsa = initial
    
        #teste = copy.deepcopy(src)
        #o3d.visualization.draw_geometries([teste.transform(Tsa).paint_uniform_color([0, 1, 1]), tgt], zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172,  2.0475,  1.5320], up=[-0.0694, -0.9768, 0.2024])
        # Rodar para cada conjunto de voxels o procedimento
        for i in range(len(voxels)):
            target = no_plane(tgt.voxel_down_sample(voxels[i])) if planes_removed else tgt.voxel_down_sample(voxels[i])
            source = no_plane(src.voxel_down_sample(voxels[i])) if planes_removed else src.voxel_down_sample(voxels[i])
            source, target = select_best_region(source, target)
            target, _ = target.remove_statistical_outlier(nb_neighbors=10, std_ratio=2)
            source, _ = source.remove_statistical_outlier(nb_neighbors=10, std_ratio=2)

            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-15, relative_rmse=1e-15, max_iteration=iterations[i])
            icp_fine = o3d.pipelines.registration.registration_icp(source, target, min_dists[i], Tsa, o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria)
            if len(icp_fine.correspondence_set) == 0:
                continue
            if abs(check_angle(icp_fine.transformation, np.eye(4))) < 50 and icp_fine.fitness != 1.0:
                Tsa = icp_fine.transformation
            #print(icp_fine, flush=True)

            #teste = copy.deepcopy(source).transform(Tsa)
            #o3d.visualization.draw_geometries([teste.paint_uniform_color([0, 1, 0]), target], zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172,  2.0475,  1.5320], up=[-0.0694, -0.9768, 0.2024])
    
        return Tsa, _
#######################################################################################################
def full_registration(pcds, voxel_size, poses, loop_closure=True, transfs=[], infos=[]):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(poses[0])))
    n_pcds = len(pcds)
    for target_id in range(n_pcds):
        print(f"Graph node for cloud {target_id+1:d} ...", flush=True)
        for source_id in range(target_id + 1, n_pcds):            
            if source_id == target_id + 1:  # odometry case
                #transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id], voxel_size)
                pose_graph.nodes.append( o3d.pipelines.registration.PoseGraphNode(poses[source_id]))
                pose_graph.edges.append( o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transfs[source_id-1], infos[source_id-1], uncertain=False) )
            elif source_id == target_id + 2:  # loop closure case
                transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id], voxel_size)
                pose_graph.edges.append( o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transformation_icp, information_icp, uncertain=True) )
    # Add the final with loop closure
    if loop_closure:
        transformation_icp, information_icp = pairwise_registration(pcds[n_pcds-1], pcds[0], voxel_size)
        pose_graph.nodes.append( o3d.pipelines.registration.PoseGraphNode(poses[n_pcds-1]))
        pose_graph.edges.append( o3d.pipelines.registration.PoseGraphEdge(n_pcds-1, 0, transformation_icp, information_icp, uncertain=False))

    return pose_graph
#######################################################################################################
def color_point_clouds(folder_path, clouds, k, t):
    clouds2 = []
    images = get_file_list(folder_path, extension=".png")

    t = t[:, np.newaxis]

    if len(images) != len(clouds):
        raise Exception("Deve haver mesmo numero de imagens e nuvens no diretorio, observe.")
        return 
    
    for im_id, cloud in enumerate(clouds):
        cloud2 = copy.deepcopy(cloud)
        im = cv2.imread(images[im_id])
        for id, p in enumerate(cloud.points):
            pT = p[:, np.newaxis]
            pixel_homogeneous = np.dot(k, (pT+t))
            pixel_homogeneous = (pixel_homogeneous/pixel_homogeneous[2]).astype(int)
            if 0 <= pixel_homogeneous[1][0] < im.shape[0] and 0 <= pixel_homogeneous[0][0] < im.shape[1]:
                pixel_color = (im[pixel_homogeneous[1][0], pixel_homogeneous[0][0]]).astype(float)/255
                cloud2.colors[id] = pixel_color[::-1]
        clouds2.append(cloud2)

    return clouds2
#######################################################################################################
def color_point_cloud(image_path, cloud, k, t):  
    t = t[:, np.newaxis]
    cloud2 = copy.deepcopy(cloud)
    im = cv2.imread(image_path)
    for id, p in enumerate(cloud.points):
        pT = p[:, np.newaxis]
        pixel_homogeneous = np.dot(k, (pT+t))
        pixel_homogeneous = (pixel_homogeneous/pixel_homogeneous[2]).astype(int)
        if 0 <= pixel_homogeneous[1][0] < im.shape[0] and 0 <= pixel_homogeneous[0][0] < im.shape[1]:
            pixel_color = (im[pixel_homogeneous[1][0], pixel_homogeneous[0][0]]).astype(float)/255
            cloud2.colors[id] = pixel_color[::-1]

    return cloud2
#######################################################################################################
def raycasting(cloud, step, fov_lat, fov_lon, thresh, max_depth):
    cloud2 = o3d.geometry.PointCloud()
    # Campo de visao para procurar - angulos
    lats = math.pi/180*np.arange(-fov_lat/2, fov_lat/2, step)
    lons = math.pi/180*np.arange(-fov_lon/2, fov_lon/2, step)
    # Ver o ponto mais proximo e mais distante para limitar a busca
    point_norms = [np.linalg.norm(p) for p in np.asarray(cloud.points)]
    closest = np.min(np.asarray(point_norms))
    further = np.max(np.asarray(point_norms))
    dists = np.arange(closest, further, thresh)
    # Criar a esfera a partir dos angulos de coordenada desejados, para avancar como uma onda sobre a nuvem
    sphere_wave = o3d.geometry.PointCloud()
    pp = []
    for lat in lats:
        for lon in lons:
            vec = np.array([np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)])
            vec = vec/np.linalg.norm(vec)
            pp.append(vec)
    sphere_wave.points = o3d.utility.Vector3dVector(np.array(pp))
    # Varrer para cada distancia os pontos da nuvem que passam, e seleciona-los
    sphere_ignore_indices = []
    point_cloud_valid_indices = []
    sphere_wave_temp = o3d.geometry.PointCloud()
    for d in dists:
        sphere_wave_temp.points = o3d.utility.Vector3dVector(np.asarray(sphere_wave.points)*d)
        distances1 = np.squeeze(sphere_wave_temp.compute_point_cloud_distance(cloud))
        # Pontos proximos na esfera serao apagados
        sphere_ignore_indices = np.argwhere(distances1 < thresh)
        # Pontos proximos na nuvem sao selecionados
        if len(sphere_ignore_indices) > 0:
            distances2 = np.squeeze(cloud.compute_point_cloud_distance(sphere_wave_temp))
            valids = np.argwhere(distances2 < thresh)
            if len(valids) > 0:
                point_cloud_valid_indices = np.append(point_cloud_valid_indices, valids)
            test = sphere_wave.select_by_index(sphere_ignore_indices)
            sphere_wave = sphere_wave.select_by_index(sphere_ignore_indices, invert=True)
            #o3d.visualization.draw_geometries([sphere_wave_temp.paint_uniform_color([1, 0, 0]), cloud, test])

    # Separar os pontos validos
    if len(point_cloud_valid_indices) > 0:
        return cloud.select_by_index(np.unique(point_cloud_valid_indices.astype(int)))


    #for lat in lats:
    #    for lon in lons:
    #        vec = np.array([np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)])
    #        # Para cada direcao, criar a linha que representa o raio 
    #        ray_points = np.array([d*vec for d in dists])
    #        # Testar se ha vizinhos na nuvem para aquele ponto, se sim adicionar o ponto na nova nuvem  
    #        for r in ray_points:              
    #            [_, indices, _] = tree.search_radius_vector_3d(r, 0.1)
    #            if len(indices) >= 1:
    #                points_indices.append(indices[0])
    #                break
            
    #if len(points_indices) > 0:
    #    return cloud.select_by_index(points_indices)
               
    #return cloud

    ##            # Testar se ha vizinhos na nuvem para aquele ponto, se sim adicionar o ponto na nova nuvem                
    ##            [points, indices, _] = tree.search_knn_vector_3d(, 2*voxel_size)
    ##            if len(indices) >= 1:
    ##                cloud2.points.append(cloud.points[indices[0]])
    ##                cloud2.colors.append(cloud.colors[indices[0]])
    ##                break

    ##return cloud2
#######################################################################################################
def create_sfm_file(name, images_list, Ts, k=np.identity(3, float), Tcam=np.identity(4, float), only_write=False):
    fx = np.asarray(k)[0][0]
    fy = np.asarray(k)[1][1]
    cx = np.asarray(k)[0][2] 
    cy = np.asarray(k)[1][2]

    sfm = open(name, 'w')
    sfm.write(str(len(Ts))+"\n\n")
    
    pose = np.identity(4, float)
    for i, T in enumerate(Ts):
        if only_write:
            pose = T
        else:
            pose = np.matmul(Tcam, np.linalg.inv(T))

        ps = images_list[i].split('\\')
        linha  = os.path.join(ps[-2], ps[-1]) + " "
        linha += str(np.asarray(pose)[0][0]) + " " + str(np.asarray(pose)[0][1]) + " " + str(np.asarray(pose)[0][2]) + " "
        linha += str(np.asarray(pose)[1][0]) + " " + str(np.asarray(pose)[1][1]) + " " + str(np.asarray(pose)[1][2]) + " "
        linha += str(np.asarray(pose)[2][0]) + " " + str(np.asarray(pose)[2][1]) + " " + str(np.asarray(pose)[2][2]) + " "
        linha += str(np.asarray(pose)[0][3]) + " " + str(np.asarray(pose)[1][3]) + " " + str(np.asarray(pose)[2][3]) + " "
        linha += str(fx) + " " + str(fy) + " " + str(cx) + " " + str(cy) + "\n"

        sfm.write(linha)

    sfm.close()
#######################################################################################################
def assemble_sfm_lines(images_list, Ts, k=np.identity(3, float)):
    fx = np.asarray(k)[0][0]
    fy = np.asarray(k)[1][1]
    cx = np.asarray(k)[0][2] 
    cy = np.asarray(k)[1][2]

    linhas = []
    for i, pose in enumerate(Ts):
        linha  = images_list[i] + " "
        linha += str(np.asarray(pose)[0][0]) + " " + str(np.asarray(pose)[0][1]) + " " + str(np.asarray(pose)[0][2]) + " "
        linha += str(np.asarray(pose)[1][0]) + " " + str(np.asarray(pose)[1][1]) + " " + str(np.asarray(pose)[1][2]) + " "
        linha += str(np.asarray(pose)[2][0]) + " " + str(np.asarray(pose)[2][1]) + " " + str(np.asarray(pose)[2][2]) + " "
        linha += str(np.asarray(pose)[0][3]) + " " + str(np.asarray(pose)[1][3]) + " " + str(np.asarray(pose)[2][3]) + " "
        linha += str(fx) + " " + str(fy) + " " + str(cx) + " " + str(cy) + "\n"
        linhas.append(linha)

    return linhas

#######################################################################################################
def read_sfm_file(name):
    file = open(name, 'r')
    lines = file.readlines()[2:]

    transforms = []
    for n,line in enumerate(lines):
        l = line.split()
        T = np.array([[l[1], l[2], l[3], l[10]],
                      [l[4], l[5], l[6], l[11]],
                      [l[7], l[8], l[9], l[12]],
                      [0, 0, 0, 1]], dtype=np.float)
        transforms.append(T)

    return transforms
#######################################################################################################
def remove_floor(cloud, height=10):
    cloud2 = o3d.geometry.PointCloud()
    for i, p in enumerate(cloud.points):
        if p[1] > height:
            cloud2.points.append(cloud.points[i])
            cloud2.colors.append(cloud.colors[i])
            cloud2.normals.append(cloud.normals[i])

    return cloud2
#######################################################################################################
def check_angle(T1, T2):    
    z = np.array([[0],[0],[1]], dtype=np.float)
    v1 = T1[0:3, 0:3].dot(z)/np.linalg.norm(z)
    v2 = T2[0:3, 0:3].dot(z)/np.linalg.norm(z)
    v1_ = np.squeeze(np.asarray(v1))
    v2_ = np.squeeze(np.asarray(v2))
    a = np.arccos( v1_.dot(v2_) )
    b = abs(np.rad2deg(a))

    return b
#######################################################################################################
def find_transform_from_GPS(gps_s, gps_t, ambiente_interno=True):    
    T = np.identity(4, float)
    # Se nao e ambiente interno, utiliza as coordenadas de GPS obtidas
    if not ambiente_interno:        
        source = [float(a) for a in gps_s]
        target = [float(a) for a in gps_t]
        # Garante que nenhuma coordenada e 0
        if source[0] != 0 and target[0] != 0:
            # Diferenca em latitude e longitude entre os pontos
            # A principio o norte (eixo Z) esta apontado para latitude positiva, eixo X lata longitude positiva
            dlat = target[0] - source[0]
            dlon = target[1] - source[1]
            dalt = target[2] - source[2] if target[2] - source[2] > 15 else 0
            # Converte para metros com formula consagrada e retorna transformacao
            dz = dlat*1.113195e5
            dx = dlon*1.113195e5
            T[0][3] = -dx
            T[2][3] = -dz

    return T
#######################################################################################################
def enclose_fov(c, pose, hor=70, ver=15):
    cloud = o3d.geometry.PointCloud()
    c.transform(pose)
    for i, p in enumerate(c.points):
        if abs(math.atan2(p[0], p[2])) <= math.radians(hor/2) and abs(math.atan2(p[1], p[2])) <= math.radians(ver/2):
            cloud.points.append(p)
            cloud.colors.append(c.colors[i])
            if len(c.normals) > 0:
                cloud.normals.append(c.normals[i])

    cloud.transform(np.linalg.inv(pose))
    return cloud
#######################################################################################################
def blueprint(cl, poses, fl, scale):
    # Nome dos scans para indicar em qual lugar tiramos aquela pose no json
    folders_list = [fo.split('\\')[-1] for fo in fl]
    ### Separando a nuvem em clusters perpendiculares ao eixo y - y negativo para cima
    ###
    cloud = copy.deepcopy(cl)
    # Filtrando a altura que vai entrar na roda (acima do robo)
    altura_considerada = 0.5
    ys = np.asarray(cloud.points)[:, 1]
    indices = np.squeeze(np.argwhere(ys >= altura_considerada))
    cloud.points = o3d.utility.Vector3dVector(np.asarray(cloud.points)[indices])
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[indices])
    if len(cloud.normals) > 1:
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals)[indices])
    # Filtrando pontos longe de todos os centros passados
    bp_radius_limit = 30
    centers_cloud = o3d.geometry.PointCloud()
    for i, p in enumerate(poses):
        centers_cloud.points.append(np.squeeze(np.linalg.inv(p)[0:3, 3]))
    dists = cloud.compute_point_cloud_distance(centers_cloud)
    indices = np.squeeze(np.argwhere(np.asarray(dists) < bp_radius_limit))
    cloud.points = o3d.utility.Vector3dVector(np.asarray(cloud.points)[indices])
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[indices])
    if len(cloud.normals) > 1:
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals)[indices])
    ### Definir aqui quantos pixels por metro quadrado de planta baixa
    # Dimensoes maximas na nuvem de pontos para saber a quantidade relativa de pixels necessarios
    # w esta para X e h esta para Z
    pixels_side_base = 1000
    points = np.asarray(cloud.points)
    xlims = [np.min(points[:, 0]), np.max(points[:, 0])]
    zlims = [np.min(points[:, 2]), np.max(points[:, 2])]
    xa = xlims[1] - xlims[0] # Comprimento para area em X
    za = zlims[1] - zlims[0] # Comprimento para area em Z
    w = pixels_side_base
    h = int(w * za/xa)
    pixels_meter = w/xa
    bp_points = 100*np.ones((w, h, 3), dtype=float) # Guardar os pontos mais altos nos bins    
    blueprint_im = np.zeros((h, w, 3), dtype=float) # Criando imagem da planta baixa e colorindo com a matriz de pontos
    colors = np.asarray(cloud.colors)
    for i, p in enumerate(cloud.points):
        u = int(abs(p[0] - xlims[0])/xa * w)
        v = h - int(abs(p[2] - zlims[0])/za * h)
        if 0 <= u < w and 0 <= v < h:
            if(p[1] < bp_points[u, v, 1]):
                bp_points[u, v, :] = p
                blueprint_im[v, u, :] = 255*colors[i]
    # Desenhar pontos de aquisicao e salvar local em vetor:
    json_out = {}
    json_out['scans'] = []
    for i, p in enumerate(poses):
        x, z = np.linalg.inv(p)[0, 3], np.linalg.inv(p)[2, 3]
        u = int(abs(x - xlims[0])/xa * w)
        v = h - int(abs(z - zlims[0])/za * h)
        if 0 <= u < w and 0 <= v < h:
            json_out['scans'].append({'nome':folders_list[i], 'x':int(scale[0]/w*u), 'y':int(scale[1]/h*v)})
            indice_scan = int(folders_list[i][4:])
            cv2.circle(blueprint_im, (u,v), 12, (255, 100,   0), thickness=30)
            cv2.circle(blueprint_im, (u,v), 20, (255, 255, 255), thickness=4 )
            cv2.putText(blueprint_im, f'{indice_scan:02d}', (u-12,v+7), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=2)

    return np.uint8(blueprint_im), json_out
#######################################################################################################
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#######################################################################################################