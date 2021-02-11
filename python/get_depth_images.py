import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
from skimage.util import img_as_uint, img_as_int

## PATH
path  = "C:/Users/vinic/Desktop/aquisicao1/"
nomes = ('pf_001', 'pf_002', 'pf_003', 'pf_004', 'pf_005')
for n,nome in enumerate(nomes):
    print('Obtendo Imagem D nuvem {}'.format(n+1))
    ## Ler o ply e criar mesh - NAO ESTOU USANDO MESH, direto mesmo a nuvem densa
    ptc = o3d.io.read_point_cloud(os.path.join(path, nome+".ply"))
    ptcv = ptc.voxel_down_sample(voxel_size=0.02)
    ptcv.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    #ptcv.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=-1, max_nn=100))
    #ptcv.orient_normals_towards_camera_location()
    #radii = [0.005, 0.01, 0.02, 0.04]
    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(ptcv, o3d.utility.DoubleVector(radii))
    #mesh.filter_smooth_laplacian(10, 0.5)

    ## Projetar nuvem de pontos na imagem
    w, h = 1920, 1080
    K = np.array([[1427.1, -0.063, 987.9],[0.041, 1449.4, 579.4],[0, 0, 1]])
    #K = np.array([[1427.1, 0, w/2],[0, 1449.4, h/2],[0, 0, 1]])
    toff = np.array([0.0077, 0.0329, 0.0579])
    #toff = np.array([0,0,0])
    scale = 1000
    depth_img = np.zeros(shape=(h, w), dtype=np.float)
    for t in np.arange(len(ptc.points)):
        p = K.dot(ptc.points[t]+toff)
        p=p/p[2]
        if p[0] < w and p[0] >= 0 and p[1] < h and p[1] > 0:
            depth_img[np.int(p[1]), np.int(p[0])] = scale*ptc.points[t][2]
    ks = 5
    #for u in np.arange(ks+1, w-ks-1):
    #    for v in np.arange(ks+1, h-ks-1):
    #        if depth_img[v, u] == 0:
    #            for i in np.arange(u-ks, u+ks):
    #                for j in np.arange(v-ks, v+ks):
    #                    if depth_img[j, i] is not 0:
    #                        depth_img[v, u] = depth_img[j, i]
    #                        break
    #depth_img = depth_img2
    cv2.imwrite(os.path.join(path, 'depth', nome+".png"), depth_img.astype(np.uint16))