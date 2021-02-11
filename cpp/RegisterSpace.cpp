#include <iostream>
#include <chrono>
#include <cmath>

#include "open3d/io/PointCloudIO.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/VoxelGrid.h"
#include "open3d/geometry/Geometry3D.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/geometry/VoxelGrid.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/pipelines/registration/ColoredICP.h"
#include "open3d/pipelines/registration/FastGlobalRegistration.h"
#include "open3d/pipelines/registration/CorrespondenceChecker.h"
#include "open3d/pipelines/registration/Feature.h"
#include "open3d/pipelines/registration/TransformationEstimation.h"
#include "open3d/pipelines/registration/GlobalOptimization.h"
#include "open3d/pipelines/registration/GlobalOptimizationConvergenceCriteria.h"
#include "open3d/pipelines/registration/Registration.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/Open3D.h"

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include "ImageMatch.h"

using namespace std;
using namespace Eigen;
using namespace open3d;
using namespace open3d::pipelines;

Matrix4d find_common_area_relative_transform(geometry::PointCloud tgt, geometry::PointCloud src, Matrix4d Ttgt, Matrix4d Tsrc) {
    Matrix4d Trel = Matrix4d::Identity();
    // Encontrar area em comum para a nuvem tgt e src
    geometry::PointCloud temptgt, tempsrc, tt;
    tgt.Transform(Ttgt); src.Transform(Tsrc);
    double fov_rad = 100 * 3.1415/180.0;
    for (int i = 0; i < tgt.points_.size(); i++) {        
        if (tgt.points_[i].z() > 0 && abs(acos(abs(tgt.points_[i].z()) / tgt.points_[i].norm())) < fov_rad/2) {
            temptgt.points_.emplace_back(tgt.points_[i]);
            temptgt.colors_.emplace_back(tgt.colors_[i]);
        }
    }
    tgt.Clear();
    for (int i = 0; i < src.points_.size(); i++) {
        if (src.points_[i].z() > 0 && abs(acos(abs(src.points_[i].z()) / src.points_[i].norm())) < fov_rad/2) {
            tempsrc.points_.emplace_back(src.points_[i]);
            tempsrc.colors_.emplace_back(src.colors_[i]);
        }
    }
    src.Clear();

    temptgt.Transform(Ttgt.inverse()); tempsrc.Transform(Tsrc.inverse());

    // Simplificando por voxels, retirando outliers e acertando normais
    double vs = 0.05;
    auto tgtv = temptgt.VoxelDownSample(vs);
    auto srcv = tempsrc.VoxelDownSample(vs);
    //tgtv->RemoveStatisticalOutliers(50, 3);
    //srcv->RemoveStatisticalOutliers(50, 3);
    tgtv->EstimateNormals(geometry::KDTreeSearchParamKNN(30));
    tgtv->OrientNormalsTowardsCameraLocation();
    srcv->EstimateNormals(geometry::KDTreeSearchParamKNN(30));
    srcv->OrientNormalsTowardsCameraLocation();

    //Eigen::Matrix4d Ts;
    //Ts << -1.00, 0.01, -0.03, 68.56, 0.01, 1.00, -0.01, -0.94, 0.03, -0.01, -1.00, -68.32, 0.00, 0.00, 0.00, 1.00;
    //srcv->Transform(Ts);

    //visualization::Visualizer vis2;
    //vis2.CreateVisualizerWindow("teste", 1400, 700);
    //vis2.AddGeometry(tgtv);
    //vis2.AddGeometry(srcv);
    //vis2.Run();
    //vis2.DestroyVisualizerWindow();
    //vis2.ClearGeometries();

    // Encontrar features fpfhtgtv
    auto tgtfpfh = registration::ComputeFPFHFeature(*tgtv, geometry::KDTreeSearchParamRadius(0.2));
    auto srcfpfh = registration::ComputeFPFHFeature(*srcv, geometry::KDTreeSearchParamRadius(0.2));
    // Encontrar transformada final por metodo fast
    printf("\nRodando transformacao FAST ...");
    auto result_fast = registration::FastGlobalRegistration(*srcv, *tgtv, *srcfpfh, *tgtfpfh, registration::FastGlobalRegistrationOption(1.4, true, true, 0.1, 100, 0.95, 1000));
    srcv->Transform(result_fast.transformation_);
    tt = *srcv + *tgtv;
    auto ttt = tt.VoxelDownSample(vs);
    visualization::Visualizer vis4;
    vis4.CreateVisualizerWindow("teste2", 1400, 700);
    vis4.AddGeometry(ttt);
    vis4.Run();
    vis4.DestroyVisualizerWindow();
    vis4.ClearGeometries();
    printf("\nRodando transformacao ICP intermediario ...");
    auto icp_result = registration::RegistrationICP(*srcv, *tgtv, 3 * vs, Matrix4d::Identity(),
        registration::TransformationEstimationPointToPlane(),
        registration::ICPConvergenceCriteria(9.9999e-07, 9.9999e-07, 50));
   /* vector<reference_wrapper<const registration::CorrespondenceChecker>> checkers;
    auto result_fast = registration::RegistrationRANSACBasedOnFeatureMatching(*srcv, *tgtv, *srcfpfh, *tgtfpfh, 3*vs, registration::TransformationEstimationPointToPlane(), 4, checkers, registration::RANSACConvergenceCriteria());*/
    Trel = icp_result.transformation_;

    srcv->Transform(Trel);
    tt = *srcv + *tgtv;
    ttt = tt.VoxelDownSample(vs);
    visualization::Visualizer vis3;
    vis3.CreateVisualizerWindow("teste3", 1400, 700);
    vis3.AddGeometry(ttt);
    vis3.Run();
    vis3.DestroyVisualizerWindow();
    vis3.ClearGeometries();

    return Trel;
}

int main()
{
    // Nuvens de pontos 
    geometry::PointCloud tgt, src, tgt_match, src_match;

    //// Matrizes iniciais
    //vector<Matrix4d> Ts(3);
    ////Ts[0] << 0.52, -0.01, 0.85, 7.68, 0.00, 1.00, 0.01, -0.08, -0.85, -0.01, 0.52, -6.49, 0.00, 0.00, 0.00, 1.00;;
    //Ts[0] << 0.94, 0.00, 0.34, 28.15, - 0.01, 1.00, 0.04, 0.09, - 0.34, - 0.04, 0.94, - 27.72, 0.00, 0.00, 0.00, 1.00;
    //Ts[1] << -0.03, 0.01, - 1.00, 50.18, 0.03, 1.00, 0.01, - 0.70, 1.00, - 0.03, - 0.03, - 49.47, 0.00, 0.00, 0.00, 1.00;
    //Ts[2] << -1.00, 0.01, - 0.03, 68.56, 0.01, 1.00, - 0.01, - 0.94, 0.03, - 0.01, - 1.00, - 68.32, 0.00, 0.00, 0.00, 1.00;

    // Pastas a trabalhar
    vector<string> pastas(4);
    pastas[0] = "C:/Users/vinic/Desktop/SANTOS_DUMONT_2/galpao/scan2/";
    //pastas[1] = "C:/Users/vinic/Desktop/SANTOS_DUMONT_2/patio/scan3_1/";
    pastas[1] = "C:/Users/vinic/Desktop/SANTOS_DUMONT_2/galpao/scan3/";
    pastas[2] = "C:/Users/vinic/Desktop/SANTOS_DUMONT_2/galpao/scan4/";
    //pastas[3] = "C:/Users/vinic/Desktop/SANTOS_DUMONT_2/patio/scan6/";

    // Objeto da classe de match de imagens
    ImageMatch im;

    // Lendo a nuvem de pontos inicial
    printf("\nLendo a nuvem de pontos ...");
    io::ReadPointCloudFromPLY(pastas[0]+"acumulada.ply", tgt, io::ReadPointCloudOption());

    // Simplificando nuvem por voxel
    printf("\nSimplificando por voxel ...");
    double vs = 0.03;
    auto tgtv = tgt.VoxelDownSample(vs);

    // Retirando outliers
    printf("\nRemovendo Outliers ...");
    tgtv->RemoveStatisticalOutliers(50, 3);

    // Calculando normais e virando para a origem
    printf("\nEstimando normais ...\n");
    tgtv->EstimateNormals(geometry::KDTreeSearchParamKNN(100));
    tgtv->OrientNormalsTowardsCameraLocation();

    // Transformacao de referencia
    Matrix4d Tref = Matrix4d::Identity();

    for (int n = 0; n < pastas.size()-1; n++) {
        // Le imagens dos dois conjuntos, descobre melhor combinacao e retorna transformada 
        // com rotacao razoavel e translacao em escala somente mesmo pela imagem
        printf("\nCalculando transformacao relativa inicial por imagens ...");
        im.set_folders(pastas[n + 1], pastas[n]);
        im.run();

        Matrix4d Ttgt, Tsrc;
        im.get_matched_poses(Ttgt, Tsrc);

        // Lendo as nuvens de pontos
        printf("\nLendo a nuvem de pontos %d de %zu ...", n + 2, pastas.size());
        io::ReadPointCloudFromPLY(pastas[n    ] + "acumulada.ply", tgt_match, io::ReadPointCloudOption());
        io::ReadPointCloudFromPLY(pastas[n + 1] + "acumulada.ply", src      , io::ReadPointCloudOption());

        // Separando area em comum pelas poses encontradas pelas imagens
        printf("\nCalculando area comum das nuvens e transformacao relativa ...");
        Matrix4d Trel = find_common_area_relative_transform(tgt_match, src, Ttgt, Tsrc);

        // Simplificando nuvens por voxel
        printf("\nSimplificando por voxel ...");
        auto srcv = src.VoxelDownSample(vs);

        // Retirando outliers
        printf("\nRemovendo Outliers ...");
        srcv->RemoveStatisticalOutliers(50, 3);

        // Calculando normais e virando para a origem
        printf("\nEstimando normais ...");
        srcv->EstimateNormals(geometry::KDTreeSearchParamKNN(100));
        srcv->OrientNormalsTowardsCameraLocation();

        //// Visualizar estagio atual
        //visualization::Visualizer vis2;
        //vis2.CreateVisualizerWindow("Acc", 1400, 700);
        //vis2.AddGeometry(tgtv);
        //vis2.AddGeometry(srcv);
        //vis2.Run();
        //vis2.DestroyVisualizerWindow();
        //vis2.ClearGeometries();

        // ICP, aqui usa a transformada que leva a referencia de onde paramos na ultima 
        // e a descoberta como relativa entre as nuvens como chute inicial
        auto time = chrono::steady_clock::now();
        int it = 200;
        printf("\nPerformando icp colorido com %d iteracoes ...", it);
        auto icp_result = registration::RegistrationColoredICP(*srcv, *tgtv, 3 * vs, Trel*Tref,
            registration::TransformationEstimationForColoredICP(),
            registration::ICPConvergenceCriteria(9.9999e-09, 9.9999e-09, it));
        printf("\nTempo decorrido: %f ...", chrono::duration<double, milli>(chrono::steady_clock::now() - time).count() / 1000);
        time = chrono::steady_clock::now();

        srcv->Transform(icp_result.transformation_);

        // Somando e tirando vizinhos
        printf("\nSomando pontos novos ...");
        geometry::KDTreeFlann tree;
        vector<int> inds;
        vector<double> dists;
        int neighbors, nmax = 10;
        Vector3d p;
        shared_ptr<geometry::PointCloud> temp(new geometry::PointCloud);
        *temp = *tgtv;
        tree.SetGeometry(*tgtv);
        for (int i = 0; i < srcv->points_.size(); i++) {
            p << srcv->points_[i].x(), srcv->points_[i].y(), srcv->points_[i].z();
            neighbors = tree.SearchRadius(p, 5 * vs, inds, dists);
            if (neighbors < nmax) {
                temp->points_.emplace_back(srcv->points_[i]);
                temp->colors_.emplace_back(srcv->colors_[i]);
                temp->normals_.emplace_back(srcv->normals_[i]);
            }
        }
        *tgtv = *temp;
        printf("\nTempo decorrido: %f ...\n", chrono::duration<double, milli>(chrono::steady_clock::now() - time).count() / 1000);
        time = chrono::steady_clock::now();

        // Visualizar estagio atual
        visualization::Visualizer vis2;
        vis2.CreateVisualizerWindow("Acc", 1400, 700);
        vis2.AddGeometry(tgtv);
        vis2.Run();
        vis2.DestroyVisualizerWindow();
        vis2.ClearGeometries();

        // Guarda para proxima iteracao a transformacao final encontrada
        Matrix4d ticp = icp_result.transformation_;
        Tref = ticp * Trel * Tref;
    }

    ////// Calculando features para match
    //auto time = chrono::steady_clock::now();
    ////printf("\nCalculando fpfh ...");
    ////auto tgtfpfh = registration::ComputeFPFHFeature(*tgtv);
    ////auto srcfpfh = registration::ComputeFPFHFeature(*srcv);
    ////printf("\nTempo decorrido: %f ...", chrono::duration<double, milli>(chrono::steady_clock::now() - time).count() / 1000);
    ////time = chrono::steady_clock::now();

    ////// Match por fast
    ////printf("\nEstimando transformacao por metodo fast ...");
    ////auto result_fast  = registration::FastGlobalRegistration(*srcv, *tgtv, *srcfpfh, *tgtfpfh, registration::FastGlobalRegistrationOption());
    ////printf("\nTempo decorrido: %f ...", chrono::duration<double, milli>(chrono::steady_clock::now() - time).count() / 1000);
    ////time = chrono::steady_clock::now();
    //////auto corresp_eval = registration::EvaluateRegistration(*srcv, *tgtv, 0.10, result_fast.transformation_);
    //////pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength corel;
    //////corel.Check(*srcv, *tgtv, result_fast.correspondence_set_, result_fast.transformation_);

    //// Agora refinando por ICP
    //int it = 200;
    //printf("\nPerformando icp colorido com %d iteracoes ...", it);
    //auto icp_result = registration::RegistrationColoredICP(*srcv, *tgtv, 3*vs, T,
    //                                                 registration::TransformationEstimationForColoredICP(), 
    //                                                 registration::ICPConvergenceCriteria(9.9999e-09, 9.9999e-09, it));
    //printf("\nTempo decorrido: %f ...", chrono::duration<double, milli>(chrono::steady_clock::now() - time).count() / 1000);
    //time = chrono::steady_clock::now();

    //srcv->Transform(icp_result.transformation_);

    ////// Visualizando
    ////printf("\nVisualizando ...");
    //visualization::Visualizer vis;
    ////vis.CreateVisualizerWindow("Teste registro", 1400, 700);
    ////vis.AddGeometry(tgtv);
    ////vis.AddGeometry(srcv);
    ////vis.Run();
    ////vis.DestroyVisualizerWindow();
    ////printf("\nTempo decorrido: %f ...", chrono::duration<double, milli>(chrono::steady_clock::now() - time).count() / 1000);
    ////time = chrono::steady_clock::now();

    //// Somando e tirando vizinhos
    //printf("\nSomando pontos novos ...");
    //geometry::KDTreeFlann tree;
    //vector<int> inds;
    //vector<double> dists;
    //int neighbors, nmax = 10;
    //Vector3d p;
    //shared_ptr<geometry::PointCloud> temp(new geometry::PointCloud);
    //*temp = *tgtv;
    //tree.SetGeometry(*tgtv);
    //for (int i = 0; i < srcv->points_.size(); i++) {
    //    p << srcv->points_[i].x(), srcv->points_[i].y(), srcv->points_[i].z();
    //    neighbors = tree.SearchRadius(p, 5 * vs, inds, dists);
    //    if (neighbors < nmax) {
    //        temp->points_.emplace_back(srcv->points_[i]);
    //        temp->colors_.emplace_back(srcv->colors_[i]);
    //    }
    //}
    //*tgtv = *temp;
    //printf("\nTempo decorrido: %f ...", chrono::duration<double, milli>(chrono::steady_clock::now() - time).count() / 1000);
    //time = chrono::steady_clock::now();

    io::WritePointCloudToPLY("C:/Users/vinic/Desktop/SANTOS_DUMONT_2/patio/registro_final2.ply", *tgtv, io::WritePointCloudOption());

    return 0;
}