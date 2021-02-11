#include "ImageMatch.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
ImageMatch::ImageMatch() {
    // Calibracao da camera
    K = Mat::zeros(3, 3, CV_64FC1);
    // Nome das nuvens
    nomes_nuvens = { "004", "008", "012", "016", "020", "024",
                    "028", "032", "036", "040", "044", "048" };
#pragma omp parallel for
    for (int i = 0; i < nomes_nuvens.size(); i++)
        nomes_nuvens[i] = "pf_" + nomes_nuvens[i] + ".ply";
}
/////////////////////////////////////////////////////////////////////////////////////////////////
ImageMatch::~ImageMatch() {

}
/////////////////////////////////////////////////////////////////////////////////////////////////
void ImageMatch::run() {
    // Obter os dados
    this->get_data();
    // Calcular features
    this->calculate_rootSIFT_features();
    // Match de features
    this->find_best_match();
    // Estimar transformacao pelo melhor match
    //this->find_relative_transform();

    // Apagar toda a memoria que possa atrapalhar quando rodar novamente
    kpts_tgt.clear(); kpts_src.clear();
    descp_tgt.clear(); descp_src.clear();
    best_kptgt.clear(); best_kptgt.clear();
    best_matches.clear();
    rots_tgt.clear(); rots_src.clear();
    nomes_nuvens.clear(); imagens_tgt.clear(); imagens_src.clear();

}
/////////////////////////////////////////////////////////////////////////////////////////////////
void ImageMatch::set_folders(string tgt, string src) {
    pasta_tgt = tgt;
    pasta_src = src;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void ImageMatch::get_data() {
    /// Iniciar leitura das imagens da vista src e da tgt
    ///
    printf("\nLendo ambas as pastas os arquivos .sfm ...");
    vector<string> linhas_tgt, linhas_src;
    string linha;
    int contador_linhas = 0;
    vector<int> linhas_horizontal{ 4, 9, 12, 17, 20, 25, 28, 33, 36, 41, 44, 49, 52, 57, 60 };
    // Lendo pasta 1
    string arquivo_sfm = pasta_tgt + "cameras_ok.sfm";
    ifstream sfm_tgt(arquivo_sfm);
    if (sfm_tgt.is_open()) {
        while (getline(sfm_tgt, linha)) {
            for (auto i : linhas_horizontal) {
                if (contador_linhas > 2 && linha.size() > 4 && (contador_linhas + 1) == i)
                    linhas_tgt.push_back(linha);
            }
            contador_linhas++;
        }
    }
    sfm_tgt.close();
    // Lendo pasta 2
    arquivo_sfm = pasta_src + "cameras_ok.sfm";
    contador_linhas = 0;
    ifstream sfm_src(arquivo_sfm);
    if (sfm_src.is_open()) {
        while (getline(sfm_src, linha)) {
            for (auto i : linhas_horizontal) {
                if (contador_linhas > 2 && linha.size() > 4 && (contador_linhas + 1) == i)
                    linhas_src.push_back(linha);
            }
            contador_linhas++;
        }
    }
    sfm_src.close();

    Vector2f foco, centro_otico;
    for (auto s : linhas_tgt) {
        istringstream iss(s);
        vector<string> splits(istream_iterator<string>{iss}, istream_iterator<string>());
        // Nome
        string nome_fim = splits[0].substr(splits[0].find_last_of('/') + 1, splits[0].size() - 1);
        imagens_tgt.push_back(pasta_tgt + nome_fim);

        // Rotation
        Matrix3d r;
        r << stof(splits[1]), stof(splits[2]), stof(splits[3]),
            stof(splits[4]), stof(splits[5]), stof(splits[6]),
            stof(splits[7]), stof(splits[8]), stof(splits[9]);
        rots_tgt.push_back(r);

        // Foco e centro para matriz K - igual, sempre mesma camera
        foco << stof(splits[13]), stof(splits[14]);
        centro_otico << stof(splits[15]), stof(splits[16]);
    }
    K.at<double>(0, 0) = foco(0); K.at<double>(1, 1) = foco(1);
    K.at<double>(0, 2) = centro_otico(0); K.at<double>(1, 2) = centro_otico(1);
    K.at<double>(2, 2) = 1;
    for (auto s : linhas_src) {
        istringstream iss(s);
        vector<string> splits(istream_iterator<string>{iss}, istream_iterator<string>());
        // Nome
        string nome_fim = splits[0].substr(splits[0].find_last_of('/') + 1, splits[0].size() - 1);
        imagens_src.push_back(pasta_src + nome_fim);

        // Rotation
        Matrix3d r;
        r << stof(splits[1]), stof(splits[2]), stof(splits[3]),
            stof(splits[4]), stof(splits[5]), stof(splits[6]),
            stof(splits[7]), stof(splits[8]), stof(splits[9]);
        rots_src.push_back(r);
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void ImageMatch::calculate_rootSIFT_features() {
    printf("\nCalculando SIFT para todas as imagens e guardando ...");
    // Resize do vetor geral de kpts e descritores para a quantidade de imagens de cada caso
    descp_src.resize(imagens_src.size()); descp_tgt.resize(imagens_tgt.size());
    kpts_src.resize(imagens_src.size()); kpts_tgt.resize(imagens_tgt.size());

    Ptr< xfeatures2d::SIFT > sift = xfeatures2d::SIFT::create();
//#pragma omp parallel for
    for (int i = 0; i < descp_src.size(); i++) {
        // Iniciando Keypoints e Descritores atuais
        vector<KeyPoint> kptgt, kpsrc;
        Mat dtgt, dsrc;

        // Ler a imagem inicial
        Mat imtgt = imread(imagens_tgt[i], IMREAD_COLOR);
        Mat imsrc = imread(imagens_src[i], IMREAD_COLOR);

        // Salvar aqui as dimensoes da imagem para a sequencia do algoritmo
        imcols = imtgt.cols; imrows = imtgt.rows;

        // Descritores SIFT calculados
        sift->detectAndCompute(imtgt, Mat(), kptgt, dtgt);
        sift->detectAndCompute(imsrc, Mat(), kpsrc, dsrc);
        // Calculando somatorio para cada linha de descritores
        Mat dtgtsum, dsrcsum;
        reduce(dtgt, dtgtsum, 1, CV_16UC1);
        reduce(dsrc, dsrcsum, 1, CV_16UC1);
        // Normalizando e passando raiz em cada elementos de linha nos descritores da src
#pragma omp parallel for
        for (int i = 0; i < dsrc.rows; i++) {
            for (int j = 0; j < dsrc.cols; j++) {
                dsrc.at<float>(i, j) = sqrt(dsrc.at<float>(i, j) / (dsrcsum.at<float>(i, 0) + numeric_limits<float>::epsilon()));
            }
        }
        // Normalizando e passando raiz em cada elementos de linha nos descritores da tgt
#pragma omp parallel for
        for (int i = 0; i < dtgt.rows; i++) {
            for (int j = 0; j < dtgt.cols; j++) {
                dtgt.at<float>(i, j) = sqrt(dtgt.at<float>(i, j) / (dtgtsum.at<float>(i, 0) + numeric_limits<float>::epsilon()));
            }
        }

        // Salvando no vetor de keypoints
        kpts_tgt[i] = kptgt;
        kpts_src[i] = kpsrc;

        // Salvando no vetor de cada um os descritores
        descp_tgt[i] = dtgt;
        descp_src[i] = dsrc;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void ImageMatch::find_best_match() {
    printf("\nComparando os descritores para melhor combinacao de imagens ...");
    // Ajustar matriz de quantidade de matches
    MatrixXi matches_count = MatrixXi::Zero(descp_tgt.size(), descp_src.size());
    vector<vector<  vector<DMatch> >> matriz_matches(descp_tgt.size());
    for (int i = 0; i < matriz_matches.size(); i++)
        matriz_matches.at(i).resize(descp_src.size());

    // Matcher de FLANN
    cv::Ptr<DescriptorMatcher> matcher;
    matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    // Para cada combinacao de imagens, fazer match e salvar quantidade final para ver qual
    // a melhor depois
    //#pragma omp parallel for
    for (int i = 0; i < descp_tgt.size(); i++) {
        for (int j = 0; j < descp_src.size(); j++) {
            vector<vector<DMatch>> matches;
            vector<DMatch> good_matches;
            if (!descp_src[j].empty() && !descp_tgt[i].empty()) {
                matcher->knnMatch(descp_src[j], descp_tgt[i], matches, 2);
                for (size_t k = 0; k < matches.size(); k++) {
                    if (matches.at(k).size() >= 2) {
                        if (matches.at(k).at(0).distance < 0.6 * matches.at(k).at(1).distance) // Se e bastante unica frente a segunda colocada
                            good_matches.push_back(matches.at(k).at(0));
                    }
                }
                if (good_matches.size() > 0) {
                    // Filtrar keypoints repetidos
                    //this->filter_repeated_kpts(kpts_tgt[i], kpts_src[j], good_matches);
                    // Filtrar por matches que nao sejam muito horizontais
                    //this->filter_matches_line_coeff(good_matches, kpts_tgt[i], kpts_src[j], imcols, (50) * 3.1415/180.0);

                    // Anota quantas venceram nessa combinacao
                    matches_count(i, j) = good_matches.size();
                    matriz_matches.at(i).at(j) = good_matches;
                }
            }
        }
    }

    cout << "\nMatriz de matches:\n" << matches_count << endl << "\nMaximo de matches: " << matches_count.maxCoeff() << endl;

    // Atraves do melhor separar matches daquelas vistas
    int max_matches = matches_count.maxCoeff();
    for (int i = 0; i < descp_tgt.size(); i++) {
        for (int j = 0; j < descp_src.size(); j++) {
            if (matches_count(i, j) == max_matches) {
                best_matches = matriz_matches.at(i).at(j);
                im_tgt_indice = i; im_src_indice = j;
                break;
            }
        }
    }

    // Libera memoria
    descp_tgt.clear(); descp_src.clear();

    // Pegar somente bons kpts
    vector<KeyPoint> curr_kpts_tgt = kpts_tgt[im_tgt_indice], curr_kpts_src = kpts_src[im_src_indice];
    for (auto m : best_matches) {
        best_kptgt.emplace_back(curr_kpts_tgt[m.trainIdx]);
        best_kpsrc.emplace_back(curr_kpts_src[m.queryIdx]);
    }

    // Plotar imagens
    Mat im1 = imread(imagens_tgt[im_tgt_indice], IMREAD_COLOR);
    Mat im2 = imread(imagens_src[im_src_indice], IMREAD_COLOR);
    for (int i = 0; i < best_kpsrc.size(); i++) {
         int r = rand() % 255, b = rand() % 255, g = rand() % 255;
         circle(im1, Point(best_kptgt[i].pt.x, best_kptgt[i].pt.y), 8, Scalar(r, g, b), FILLED, LINE_8);
         circle(im2, Point(best_kpsrc[i].pt.x, best_kpsrc[i].pt.y), 8, Scalar(r, g, b), FILLED, LINE_8);
    }
    imshow("targetc", im1);
    imshow("sourcec", im2);
    imwrite(pasta_src + "im_tgt.png", im1);
    imwrite(pasta_src + "im_src.png", im2);
    waitKey(1);    

    // Libera memoria
    kpts_tgt.clear(); kpts_src.clear();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void ImageMatch::filter_repeated_kpts(vector<KeyPoint> kt, vector<KeyPoint> ks, vector<DMatch>& m) {
    // Matriz de bins para keypoints de target e source
    const int w = 384, h = 216;
    vector<DMatch> matriz_matches[w][h];

    // Itera sobre os matches pra colocar eles nos bins certos
    for (int i = 0; i < m.size(); i++) {
        KeyPoint ktt = kt[m[i].trainIdx];
        int u = ktt.pt.x / 5, v = ktt.pt.y / 5;
        matriz_matches[u][v].push_back(m[i]);
    }
    // Vetor auxiliar de matches que vao passar no teste de melhor distancia
    vector<DMatch> boas_matches;
    // Procurando na matriz de matches
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            if (matriz_matches[i][j].size() > 0) {
                // Se ha matches e for so uma, adicionar ela mesmo
                if (matriz_matches[i][j].size() == 1) {
                    boas_matches.push_back(matriz_matches[i][j][0]);
                }
                else { // Se for mais de uma comparar a distancia com as outras
                    DMatch mbest = matriz_matches[i][j][0];
                    for (int k = 1; k < matriz_matches[i][j].size(); k++) {
                        if (matriz_matches[i][j][k].distance < mbest.distance) {
                            mbest = matriz_matches[i][j][k];
                        }
                    }
                    // Adicionar ao vetor a melhor opcao para aquele bin
                    boas_matches.push_back(mbest);
                }
            }
            matriz_matches[i][j].clear(); // Ja podemos limpar aquele vetor, ja trabalhamos
        }
    }
    m = boas_matches;
    // Fazer o mesmo agora para as matches que sobraram e kpts da src
    // Itera sobre os matches pra colocar eles nos bins certos
    for (int i = 0; i < boas_matches.size(); i++) {
        KeyPoint kst = ks[m[i].queryIdx];
        int u = kst.pt.x / 5, v = kst.pt.y / 5;
        matriz_matches[u][v].push_back(m[i]);
    }
    // Vetor auxiliar de matches que vao passar no teste de melhor distancia
    vector<DMatch> otimas_matches;
    // Procurando na matriz de matches
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            if (matriz_matches[i][j].size() > 0) {
                // Se ha matches e for so uma, adicionar ela mesmo
                if (matriz_matches[i][j].size() == 1) {
                    otimas_matches.push_back(matriz_matches[i][j][0]);
                }
                else { // Se for mais de uma comparar a distancia com as outras
                    DMatch mbest = matriz_matches[i][j][0];
                    for (int k = 1; k < matriz_matches[i][j].size(); k++) {
                        if (matriz_matches[i][j][k].distance < mbest.distance)
                            mbest = matriz_matches[i][j][k];
                    }
                    // Adicionar ao vetor a melhor opcao para aquele bin
                    otimas_matches.push_back(mbest);
                }
            }
            matriz_matches[i][j].clear(); // Ja podemos limpar aquele vetor, ja trabalhamos
        }
    }

    // Retornando as matches que restaram
    m = otimas_matches;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void ImageMatch::filter_matches_line_coeff(vector<DMatch>& matches, vector<KeyPoint> kpref, vector<KeyPoint> kpnow, float width, float n) {
    // Fazer e calcular vetor de coeficientes para cada ponto correspondente do processo de match
    vector<float> coefs(matches.size());
//#pragma omp parallel for
    for (int i = 0; i < matches.size(); i++) {
        float xr, yr, xn, yn;
        xr = kpref[matches[i].queryIdx].pt.x;
        yr = kpref[matches[i].queryIdx].pt.y;
        xn = kpnow[matches[i].trainIdx].pt.x + width;
        yn = kpnow[matches[i].trainIdx].pt.y;
        // Calcular os coeficientes angulares
        try {
            coefs[i] = (yn - yr) / (xn - xr);
        }
        catch (Exception& e) {
            cerr << endl << e.msg << endl;
        }
        //coefs[i] = (yn - yr) / (xn - xr);
    }
    vector<DMatch> temp;
    for (int i = 0; i < coefs.size(); i++) {
        // Se os matches estao na mesma regiao da foto
        if ((kpref[matches[i].queryIdx].pt.x < width / 2 && kpnow[matches[i].trainIdx].pt.x < width / 2) ||
            (kpref[matches[i].queryIdx].pt.x > width / 2 && kpnow[matches[i].trainIdx].pt.x > width / 2)) {
            // Filtrar o vetor de matches na posicao que os coeficientes estejam fora por ngraus
            if (abs(coefs[i]) < n)
                temp.push_back(matches[i]);
        }
    }
    matches = temp;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Matrix4d ImageMatch::find_relative_transform() {
    printf("\nPegando transformacao relativa entre as imagens ...");
    // Inicia matriz de saida
    Matrix4d T = Matrix4d::Identity();

    // Converter os pontos para o formato certo
    vector<Point2f> kptgt(best_kptgt.size()), kpsrc(best_kpsrc.size());
#pragma omp parallel for
    for (int i = 0; i < best_kptgt.size(); i++) {
        kptgt[i] = best_kptgt[i].pt;
        kpsrc[i] = best_kpsrc[i].pt;
    }

    // Calcular matriz fundamental
    Mat F = findFundamentalMat(kpsrc, kptgt); // Transformacao da src para a tgt
    // Calcular pontos que ficam por conferencia da matriz F
    Matrix3f F_;
    cv2eigen(F, F_);
    vector<Point2f> tempt, temps;
    vector<int> indices_inliers;
    for (int i = 0; i < kpsrc.size(); i++) {
        Vector3f pt{ kptgt[i].x, kptgt[i].y, 1 }, ps = { kpsrc[i].x, kpsrc[i].y, 1 };
        MatrixXf erro = pt.transpose() * F_ * ps;
        if (abs(erro(0, 0)) < 0.2) {
            tempt.push_back(kptgt[i]); temps.push_back(kpsrc[i]);
            indices_inliers.push_back(i);
        }
    }
    kpsrc = temps; kptgt = tempt;

    // Segue so com os inliers dentre os best_kpts
    vector<KeyPoint> temp_kptgt, temp_kpsrc;
    for (auto i : indices_inliers) {
        temp_kptgt.push_back(best_kptgt[i]); temp_kpsrc.push_back(best_kpsrc[i]);
    }
    best_kptgt = temp_kptgt; best_kpsrc = temp_kpsrc;

    // Matriz Essencial
    //  Mat E = K.t()*F*K;
    Mat E = findEssentialMat(kpsrc, kptgt, K);
    // Recupera pose - matriz de rotacao e translacao
    Mat r, t;
    int inliers;
    inliers = recoverPose(E, kpsrc, kptgt, K, r, t);

    cout << "\nInliers:  " << inliers << " de " << best_kpsrc.size() << endl;

    // Passar para Eigen e seguir processo
    T << r.at<double>(Point(0, 0)), r.at<double>(Point(0, 1)), r.at<double>(Point(0, 2)), t.at<double>(Point(0, 0)),
         r.at<double>(Point(1, 0)), r.at<double>(Point(1, 1)), r.at<double>(Point(1, 2)), t.at<double>(Point(1, 0)),
         r.at<double>(Point(2, 0)), r.at<double>(Point(2, 1)), r.at<double>(Point(2, 2)), t.at<double>(Point(2, 0)),
         0, 0, 0, 1;

    // Transformacao de um frame para o outro
    Matrix4d T_frames = Matrix4d::Identity();
    T_frames.block<3,3>(0, 0) = rots_src[im_src_indice] * rots_tgt[im_tgt_indice].inverse();

    // Transformacao final
    T = T * T_frames;

    return T;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void ImageMatch::get_matched_poses(Matrix4d& tgt, Matrix4d& src) {
    tgt = Matrix4d::Identity();
    src = Matrix4d::Identity();
    tgt.block<3, 3>(0, 0) = rots_tgt[im_tgt_indice];
    src.block<3, 3>(0, 0) = rots_src[im_src_indice];
}
/////////////////////////////////////////////////////////////////////////////////////////////////