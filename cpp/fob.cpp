#include <iostream>
#include <string>
#include <math.h>
#include <cmath>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace Eigen;
using namespace std;
using namespace cv;

Mat image, im360; // Imagem, todas com as mesmas largura e altura, e imagem 360
double step_deg;

///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////// Funcao FOB, nao simplificada, na sequencia logica //////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
// Entrada: Uma DMatch verdadeira entre duas imagens vizinhas, apos filtradas, das imagens 1 e 2
//          Os keypoints das duas imagens 1 e 2
//          Todos os parametros a serem otimizados
// Parametros a otimizar: fx, fy, cx, cy, yaw (y), roll (r), pitch (p) de cada camera
//  `-> aqui cabe uma ressalva, talvez podemos fixar o cx e cy igual para todas as cameras!
// Saida: erro entre projecoes dos pontos em match na panoramica, como preferir, se modulo, se outra, nao sei
//
// Fazer isso para todos os pontos de match entre todas as fotos, possivelmente limitar esse numero de pontos
//

// Keypoints
vector<KeyPoint> kpts1, kpts2;
KeyPoint kp1, kp2;
// Match em questao
DMatch m;
kp1 = kpts1[m.trainIdx];
kp2 = kpts2[m.queryIdx];

// Pose da CAMERA 1, so existe aqui rotacao, vamos suprimir as translacoes 
// pois serao irrelevantes e serao compensadas por outros dados
double yaw1, roll1, pitch1; // Variaveis de entrada, so estao aqui para representar
Matrix3d r1;
r1 = AngleAxisd(yaw1, Vector3d::UnitY()) * AngleAxisd(roll1, Vector3d::UnitZ()) * AngleAxisd(pitch1, Vector3d::UnitX());

// Vamos criar o frustrum da CAMERA 1, assim como nas nossas funcoes, como o desenho do github
// Supondo a variavel F e o raio da esfera como F = R = 1, nao interferiu nas experiencias
double cx1, cy1, fx1, fy1; // Essas sao entradas da funcao, so estao aqui para representar
double dx1 = cx1 - double(image.cols) / 2, dy1 = cy1 - double(image.rows) / 2;

double maxX = (float(image.cols) - 2*dx1) / (2.0 * fx1);
double minX = (float(image.cols) + 2*dx1) / (2.0 * fx1);
double maxY = (float(image.rows) - 2*dy1) / (2.0 * fy1);
double minY = (float(image.rows) + 2*dy1) / (2.0 * fy1);

double F = 1;

Vector3d p, p1, p2, p3, p4, p5, pCenter;
p << 0, 0, 0;
p1 = r1 * p; // Nao usado a principio, pode omitir
p << minX, minY, F;
p2 = r1 * p;
p << maxX, minY, F;
p3 = r1 * p; // Nao usado a principio, pode omitir
p << maxX, maxY, F;
p4 = r1 * p;
p << minX, maxY, F;
p5 = r1 * p;
p << 0, 0, F;
pCenter = r1 * p; // Nao usado a principio, pode omitir

// Ponto no frustrum 3D correspondente a feature na imagem 1 em 2D
Vector3d ponto3d = p5 + (p4 - p5) * kp1.pt.x / image.cols + (p2 - p5) * kp1.pt.y / image.rows;
// Latitude e longitude no 360
double lat = 180 / 3.1415 * (acos(ponto3d[1] / ponto3d.norm())), lon = -180 / 3.1415 * (atan2(ponto3d[2], ponto3d[0]));
lon = (lon < 0) ? lon += 360.0 : lon;
int u = int(lon / step_deg), v = im360.rows - 1 - int(lat / step_deg);
u = (u >= im360.cols) ? im360.cols - 1 : u; // Nao deixar passar do limite de colunas por seguranca
u = (u < 0) ? 0 : u;
v = (v >= im360.rows) ? im360.rows - 1 : v; // Nao deixar passar do limite de linhas por seguranca
v = (v < 0) ? 0 : v;
// Ponto na imagem 360 devido a camera 1, finalmente apos as contas, armazenar
Vector2d ponto_fc1{ u, v };

// ------------------------------------------------------------------------------------------------

// Pose da CAMERA 2, so existe aqui rotacao, vamos suprimir as translacoes 
// pois serao irrelevantes e serao compensadas por outros dados
double yaw2, roll2, pitch2; // Variaveis de entrada, so estao aqui para representar
Matrix3d r2;
r2 = AngleAxisd(yaw2, Vector3d::UnitY()) * AngleAxisd(roll2, Vector3d::UnitZ()) * AngleAxisd(pitch2, Vector3d::UnitX());

// Vamos criar o frustrum da CAMERA 2, assim como nas nossas funcoes, como o desenho do github
// Supondo a variavel F e o raio da esfera como F = R = 1, nao interferiu nas experiencias
double cx2, cy2, fx2, fy2; // Essas sao entradas da funcao, so estao aqui para representar
double dx2 = cx2 - double(image.cols) / 2, dy2 = cy2 - double(image.rows) / 2;

maxX = (float(image.cols) - 2 * dx2) / (2.0 * fx2);
minX = (float(image.cols) + 2 * dx2) / (2.0 * fx2);
maxY = (float(image.rows) - 2 * dy2) / (2.0 * fy2);
minY = (float(image.rows) + 2 * dy2) / (2.0 * fy2);

p << 0, 0, 0;
p1 = r1 * p; // Nao usado a principio, pode omitir
p << minX, minY, F;
p2 = r1 * p;
p << maxX, minY, F;
p3 = r1 * p; // Nao usado a principio, pode omitir
p << maxX, maxY, F;
p4 = r1 * p;
p << minX, maxY, F;
p5 = r1 * p;
p << 0, 0, F;
pCenter = r1 * p; // Nao usado a principio, pode omitir

// Ponto no frustrum 3D correspondente a feature na imagem 2 em 2D
ponto3d = p5 + (p4 - p5) * kp2.pt.x / image.cols + (p2 - p5) * kp2.pt.y / image.rows;
// Latitude e longitude no 360
lat = 180 / 3.1415 * (acos(ponto3d[1] / ponto3d.norm())); lon = -180 / 3.1415 * (atan2(ponto3d[2], ponto3d[0]));
lon = (lon < 0) ? lon += 360.0 : lon;
u = int(lon / step_deg); v = im360.rows - 1 - int(lat / step_deg);
u = (u >= im360.cols) ? im360.cols - 1 : u; // Nao deixar passar do limite de colunas por seguranca
u = (u < 0) ? 0 : u;
v = (v >= im360.rows) ? im360.rows - 1 : v; // Nao deixar passar do limite de linhas por seguranca
v = (v < 0) ? 0 : v;
// Ponto na imagem 360 devido a camera 2, finalmente apos as contas, armazenar
Vector2d ponto_fc2{ u, v };

/// RESULTADO FINAL, para ir formando a FOB, com o somatorio do erro entre os pontos
///
double erro = (ponto_fc1 - ponto_fc2).norm();