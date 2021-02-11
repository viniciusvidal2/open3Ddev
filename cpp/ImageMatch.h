#pragma once
#include <iostream>
#include <string>
#include <math.h>

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

class ImageMatch
{
public:
	ImageMatch();
	virtual ~ImageMatch();
	void run();

	void set_folders(string tgt, string src);
	void get_matched_poses(Matrix4d& tgt, Matrix4d& src);

private:
	void get_data();
	void calculate_rootSIFT_features();
	void find_best_match();
	void filter_repeated_kpts(vector<KeyPoint> kt, vector<KeyPoint> ks, vector<DMatch>& m);
	void filter_matches_line_coeff(vector<DMatch>& matches, vector<KeyPoint> kpref, vector<KeyPoint> kpnow, float width, float n);
	Matrix4d find_relative_transform();

	string pasta_src, pasta_tgt;
	vector<string> nomes_nuvens, imagens_tgt, imagens_src;
	vector<Matrix3d> rots_tgt, rots_src;

	Mat K;
	int imcols, imrows;

	vector< vector<KeyPoint> > kpts_tgt, kpts_src;
	vector<Mat> descp_tgt, descp_src;
	vector<KeyPoint> best_kptgt, best_kpsrc;
	vector<DMatch> best_matches;
	int im_src_indice, im_tgt_indice;
};

