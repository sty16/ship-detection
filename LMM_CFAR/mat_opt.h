#include<mat_read.h>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<float.h>
#include<unistd.h>
#include<stdlib.h>
#include<getopt.h>
#pragma once
using namespace std;
using namespace cv;


Mat ArrayToImage(double **arraydata, dim3D arraydim);
Mat ArrayToMat(double **arraydata, dim3D arraydim);
Mat PadArray(Mat image, int padrows, int padcolumns);

