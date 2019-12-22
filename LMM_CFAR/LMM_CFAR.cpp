#include<mat_read.h>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<float.h>
using namespace std;
using namespace cv;


int main()
{
    double **im;
    const char *filename = "./radarsat2-tj.mat";   
    im = ReadDoubleMxArray(filename,"I");
    size_t m,n;
    dim3D arraydim = matGetDim3D(filename,"I");
    m = arraydim.m; n = arraydim.n;
    double im_max = -DBL_MAX;
    double im_min =  DBL_MAX;
    Mat image = Mat::zeros((int)m,(int)n,CV_8UC1);
    for(size_t i=0;i<m;i++)
    {
        for(size_t j=0;j<n;j++)
        {
            if(im[i][j] > im_max)
                im_max = im[i][j];
            if(im[i][j] < im_min)
                im_min = im[i][j];
        }
    }
    for(int i=0;i<(int)m;i++)
    {
        for(int j=0;j<(int)n;j++)
        {
            image.at<uchar>(i,j) = (unsigned char) round(255*(im[i][j]-im_min)/(im_max-im_min));
        }
    }
    imshow("gray" , image);
    while(char(waitKey())!='q') {}
	return 0 ;
       
}