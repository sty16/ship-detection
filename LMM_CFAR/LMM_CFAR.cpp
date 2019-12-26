#include<mat_read.h>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<float.h>
#include<unistd.h>
#include<stdlib.h>
#include<getopt.h>
using namespace std;
using namespace cv;


Mat ArrayToImage(double **arraydata, dim3D arraydim);
Mat ArrayToMat(double **arraydata, dim3D arraydim);
Mat PadArray(Mat image, int padrows, int padcolumns);

int main(int argc, char *argv[])
{
    double **im;
    int ch, opt_index;    // opt_index为选项在long_options中的索引
    const char *optstring = "d:v:";
    int r_c = 15, r_g = 10;
    const char *filename = "./radarsat2-tj.mat";   
    const char *variable = "I";
    static struct option long_options[] = {
        {"rc", required_argument, NULL,'c'},
        {"rg", required_argument, NULL,'g'}
    };
    while((ch = getopt_long(argc, argv, optstring, long_options, &opt_index)) != -1)
    {
        switch(ch)
        {
            case 'd':
                filename = optarg; break;
            case 'v':
                variable = optarg; break;
            case 'c':
                r_c = atoi(optarg); break;
            case 'g':
                r_g = atoi(optarg); break;
            case '?':
                cout<<"Unknown option: "<<(char)optopt<<endl;
                break;
        }
    }
    im = ReadDoubleMxArray(filename, variable);
    size_t m,n;
    dim3D arraydim = matGetDim3D(filename, variable);
    Mat image = ArrayToImage(im, arraydim);
    Mat origin_image = ArrayToMat(im, arraydim);
    double data[3][3] = { {1,2,3},{4,5,6},{7,8,9} };
    // Mat C = Mat(3,3,CV_64FC1,data);
    // Mat B = PadArray(C, 2, 2);
    // cout<<B<<endl;
    Mat pad_image = PadArray(origin_image,r_c,r_c);
    imshow("gray" , image);
    while(char(waitKey())!='q') {}
    FreeDoubleArray(im,arraydim);
    image.release();
	return 0 ;
}

Mat ArrayToImage(double **arraydata, dim3D arraydim) 
{
    int m, n;
    m = (int)arraydim.m; n = (int)arraydim.n;
    double im_max = -DBL_MAX;
    double im_min =  DBL_MAX;
    Mat image = Mat::zeros(m, n, CV_8UC1); 
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            if(arraydata[i][j] > im_max)
                im_max = arraydata[i][j];
            if(arraydata[i][j] < im_min)
                im_min = arraydata[i][j];
        }
    }
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            image.at<uchar>(i,j) = (unsigned char) round(255*(arraydata[i][j]-im_min)/(im_max-im_min));
        }
    }
    return image;
}

Mat ArrayToMat(double **arraydata, dim3D arraydim)
{
    int m, n;
    m = (int)arraydim.m; n = (int)arraydim.n;
    Mat image = Mat::zeros(m, n, CV_64FC1); 
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            image.at<double>(i,j) = arraydata[i][j];       
        }
    }
    return image;
}

Mat PadArray(Mat image, int padrows, int padcolumns)
{   
    Mat temp, padimage;
    int m = image.rows; 
    int n = image.cols; 
    temp = Mat::zeros(m, n+2*padcolumns, image.type());
    padimage = Mat::zeros(m+2*padrows, n+2*padcolumns, image.type());
    for(int i=0;i<padcolumns;i++)
    {
        image.col(i).copyTo(temp.col(padcolumns-1-i));
        image.col(n-1-i).copyTo(temp.col(n+padcolumns+i));
    }
    image.copyTo(temp.colRange(padcolumns,padcolumns+n)); // colrange 左闭右开
    for(int i=0;i<padrows;i++)
    {
        temp.row(i).copyTo(padimage.row(padrows-1-i));
        temp.row(m-1-i).copyTo(padimage.row(m+padrows+i));
    }
    temp.copyTo(padimage.rowRange(padrows,padrows+m));
    temp.release();
    return padimage;
}