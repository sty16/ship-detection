#include<mat_opt.h>
#include<stdio.h>
#include<fstream>
#include<stdint.h>
#include<assert.h>
#include<cuda_runtime.h>
#include"device_launch_parameters.h"
#include<helper_cuda.h>
#include<helper_functions.h>
using namespace cv;
__global__ void lognormal_mixture(double *im, int r_c, int r_g, int k, double Pf, int m, int n) 
{
    /***********************************************************************
    Ship detection based on lognormal mixture models
    INPUT
        im: padding SAR density image
        r_c: radius of the reference window
        r_g: radius of the guard area
        K :number of components
        Pf: false alarm rate
        m : number of rows of input image
        n : number of columns of input image
    OUTPUT
        im_prob:  the cdf of simulate distribution with the im value
    *************************************************************************/
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    printf("%d", blockIdx.y);
    if(j>500)
        printf("(%d,%d)\n", i, j);
    int size = r_c*r_c - r_g*r_g;
    double *data = new double[size];
    for(int i = 0;i<size;i++)
    {
        // data[i] = 1.0;
        // printf("%.1f\n", data[i]);
    }
    delete[] data;
}




int main(int argc, char *argv[])
{
    double **im, *im_pad, *im_dev, *data_dev;
    int ch, opt_index, channels,m,n;    // opt_index为选项在long_options中的索引
    const char *optstring = "d:c:g:";
    int r_c = 15, r_g = 10;
    dim3D arraydim;
    const char *filename = "../data/data.bin";   
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
            case 'c':
                r_c = atoi(optarg); break;
            case 'g':
                r_g = atoi(optarg); break;
            case '?':
                cout<<"Unknown option: "<<(char)optopt<<endl;
                break;
        }
    }
    ifstream infile(filename, ios::in | ios::binary);
    infile.read((char *)&channels, sizeof(int));
    infile.read((char *)&arraydim.m,sizeof(size_t));
    infile.read((char *)&arraydim.n,sizeof(size_t));
    m = (int)arraydim.m; n = (int)arraydim.n;
    cout<<m<<" "<<n<<endl;
    im = new double *[m];
    for(int i=0;i<m;i++)
    {
        im[i] = new double[n];
        for(int j=0;j<n;j++)
        {
            infile.read((char *)&im[i][j], sizeof(double));
            //cout<<im[i][j];
        }
    }
    Mat image = ArrayToImage(im, arraydim);
    Mat origin_image = ArrayToMat(im, arraydim);
    // double data[3][3] = { {1,2,3},{4,5,6},{7,8,9} };
    Mat pad_image = PadArray(origin_image,r_c,r_c);
    im_pad = pad_image.ptr<double>(0); 
    int row = pad_image.rows;int col = pad_image.cols;
    dim3 blockdim(16,16);
    dim3 griddim((m+blockdim.x-1)/blockdim.x , (n+blockdim.y-1)/blockdim.y);
    checkCudaErrors(cudaMalloc((void**)&im_dev, sizeof(double)*row*col));
    checkCudaErrors(cudaMemcpy(im_dev, im_pad, sizeof(double)*row*col, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&data_dev,sizeof(double)*(r_c*r_c-r_g*r_g)*griddim.x*griddim.y*blockdim.x*blockdim.y));
    lognormal_mixture<<<griddim,blockdim>>>(im_dev, r_c, r_g, 3, 0.0005,m,n);
    cudaThreadSynchronize();
    imshow("gray" , image);
    while(char(waitKey())!='q') {}
    // FreeDoubleArray(im,arraydim);
    image.release();
	return 0 ;
}

