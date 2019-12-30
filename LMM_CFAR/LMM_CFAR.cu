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


__device__ void  Memcpy(double *im, double *data, int row, int col, int r_c, int r_g, int n);

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
    int size = r_c*r_c - r_g*r_g;
    for(int i = 0;i<size;i++)
    {
        // data[i] = 1.0;
        // printf("%.1f\n", data[i]);
    }
}

__global__ void CFAR_Gamma(double *im, double *T, int r_c, int r_g, int m, int n) {
    // n_pad为填充后图像的列数， n为原图像的列数
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    int size = (r_c*r_c-r_g*r_g)*4;int n_pad = n + 2*r_c;
    double clutter_sum = 0, I_C = 0, I = 0, *clutter;
    __shared__  double data[4600];
    if(row < m && col < n)
    {
        int index = threadIdx.x + threadIdx.y*blockDim.x;
        clutter =  &data[index*size];
        Memcpy(im, clutter, row, col, r_c, r_g, n_pad);
        int number = size * 0.7;
        for(int i = 0; i< number; i++)
        {
            clutter_sum += clutter[i];
        }
        I_C = clutter_sum/number;
        I = im[row*n_pad+col];
        T[row*n+col] = I/I_C; 
        if(row==30&&col==30)
        {
            printf("%f", im[row*n_pad+col]);
        }
    }
}
__device__ void  Memcpy(double *im, double *data, int row, int col, int r_c, int r_g, int n)
{   
    //上部杂波 5x30
    int index = 0;
    for(int i = row-r_c;i<row-r_g;i++)
    {
        for(int j=col-r_c;j<=col+r_c;j++)
        {
            data[index] = im[i*n+j];   
            index += 1;
        }
    }
    //下部杂波 5x30
    for(int i = row+r_g+1;i<=row+r_c;i++)
    {
        for(int j=col-r_c;j<=col+r_c;j++)
        {
            data[index] = im[i*n+j];   
            index += 1;
        }
       
    }
    //左侧杂波20x5
    for(int i = row-r_g;i<=row+r_g;i++)
    {
        for(int j = col-r_c;j<col-r_g;j++)
        {
            data[index] = im[i*n+j];
            index += 1;
        }
    }
    //右侧杂波20x5
    for(int i = row-r_g;i<=row+r_g;i++)
    {
        for(int j = col+r_g+1;j<=col+r_c;j++)
        {
            data[index] = im[i*n+j];
            index += 1;
        }
    }
}



int main(int argc, char *argv[])
{
    double **im, *im_pad, *im_dev, *data_dev, *T, *result, threshold;
    int ch, opt_index, channels,m,n;    // opt_index为选项在long_options中的索引
    const char *optstring = "d:c:g:";
    int r_c = 15, r_g = 10;threshold = 3.7;
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
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int i=0;i<deviceCount;i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp,i);
        cout << "GPU device:" << i << ": " << devProp.name <<endl;
        cout << "global memory: " << devProp.totalGlobalMem / 1024 / 1024 << "MB" <<endl;
        cout << "SM number:" << devProp.multiProcessorCount <<endl;
        cout << "shared memory:" << (devProp.sharedMemPerBlock / 1024.0) <<"KB"<<endl;
        cout << "block max_thread:" << devProp.maxThreadsPerBlock <<endl;
        cout << "registers per Block:" << devProp.regsPerBlock <<endl;
        cout << "SM max theads:" << devProp.maxThreadsPerMultiProcessor <<endl;
        cout << "======================================================" <<endl;     
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
        }
    }
    Mat image = ArrayToImage(im, arraydim);
    Mat origin_image = ArrayToMat(im, arraydim);
    // double data[3][3] = { {1,2,3},{4,5,6},{7,8,9} };
    Mat pad_image = PadArray(origin_image,r_c,r_c);
    im_pad = pad_image.ptr<double>(0); 
    int row_pad = pad_image.rows;int col_pad = pad_image.cols;
    dim3 blockdim(3,3);
    dim3 griddim((m+blockdim.x-1)/blockdim.x , (n+blockdim.y-1)/blockdim.y);
    checkCudaErrors(cudaMalloc((void**)&im_dev, sizeof(double)*row_pad*col_pad));
    checkCudaErrors(cudaMalloc((void**)&T, sizeof(double)*m*n));
    checkCudaErrors(cudaMemcpy(im_dev, im_pad, sizeof(double)*row_pad*col_pad, cudaMemcpyHostToDevice));
    result = new double[m*n];
    CFAR_Gamma<<<griddim,blockdim>>>(im_dev, T, r_c, r_g, m, n); //应该传入填充后的图像长宽系数
    cudaThreadSynchronize();
    checkCudaErrors(cudaMemcpy(result, T,  sizeof(double)*m*n, cudaMemcpyDeviceToHost));
    Mat detect_result = Mat::zeros(m, n, CV_8UC1);
    for(int i = 0;i<m;i++)
    {
        for(int j = 0;j<n;j++)
        {
            if(i>15 && j>15)
                //printf("%f ", result[i*n+j]);
            if(result[i*n+j]>threshold)
                detect_result.at<uchar>(i,j) = (unsigned char)255;
            else
                detect_result.at<uchar>(i,j) = (unsigned char)0;
        }
    }
    imshow("gray" , detect_result);
    while(char(waitKey())!='q') {}
    // FreeDoubleArray(im,arraydim);
    image.release();
	return 0 ;
}

