#include<mat_opt.h>
#include<stdio.h>
#include<fstream>
#include<stdint.h>
#include<assert.h>
#include<time.h>
#include<unistd.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<thrust/sort.h>
#include"device_launch_parameters.h"
#include<helper_cuda.h>
#include<helper_functions.h>

#define MAX_DEPTH       16
#define INSERTION_SORT  32
using namespace cv;


__device__ void  Memcpy(double *im, double *data, int row, int col, int r_c, int r_g, int n);
__device__ void selection_sort(double *data, int left, int right);
__global__ void simple_quicksort(double *data, int left, int right, int depth);


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
    printf("ok");
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    int size = (r_c*r_c-r_g*r_g)*4;int n_pad = n + 2*r_c;
    double clutter_sum = 0, I_C = 0, I = 0, *clutter;
    __shared__  double data[4600];
    if(row < m && col < n)
    {
        int index = threadIdx.x + threadIdx.y*blockDim.x;
        row = row + r_c; col = col + r_c; // 延拓后数据的索引位置发生改变
        clutter =  &data[index*size];
        Memcpy(im, clutter, row, col, r_c, r_g, n_pad);
        // cudaStream_t s;
        // cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        // simple_quicksort<<<1, 1, 0, s>>>(clutter, 0, size-1, 0);
        // cudaStreamDestroy(s);
        selection_sort(clutter, 0, size-1);
        int number = size * 0.65;
        for(int i = 0; i< number; i++)
        {
            clutter_sum += clutter[i];
        }
        I_C = clutter_sum/number;
        for(int i = row-1; i <= row+1;i++)
        {
            for(int j = col-1;j <= col+1;j++)
            {
                I += im[i*n_pad+col];
            }
        }
        I = I/9;
        T[(row-r_c)*n+(col-r_c)] = I/I_C; 
        if(row==30&&col==30)
        {
            for(int i=0;i<size;i++)
            {
                printf("%f ", clutter[i]);
            }
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

__device__ void selection_sort(double *data, int left, int right)
{
   for(int i = left; i <= right; i++)
   {
        double min_val = data[i]; 
        int min_idx = i;
        for(int j = i+1; j <= right; j++)
        {
            double val_j = data[j];
            if(val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }
        if(i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
   } 
}

__global__  void simple_quicksort(double *data, int left, int right, int depth)
{

    if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT)
    {
        selection_sort(data, left, right);
        return;
    }
    double *lptr = &data[left];
    double *rptr = &data[right];
    double  pivot = data[(left+right)/2];
    while(lptr <= rptr)
    {
        double lval = *lptr;
        double rval = *rptr;
        while(lval < pivot)
        {
            lptr++;
            lval = *lptr;
        }
        while(rval > pivot)
        {
            rptr--;
            rval = *rptr;
        }
        if(lptr <= rptr)
        {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }
    int nright = rptr - data;
    int nleft  = lptr - data;
    if (left < (rptr-data))
    {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right)
    {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}

int main(int argc, char *argv[])
{
    double **im, *im_pad, *im_dev, *T, *result, threshold;
    int ch, opt_index, channels,m,n;    // opt_index为选项在long_options中的索引
    const char *optstring = "d:c:g:";
    int r_c = 15, r_g = 10;threshold = 4.7;
    dim3D arraydim;
    const char *filename = "../data/data.bin";   
    clock_t start,end;
    start = clock();
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
        printf("GPU device has compute capabilities (SM %d.%d)\n", devProp.major, devProp.minor);
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
    cudaStream_t detect;
    cudaStreamCreate(&detect);
    CFAR_Gamma<<<griddim, blockdim, 0, detect>>>(im_dev, T, r_c, r_g, m, n); //应该传入未填充的图像长宽系数
    cudaStreamSynchronize(detect);
    cudaStreamDestroy(detect);
    cudaThreadSynchronize();
    checkCudaErrors(cudaMemcpy(result, T,  sizeof(double)*m*n, cudaMemcpyDeviceToHost));
    Mat detect_result = Mat::zeros(m, n, CV_8UC1);
    for(int i = 0;i<m;i++)
    {
        for(int j = 0;j<n;j++)
        {
            if(result[i*n+j]>threshold)
                detect_result.at<uchar>(i,j) = (unsigned char)255;
            else
                detect_result.at<uchar>(i,j) = (unsigned char)0;
        }
    }
    end = clock();
    imshow("origin" , image); 
    imshow("detected" , detect_result);
    while(char(waitKey())!='q') 
	{    
	}
    // FreeDoubleArray(im,arraydim);
    image.release();
    cout<<"GPU用时："<<(float)(end-start)/CLOCKS_PER_SEC<<end<<endl;
	return 0 ;
}

