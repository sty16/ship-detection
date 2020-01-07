#include<cuda_runtime.h>
#include"device_launch_parameters.h"
#include<stdio.h>


__global__ void malloc_test(double **a)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i==10&&j==10)
    {
        int N = 10000;
        cudaMalloc((void**)a, sizeof(double)*N);
        for(int i=0;i<N;i++)
        {
            (*a)[i] = i;
        } 
    }
    __syncthreads();
    if(i==11&&j==11)
    {
        printf("%f\n",(*a)[500]);
    }

}

int main()
{
    double **a;
    cudaMalloc((void**)&a, sizeof(double *)); //为a分配显存空间
    dim3 blockdim(3,3);
    dim3 griddim(6,6);
    malloc_test<<<griddim, blockdim>>>(a);
    cudaDeviceSynchronize();
    return 0;
}