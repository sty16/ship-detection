#include"log_normal.h"

__device__ log_param f_EM(double *x, int length, int idx)
{
    log_param result;
    double var_init = var(x, length);
    int maxIter = 25;
    curandState localState;
    curand_init(idx, idx, 0, &localState);
    // set initial value
    for(int i=0;i<3;i++)
    {
        if(idx==10 || idx == 11){
            float a  = curand_uniform(&localState);
            int index = a*(length - 1);
            result.mu[i] = x[index];
            result.var[i] =  var_init;
            result.p[i] = 1.0/3.0;   // pay attention to the decimal
        }
    }
    // GMM implementation c++ https://www.cnblogs.com/luxiaoxun/archive/2013/05/10/3071672.html
    for(int i=0;i<maxIter;i++)
    {

    }
    return result; 
}

__device__ double var(double *x, int length)
{
    double sum = 0, mean = 0, var;
    for(int i=0;i<length;i++)
    {
        sum += x[i];
    }
    mean = sum/length;
    sum = 0;
    for(int i=0;i<length;i++)
    {
        sum += (x[i]-mean)*(x[i]-mean);
    }
    var = sum/(length - 1);
    return var;
}

__global__ void test(double *x, int length) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.x + threadIdx.y*blockDim.x;
    printf("%d\n", idx);
    log_param result = f_EM(x, length, idx);
}



int main()
{
    double *a, *dev_a;
    int N = 500;
    a = new double[N];
    for(int i=0;i<N;i++)
    {
       a[i] = i;
    }
    cudaMalloc((void**)&dev_a, N*sizeof(double));
    cudaMemcpy(dev_a, a, N*sizeof(double), cudaMemcpyHostToDevice);
    // printf("ok\n");
    dim3 griddim(2,2);
    dim3 blockdim(2,2);
    test<<<griddim, blockdim>>>(dev_a, N);
    cudaDeviceSynchronize();
}