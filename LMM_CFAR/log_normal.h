#ifndef __LOG_NORMAL__
#define __LOG_NORMAL__
#include<cuda_runtime.h>
#include<curand_kernel.h>
#include<stdio.h>

struct log_param{
    double mu[3];    // suppose only three component
    double var[3];
    double p[3];
};

__device__ log_param f_EM(double *x, int length);
__device__ double var(double *x, int length);
#endif