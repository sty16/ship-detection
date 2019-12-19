#include<mat.h>
#include<iostream>
#include<complex>
#pragma once
using namespace std;


struct dim3D
{
    size_t m;
    size_t n;
    size_t d;
};

complex<double>*** ReadComplexMxArray3D(const char *filename,const char *variable);
dim3D matGetDim3D(const char *filename,const char *variable);
void FreeComplexArray3D(complex<double> ***arraydata, dim3D arraydim);