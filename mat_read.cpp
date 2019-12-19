#include<mat_read.h>

complex<double>*** ReadComplexMxArray3D(const char *filename,const char *variable)
{
    //  read .mat convert to c++ matrix
    MATFile *pmatFile = NULL;
    mxArray *pMxArray = NULL;
    pmatFile = matOpen(filename , "r");
    pMxArray = matGetVariable(pmatFile, variable);
    const mwSize* array_dim = mxGetDimensions(pMxArray); // return Pointer of first element in the dimensions array
    mwSize num_dim = mxGetNumberOfDimensions(pMxArray); //  return number of dimensions in the specified mxArray(MATLAB Array)
    size_t m = array_dim[0];
    size_t n = array_dim[1];
    size_t d = array_dim[2];
    complex<double>* ptr;
    ptr = (complex<double> *)mxGetComplexDoubles(pMxArray);
    complex<double> *** c_S;
    c_S = new complex<double>**[m];
    for(int i=0;i<m;i++)
    {   
        c_S[i] = new complex<double>*[n]; 
        for(int j=0;j<n;j++)
        {
            c_S[i][j] = new complex<double>[d];
        }
    }
    for(size_t k=0;k<d;k++)
    {
        for(size_t j=0;j<n;j++)
        {
            for(size_t i=0;i<m;i++)
            {
                c_S[i][j][k] = ptr[k*m*n + j*m + i ];
            }
        }
    }
    return c_S;
}

dim3D matGetDim3D(const char *filename,const char *variable)
{
    // get .mat data dimensions
    MATFile *pmatFile = NULL;
    mxArray *pMxArray = NULL;
    pmatFile = matOpen(filename , "r");
    pMxArray = matGetVariable(pmatFile, variable);
    dim3D arraydim;
    const size_t* array_dim = (size_t *)mxGetDimensions(pMxArray); 
    arraydim.m = array_dim[0];arraydim.n = array_dim[1];
    arraydim.d = array_dim[2];
    return arraydim;
}

void FreeComplexArray3D(complex<double> ***arraydata, dim3D arraydim)
{
    // free c++ complexmatrix memory
    size_t m,n,d;
    m = arraydim.m; n = arraydim.n; d = arraydim.d;
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            delete[] arraydata[i][j];
        }
        delete[] arraydata[i];
    }
    delete[] arraydata;
}
