#include<mat_read.h>
using namespace std;

int main()
{   
    complex<double> ***c_S;
    size_t m,n,d;
    const char *filename = "./data.mat";
    c_S = ReadComplexMxArray3D(filename,"S");
    dim3D arraydim = matGetDim3D(filename,"S");
    m = arraydim.m; n = arraydim.n; d = arraydim.d;
    for(int k=0;k<d;k++)
    {
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
                cout<<c_S[i][j][k]<<" ";
            cout<<endl;
        }
    }
    FreeComplexArray3D(c_S, arraydim);
    return 0;
}