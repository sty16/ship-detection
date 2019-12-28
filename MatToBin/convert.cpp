#include<matTobin.h>
#include<fstream>
#include<unistd.h>
#include<stdlib.h>
#include<getopt.h>
using namespace std;

int main(int argc, char *argv[])
{
    int ch, channels,opt_index;
    const char *filename, *variable;
    const char *optstring = "d:v:c:";
    ofstream outfile("../data/data.bin", ios::out | ios::binary);
    static struct option long_options[] = {
        {"data", required_argument, NULL,'d'},
        {"varible", required_argument, NULL,'v'},
        {"channels", required_argument, NULL,'c'}
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
                channels = atoi(optarg); break;
            case '?':
                cout<<"Unknown option: "<<(char)optopt<<endl;
                break;
        }
    }
    if(channels == 1)
    {
        double **data = ReadDoubleMxArray(filename, variable);
        dim3D  arraydim = matGetDim3D(filename, variable);
        size_t m,n;
        m = arraydim.m;n = arraydim.n;
        outfile.write((char *)&channels,sizeof(int));
        outfile.write((char *)&m,sizeof(size_t));
        outfile.write((char *)&n,sizeof(size_t));
        for(size_t i=0;i<m;i++)
        {
            for(size_t j = 0;j<n;j++)
            {
                outfile.write((char *)&data[i][j], sizeof(double));
            }
        }
        outfile.close();
    }
}