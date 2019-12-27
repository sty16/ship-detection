#include<mat_opt.h>
#include<cuda_runtime.h>
#include"device_launch_parameters.h"


__global__ void hello(int size) 
{
    int  a =  0;
    double *b = new double[size];
    for(int i = 0;i<size;i++)
    {
        b[i] = i;
        printf("%.1f\n", b[i]);
    }
    printf("hello cuda\n");
}




int main(int argc, char *argv[])
{
    double **im;
    int ch, opt_index;    // opt_index为选项在long_options中的索引
    const char *optstring = "d:v:";
    int r_c = 15, r_g = 10;
    const char *filename = "./radarsat2-tj.mat";   
    const char *variable = "I";
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
            case 'v':
                variable = optarg; break;
            case 'c':
                r_c = atoi(optarg); break;
            case 'g':
                r_g = atoi(optarg); break;
            case '?':
                cout<<"Unknown option: "<<(char)optopt<<endl;
                break;
        }
    }
    im = ReadDoubleMxArray(filename, variable);
    size_t m,n;
    dim3D arraydim = matGetDim3D(filename, variable);
    Mat image = ArrayToImage(im, arraydim);
    Mat origin_image = ArrayToMat(im, arraydim);
    double data[3][3] = { {1,2,3},{4,5,6},{7,8,9} };
    // Mat C = Mat(3,3,CV_64FC1,data);
    // Mat B = PadArray(C, 2, 2);
    // cout<<B<<endl;
    hello<<<2,2>>>(6);
    Mat pad_image = PadArray(origin_image,r_c,r_c);
    imshow("gray" , image);
    while(char(waitKey())!='q') {}
    FreeDoubleArray(im,arraydim);
    image.release();
	return 0 ;
}

