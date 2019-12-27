#include<mat_opt.h>

Mat ArrayToImage(double **arraydata, dim3D arraydim) 
{
    int m, n;
    m = (int)arraydim.m; n = (int)arraydim.n;
    double im_max = -DBL_MAX;
    double im_min =  DBL_MAX;
    Mat image = Mat::zeros(m, n, CV_8UC1); 
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            if(arraydata[i][j] > im_max)
                im_max = arraydata[i][j];
            if(arraydata[i][j] < im_min)
                im_min = arraydata[i][j];
        }
    }
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            image.at<uchar>(i,j) = (unsigned char) round(255*(arraydata[i][j]-im_min)/(im_max-im_min));
        }
    }
    return image;
}

Mat ArrayToMat(double **arraydata, dim3D arraydim)
{
    int m, n;
    m = (int)arraydim.m; n = (int)arraydim.n;
    Mat image = Mat::zeros(m, n, CV_64FC1); 
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            image.at<double>(i,j) = arraydata[i][j];       
        }
    }
    return image;
}

Mat PadArray(Mat image, int padrows, int padcolumns)
{   
    Mat temp, padimage;
    int m = image.rows; 
    int n = image.cols; 
    temp = Mat::zeros(m, n+2*padcolumns, image.type());
    padimage = Mat::zeros(m+2*padrows, n+2*padcolumns, image.type());
    for(int i=0;i<padcolumns;i++)
    {
        image.col(i).copyTo(temp.col(padcolumns-1-i));
        image.col(n-1-i).copyTo(temp.col(n+padcolumns+i));
    }
    image.copyTo(temp.colRange(padcolumns,padcolumns+n)); // colrange 左闭右开
    for(int i=0;i<padrows;i++)
    {
        temp.row(i).copyTo(padimage.row(padrows-1-i));
        temp.row(m-1-i).copyTo(padimage.row(m+padrows+i));
    }
    temp.copyTo(padimage.rowRange(padrows,padrows+m));
    temp.release();
    return padimage;
}