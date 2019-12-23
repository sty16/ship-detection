Readme

### 1.Compile link method

```
g++ -c -I/usr/local/MATLAB/R2019b/extern/include/ -I. mat_read.cpp main.cpp
```

```
g++ main.o mat_read.o -L/usr/local/MATLAB/R2019b/bin/glnxa64/ -lmat -lmx -leng -lmex -Wl,-rpath /usr/local/MATLAB/R2019b/bin/glnxa64 -o mat
```

>注意：
>
>将-I后的路径更换为MATLAB相应的**/extern/include/路径, 将-L后的路径更换为MATLAB **/bin/glnxa64路径

### 2. LMM_CFAR folder

This folder contain the CUDA implemetation for ship detection algotithms in synthetic aperture radars images as published inthe following papers.

>Yi Cui, Jian Yang, Yoshio Yamaguchi, Gulab Singh, Sang-Eun Park, and Hirokazu Kobayashi. "On semiparametric clutter estimation for ship detection in synthetic aperture radar images." *IEEE transactions on geoscience and remote sensing* 51, no. 5 (2013): 3170-3180.

#### 2.1 Prerequisites

```
MATLAB r2019b ------>   installed in the /usr/local/MATLAB
opencv4       ------>   installed in the /usr/local/include
CUDA          ------>   installed in the /usr/local/cuda
```

#### 2.2 make

```
cd ./LMM_CFAR
make
./LMM_CFAR.out
```




