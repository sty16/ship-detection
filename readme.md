Readme

### 1.编译链接方法

```
g++ -c -I/usr/local/MATLAB/R2019b/extern/include/ -I. mat_read.cpp main.cpp
```

```
g++ main.o mat_read.o -L/usr/local/MATLAB/R2019b/bin/glnxa64/ -lmat -lmx -leng -lmex -Wl,-rpath /usr/local/MATLAB/R2019b/bin/glnxa64 -o mat
```

>注意：
>
>将-I后的路径更换为MATLAB相应的**/extern/include/路径, 将-L后的路径更换为MATLAB **/bin/glnxa64路径