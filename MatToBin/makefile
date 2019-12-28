target = matTobin
objs = convert.o matTobin.o
CC = g++
CPPFLAGS = -I. -I/usr/local/MATLAB/R2019b/extern/include/  -std=c++11
LDFLAGS = -L/usr/local/lib  -L/usr/local/MATLAB/R2019b/bin/glnxa64/ \
 -lmat -lmx -leng -lmex  -Wl,-rpath /usr/local/MATLAB/R2019b/bin/glnxa64 \


$(target):$(objs)
	$(CC) $(objs) $(LDFLAGS) -o $(target)

%.o:%.cpp
	$(CC) -c $(CPPFLAGS) $^
.PHONY:clean  
clean:
	rm -f $(target) $(objs) 
