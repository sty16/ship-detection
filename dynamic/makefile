target = dynamic.out
objs = dynamic.o
NVCC = nvcc
CC = g++
CUGENCODE_FLAGS =  -arch=sm_50 -rdc=true  -std=c++11
LDFLAGS  = -L/usr/local/cuda/lib64/  -lcudart   -lcudadevrt

$(target):$(objs)
	$(NVCC) -arch=sm_50 -dlink -o dynamic_link.o dynamic.o 
	$(CC) dynamic_link.o dynamic.o $(LDFLAGS) -o $(target) 

%.o:%.cu
	$(NVCC) -c  $(CUGENCODE_FLAGS) $^

.PHONY:clean  
clean:
	rm -f $(target) $(objs) dynamic_link.o

