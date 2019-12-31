target = LMM_CFAR.out
objs =  LMM_CFAR.o  mat_opt.o
CC = g++
NVCC = nvcc
CPPFLAGS = -I. -I/usr/local/include/opencv4/opencv -I/usr/local/include/opencv4 -std=c++11
CUINCLUDES = -I. -I/usr/local/cuda/samples/common/inc \
-I/usr/local/include/opencv4/opencv -I/usr/local/include/opencv4 
CUGENCODE_FLAGS =  -arch=sm_35  -rdc=true  -std=c++11
LDFLAGS = -L/usr/lib/x86_64-linux-gnu/  -L/usr/local/lib \
-L/usr/local/cuda/lib64/  -lcudart   -lcudadevrt                                             \
-lopencv_dnn -lopencv_gapi -lopencv_highgui -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching \
-lopencv_video -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_videoio -lopencv_imgcodecs \
-lopencv_imgproc -lopencv_core -ltiff 


$(target):$(objs)
	nvcc -arch=sm_35 -dlink -o LMM_CFAR_link.o LMM_CFAR.o -lcudadevrt -lcudart
	$(CC) LMM_CFAR_link.o $(objs) $(LDFLAGS) -o $(target) 
	rm -f $(objs)

%.o:%.cu
	$(NVCC) -c $(CUINCLUDES) $(CUGENCODE_FLAGS) $^
%.o:%.cpp
	$(CC) -c $(CPPFLAGS) $^


.PHONY:clean  
clean:
	rm -f $(target) $(objs) 
