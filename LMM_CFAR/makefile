target = LMM_CFAP.out
objs = LMM_CFAR.o mat_read.o
CC = g++
CPPFLAGS = -I. -I/usr/local/MATLAB/R2019b/extern/include/  \
-I/usr/local/include/opencv4/opencv -I/usr/local/include/opencv4

LDFLAGS = -L/usr/lib/x86_64-linux-gnu/  -L/usr/local/lib  -L/usr/local/MATLAB/R2019b/bin/glnxa64/ \
-lopencv_dnn -lopencv_gapi -lopencv_highgui -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching \
-lopencv_video -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_videoio -lopencv_imgcodecs \
-lopencv_imgproc -lopencv_core -ltiff  -lmat -lmx -leng -lmex  -Wl,-rpath /usr/local/MATLAB/R2019b/bin/glnxa64

$(target):$(objs)
	$(CC) $(objs) $(LDFLAGS) -o $(target)

%.o:%.cpp
	$(CC) -c $(CPPFLAGS) $^

.PHONY:clean  
clean:
	rm -f $(target) $(objs) 
