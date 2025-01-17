
thermal: thermal.cpp
	g++ -I/usr/include/opencv4 -O3 -march=native -o $@ $? -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lbcm2835 -lm


