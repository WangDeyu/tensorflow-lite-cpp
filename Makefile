CXX=g++

CXXFLAGS=-Wl,--no-as-needed

INCLUDEDIRS := \
-Itensorflow-source-code \
-Itensorflow-source-code/tensorflow/contrib/lite/tools/make/downloads/flatbuffers/include/

LIBS := \
-ldl \
-lpthread

x86LIBPATHS := \
tensorflow-source-code/tensorflow/contrib/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a

armLIBPATHS := \
tensorflow-source-code/tensorflow/contrib/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a

all: x86 arm

x86:
	${CXX} ${CXXFLAGS} main.cpp -o main.x86.out ${INCLUDEDIRS} ${LIBS} ${x86LIBPATHS}

arm:
	arm-linux-gnueabihf-${CXX} ${CXXFLAGS} main.cpp -o main.arm.out ${INCLUDEDIRS} ${LIBS} ${armLIBPATHS}

# backup:
# 	g++ main.cpp linux_x86_64_with_fPIC/lib/libtensorflow-lite.a  -I ../../tensorflow/tensorflow/tensorflow/contrib/lite/tools/make/downloads/flatbuffers/include/ -I ../../tensorflow/tensorflow -Wl,--no-as-needed -ldl -lpthread
