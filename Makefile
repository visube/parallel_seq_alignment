
CUDA_EXECUTABLE := align

OPENMP_EXECUTABLE := alignopenmp

CU_FILES   := cuda/align.cu

CU_DEPS    :=

CUDA_CC_FILES   := cuda/main.cpp


###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=cuda/objs
OPENMP_OBJSDIR=openmp/objs
CXX=g++ -std=c++11 -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/depot/cuda-10.2/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc --std=c++11
#NVCCFLAGS= -O3 -m64 --std=c++11 -rdc=true --gpu-architecture compute_61  -ccbin /usr/bin/gcc
OPENMP_CXXFLAGS = -I. -O3 -Wall -fopenmp -Wno-unknown-pragmas

OBJS=$(OBJDIR)/main.o  $(OBJDIR)/align.o
OPENMP_OBJS=openmp/objs/alignopenmp.o

.PHONY: dirs clean

all: $(CUDA_EXECUTABLE) $(OPENMP_EXECUTABLE)

cuda: $(CUDA_EXECUTABLE)

openmp: $(OPENMP_EXECUTABLE)

$(OPENMP_EXECUTABLE): openmp_dirs $(OPENMP_OBJS)
	$(CXX) $(OPENMP_CXXFLAGS) -o $@ $(OPENMP_OBJS)

$(OPENMP_OBJS): openmp/main.cpp
	$(CXX) $< $(OPENMP_CXXFLAGS) -c -o $@

openmp_dirs:
		mkdir -p $(OPENMP_OBJSDIR)/
dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) $(OPENMP_OBJSDIR) *.ppm *~ $(CUDA_EXECUTABLE) $(OPENMP_EXECUTABLE)

$(CUDA_EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: cuda/%.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: cuda/%.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
