
CUDA_EXECUTABLE := align

CU_FILES   := cuda/align.cu

CU_DEPS    :=

CUDA_CC_FILES   := cuda/main.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=cuda/objs
CXX=g++ -std=c++11 -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/depot/cuda-10.2/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

OBJS=$(OBJDIR)/main.o  $(OBJDIR)/align.o


.PHONY: dirs clean

all: $(CUDA_EXECUTABLE)

cuda: $(CUDA_EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(CUDA_EXECUTABLE)

$(CUDA_EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: cuda/%.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: cuda/%.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
