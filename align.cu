#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define MAX_BLOCK_SIZE 1024

#define DEBUG

// Cuda error checker
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

__global__ void align_kernel(int N1, int N2, int* seq1, int* seq2, int* matrix){
    for(int i = 0; i < N1 + N2 - 1; i++){
        // Compute if self is worker for this iteration

        // Update matrix

        // sync threads 
    }
}

void alignCuda(int N1, int N2, int* seq1, int*seq2, int* matrix){

    // Data structures 
    int *dev_seq1;
    int *dev_seq2;
    int *dev_matrix;

    // Allocate device memory
    cudaCheckError(cudaMalloc(&dev_seq1, N1*sizeof(int)));
    cudaCheckError(cudaMalloc(&dev_seq2, N2*sizeof(int)));
    cudaCheckError(cudaMalloc(&dev_matrix, (N1+1)*(N2+1)*sizeof(int)));

    // Copy data host to device
    cudaCheckError(cudaMemcpy(dev_seq1, seq1, N1, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(dev_seq2, seq2, N2, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(dev_matrix, matrix, (N1+1)*(N2+1), cudaMemcpyHostToDevice));

    // Smaller of N1 and N2 decides how many elements in the matrix can be processed together
    int max_concurrency = N1 < N2 ? N1 : N2;
    // Smaller of MAX_BLOCK_SIZE(decided by architecture) and max_concurrency decides block size
    int block_size = max_concurrency < MAX_BLOCK_SIZE ? max_concurrency : MAX_BLOCK_SIZE;
    dim3 blockDim(block_size);
    int grid_size = (max_concurrency + block_size - 1) / block_size;
    dim3 gridDim(grid_size);
    printf("------------BLOCKSIZE:%d GRIDSIZE:%d------------\n",block_size, grid_size);
    
    // Kernel call


    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )  printf("CUDA Error: %s\n", cudaGetErrorString(err));       

    // Free device memory 
    cudaCheckError(cudaFree(dev_seq1));
    cudaCheckError(cudaFree(dev_seq2));
    cudaCheckError(cudaFree(dev_matrix));
}



void printCudaInfo() {
    // For fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}