#include <stdio.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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

// Implementation requires N1 >= N2;

__global__ void align_kernel(int N1, int N2, int* seq1, int* seq2, int* matrix, int pass, int x_concurrency){
    //auto g = cg::this_grid();
    if(threadIdx.x >= x_concurrency) return;
    int num_iter = N1 + x_concurrency - 1;
    int worker_index = pass * blockDim.x + threadIdx.x;
    int x_index = worker_index + 1;
    int y_index = 1;
    int iters_to_work = N1; 
    int worker_start_delay = threadIdx.x;
    for(int i = 0; i < num_iter; i++){
        // Only work for 'iters_to_work' iterations, with delay 'worker_start_delay'
        // This gives correct update behavior
        if(worker_start_delay <= i && i < worker_start_delay + iters_to_work){
            int match = seq1[y_index-1] == seq2[x_index-1] ? 1 : -1;
            int score_top = matrix[(y_index - 1) * (N2 + 1) + x_index] - 1;
            int score_left = matrix[y_index * (N2 + 1) + x_index - 1] - 1;
            int score_topleft = matrix[(y_index - 1) * (N2 + 1) + x_index - 1] + match;
            int max = score_top > score_left ? score_top : score_left;
            max = max > score_topleft ? max : score_topleft;
            /*printf("iter: %d, worker: %d, x:%d, y:%d, match:%d, max:%d\n", 
            i, worker_index, x_index, y_index, match, max);*/
            matrix[y_index * (N2+1) + x_index] = max;
            y_index++;
        }
        // sync threads
        // cooperative_groups::this_grid().sync();
        __syncthreads();
    }
    //g.sync();
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
    cudaCheckError(cudaMemcpy(dev_seq1, seq1, N1*sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(dev_seq2, seq2, N2*sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(dev_matrix, matrix, (N1+1)*(N2+1)*sizeof(int), cudaMemcpyHostToDevice));

    // Smaller of N1 and N2 decides how many elements in the matrix can be processed together
    int max_concurrency = N1 < N2 ? N1 : N2;
    // Smaller of MAX_BLOCK_SIZE(decided by architecture) and max_concurrency decides block size
    int block_size = max_concurrency < MAX_BLOCK_SIZE ? max_concurrency : MAX_BLOCK_SIZE;
    dim3 blockDim(block_size);
    dim3 gridDim(1);
    int num_pass = (max_concurrency + block_size - 1) / block_size;
    printf("-----------------BLOCKSIZE:%d GRIDSIZE:%d-----------------\n",block_size, 1);
    
    // Kernal timer
    double kernalStartTime = CycleTimer::currentSeconds();

    // Kernel call
    for(int i = 0; i < num_pass; i++){
        int x_concurrency;
        if(i == num_pass-1){
            x_concurrency = max_concurrency % block_size;
            if(x_concurrency == 0){
                x_concurrency = block_size;
            }
        }else{
            x_concurrency = block_size;
        }
        printf("N1: %d, N2: %d, pass: %d, x_concurrency:%d\n", N1, N2, i, x_concurrency);
        align_kernel<<<gridDim, blockDim>>>(N1, N2, dev_seq1, dev_seq2, dev_matrix, i, x_concurrency);
        cudaDeviceSynchronize();
    }
    

    // End of timer and printout
    double kernalEndTime = CycleTimer::currentSeconds();
    double kernalOverallDuration = kernalEndTime - kernalStartTime;
    printf("Kernel: %.3f ms\n", 1000.f * kernalOverallDuration);  

    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )  printf("CUDA Error: %s\n", cudaGetErrorString(err));

    cudaCheckError(cudaMemcpy(matrix, dev_matrix, (N1+1)*(N2+1)*sizeof(int), cudaMemcpyDeviceToHost));

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