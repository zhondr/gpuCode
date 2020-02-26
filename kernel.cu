#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.h"
#include <stdlib.h>

using namespace std;

__global__ void matrixMultiplicationKernel(int* A, int* B, int* C, int N) {

    int ROW = blockIdx.y*blockDim.y + threadIdx.y; //local row
    int Z = blockIdx.x*blockDim.x*blockDim.y;  //sum of threads in previous blocks
    int COL = threadIdx.x;  //local column

    int tmpSum = 0;

    if (Z + ROW * N + COL < N*N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[Z + ROW * N + i] * B[i * N + COL];
        }
	C[Z + ROW * N + COL] = tmpSum;
    }

}


void matrixMultiplication(int *A, int *B, int *C, int N){

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 1024 threads per block
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
      if (N*N > 1024){
          threadsPerBlock.y = floor(1024/N);
          blocksPerGrid.x = ceil(N*N/threadsPerBlock.y);
      }

    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);
}
