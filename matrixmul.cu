#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "kernel.cu"
#include "dev_array.h"
#include <math.h>

using namespace std;

int main()
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int N = 10;
    int SIZE = N*N;

    // Allocate memory on the host
    vector<int> h_A(SIZE);
    vector<int> h_B(SIZE);
    vector<int> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = i+1;
            h_B[i*N+j] = i+1;
        }
    }

    // Allocate memory on the device
    dev_array<int> d_A(SIZE);
    dev_array<int> d_B(SIZE);
    dev_array<int> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);

    double time_spent=0;

    for (int i=0;i<10000;i++) {
    clock_t begin = clock();

    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    
    clock_t end = clock();
    time_spent = time_spent + ((double)(end - begin) / CLOCKS_PER_SEC);

    cudaDeviceSynchronize();
    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();

//    clock_t end = clock();
//    time_spent = time_spent + ((double)(end - begin) / CLOCKS_PER_SEC);
   }

    //cudaMemcpy( h_C, d_C, bytes, cudaMemcpyDeviceToHost );
    printf("N = %d\n",N);
    printf("Time is calculated on 10000 cycles\n");
    printf("\n");
    printf("Overall time on GPU: %f\n",time_spent);
    printf("Average time on GPU: %f\n",time_spent/10000);
    printf("-------------------------------\n");
    /*printf("Product of the matrices:\n");

    for (int c = 0; c < N ; c++) {
      for (int d = 0; d < N; d++)
        printf("%d\t", h_C[c*N+d]);

      printf("\n");
    }
*/

    int *cpu_C;
    cpu_C=new int[SIZE];
    time_spent=0;

    // Now do the matrix multiplication on the CPU
    int sum;
    for (int i=0;i<10000;i++) {
    sum=0;
    clock_t begin = clock();

    for (int row=0; row<N; row++){
        for (int col=0; col<N; col++){
            sum = 0.f;
            for (int n=0; n<N; n++){
                sum += h_A[row*N+n]*h_B[n*N+col];
            }
            cpu_C[row*N+col] = sum;
        }
    }
    clock_t end = clock();
    time_spent = time_spent + ((double)(end - begin) / CLOCKS_PER_SEC);
   }

    printf("Overall time on CPU: %f\n",time_spent);
    printf("Average time on CPU: %f\n",time_spent/10000);
    printf("\n");
    printf("Product of the matrices:\n");

    for (int c = 0; c < N ; c++) {
      for (int d = 0; d < N; d++)
        printf("%d\t", cpu_C[c*N+d]);

      printf("\n");
    }

    double err = 0;
    // Check the result and make sure it is correct
    for (int ROW=0; ROW < N; ROW++){
        for (int COL=0; COL < N; COL++){
            err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];
        }
    }

    cout << "Error: " << err << endl;

    return 0;
}
