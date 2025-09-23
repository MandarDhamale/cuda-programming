#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void vectorAdd(float *A, float *B, float *C, int N){

    //intialize global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N){
        C[idx] = A[idx] + B[idx];
    }

}

int main(){


    const int N = 1024;
    size_t size = N * sizeof(float);

    //cpu variables

    float *h_A = (float*) malloc(size);
    float *h_B = (float*) malloc(size);
    float *h_C = (float*) malloc(size);

    //intializing the array 

    for(int i=0; i < N; i++){
        h_A[i] = (float) i;
        h_B[i] = (float) i;
    }

    //gpu variables

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // copy the arrays from host to device

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    //defining thread & grid size 

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // wait until the gpu is done 

    cudaDeviceSynchronize();
    cudaGetLastError();

    // copy the result back from gpu to cpu

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    //priting the reults to verify if it is correct

    for(int i=0; i<N; i++){
        printf("%f + %f = %f", h_A[i], h_B[i], h_C[i]);
        printf("\n");
    }

    //free up memory to avoid leaks 

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);


}