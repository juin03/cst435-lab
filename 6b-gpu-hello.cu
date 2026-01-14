%%writefile gpu_hello.cu
/*
Purpose:
This CUDA C program demonstrates a simple "Hello World" example using GPU threads.
Key Features:
- Launches a kernel with multiple threads.
- Each thread prints its thread ID.
- Demonstrates basic CUDA programming concepts like kernel launch and error checking.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_kernel(){
  printf("Hello world from GPU Thread %d\n", threadIdx.x);
}

int main(){
  printf("--------GPU START----------\n");

  // launch kernel
  hello_kernel<<<1,5>>>();

  // CHECK ERRORS
  // This catches if the kernel failed to start
  cudaError_t err =cudaGetLastError();
  if (err != cudaSuccess){
    printf("GPU Launch Error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
  printf("--------GPU END----------\n");
  return 0;
}