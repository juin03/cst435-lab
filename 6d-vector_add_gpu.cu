/*
Purpose:
This CUDA C program performs vector addition on the GPU.
Key Features:
- Allocates memory for large vectors on both host and device.
- Transfers data between host and device.
- Launches a kernel to perform element-wise addition in parallel.
- Measures execution time for the operation.
- Verifies the correctness of the result.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1000000 // 1 million elements, matching CPU version

// Error checking macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at %s:%d: %s (%s)\n",
                file, line, cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

__global__ void vector_add_kernel(float *a, float *b, float *c, int n) {
  // Calculate the index of the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check: prevent reading outside array limits
  if (i < n){
    c[i] = a[i] + b[i];
  }
}

int main(){
  printf("--------GPU VECTOR ADD START----------\n");

  size_t bytes = N * sizeof(float);

  // Allocate memory for host arrays
  float *h_a = (float*)malloc(bytes);
  float *h_b = (float*)malloc(bytes);
  float *h_c = (float*)malloc(bytes);

  // Initialize host arrays to match CPU version
  for (int i = 0; i < N; i++) {
      h_a[i] = 1.0f;
      h_b[i] = 2.0f;
  }

  // Allocate memory for device arrays
  float *d_a, *d_b, *d_c;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, bytes));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, bytes));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, bytes));

  // Copy data from host to device
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  // Set up kernel launch parameters
  int threadsPerBlock = 256; // Typical value
  // Calculate blocksPerGrid carefully to ensure all elements are covered
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  printf("GPU: Starting vector addition of %d elements...\n", N);

  // Start timing
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

  // Launch kernel
  vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

  // Check for kernel launch errors
  CHECK_CUDA_ERROR(cudaGetLastError());

  CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

  // Copy results from device to host
  CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

  // Verify Results, matching CPU version check
  printf("GPU: Result h_c[0] = %f (Expected 3.0)\n", h_c[0]);
  // Optional: check last element too
  printf("GPU: Result h_c[%d] = %f (Expected 3.0)\n", N-1, h_c[N-1]);

  printf("\nGPU time taken: %f seconds\n", milliseconds / 1000.0f);

  // Free allocated device memory
  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));

  // Free allocated host memory
  free(h_a);
  free(h_b);
  free(h_c);

  printf("--------GPU VECTOR ADD END----------\n");
  return 0;
}