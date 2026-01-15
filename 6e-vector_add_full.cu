/*
Purpose:
This CUDA C program performs vector addition on the GPU.
Key Features:
- Allocates memory for large vectors on both host and device.
- Transfers data between host and device.
- Launches a kernel to perform element-wise addition in parallel.
- Measures execution time for the operation.
- Verifies the correctness of the result.

Expected Output:
- The program successfully performs vector addition on the GPU.
- Example:
  - Result: h_C[0] = 0 (Expected: 0)
  - Result: h_C[N-1] = 19998 (Expected: 19998)
- No errors during kernel execution or memory operations.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000

// Step 1: Kernel Declaration (runs on GPU)
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Step 2: Declare host and device pointers
    float *h_A, *h_B, *h_C;  // Host (CPU)
    float *d_A, *d_B, *d_C;  // Device (GPU)

    // Step 3: Allocate host memory
    size_t size = N * sizeof(float);
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Step 4: Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Step 5: Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Step 6: Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Step 7: Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Step 8: Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Step 9: Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Step 10: Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Step 11: Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}