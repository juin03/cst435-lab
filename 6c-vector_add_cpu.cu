%%writefile vector_add_cpu.c
/*
Purpose:
This C program performs vector addition on the CPU.
Key Features:
- Allocates memory for large vectors.
- Initializes vectors and performs element-wise addition.
- Measures execution time for the operation.
- Verifies the correctness of the result.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000000 // 1 million elements

// Function to perform vector addition on the CPU
void vector_add_cpu(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    printf("--------CPU Vector Add START----------\n");

    size_t bytes = N * sizeof(float);

    // Allocate memory for host arrays
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    printf("CPU: Starting vector addition of %d elements...\n", N);

    // Start timing
    clock_t start_time = clock();

    // Perform vector addition on CPU
    vector_add_cpu(h_A, h_B, h_C, N);

    // End timing
    clock_t end_time = clock();

    // Verify Results
    printf("CPU: Result h_C[0] = %f (Expected 3.0)\n", h_C[0]);
    printf("CPU: Time taken %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

    // Free allocated memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("--------CPU Vector Add END----------\n");
    return 0;
}