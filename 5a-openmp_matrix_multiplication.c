/*
-----------------------------------------------------------------------
 PURPOSE OF THIS PROGRAM
-----------------------------------------------------------------------
This program demonstrates PARALLEL MATRIX MULTIPLICATION using OpenMP.

KEY IDEAS DEMONSTRATED:
1) Large dynamic memory allocation for matrices.
2) Use of OpenMP to parallelize computation on multiple CPU cores.
3) Measuring execution time using omp_get_wtime().
4) Understanding how loop-level parallelism improves performance.

WHAT TO OBSERVE:
- Matrices A, B, and C are stored as 1D arrays (row-major order).
- Only the OUTER loop is parallelized for efficiency.
- Each thread computes different rows of matrix C.
- No synchronization is needed because each thread writes to
  a unique part of matrix C.

-----------------------------------------------------------------------
EXPECTED OUTPUT:
=== OpenMP Matrix Multiplication (N=2000) ===
Requested RAM: 0.0448 GB
Threads available: <number of threads>
Initialization Done. Computing...
Success! Time: <execution time> seconds
Result Check: C[0][0] = 2000

-----------------------------------------------------------------------
DIFFERENCE FROM MPI VERSION:
1) OpenMP uses threads and shared memory, while MPI uses processes and distributed memory.
2) OpenMP is limited to a single machine, while MPI can work across multiple machines.
3) OpenMP requires no explicit communication, while MPI requires explicit communication (e.g., MPI_Bcast).
4) OpenMP is simpler to implement for shared-memory systems, while MPI is better for distributed systems.
-----------------------------------------------------------------------
*/

#include <stdio.h>     // printf
#include <stdlib.h>    // malloc, free
#include <omp.h>       // OpenMP functions
#include <time.h>      // (not required here, but often used)

// ------------------------------------------------------------
// Matrix dimension (N x N)
// FAST TEST SIZE - keep small for quick testing
// ------------------------------------------------------------
#define N 2000

int main() {

    // ------------------------------------------------------------
    // 1. Calculate theoretical memory requirement
    // We allocate 3 matrices: A, B, and C
    // Each element is an int (4 bytes)
    // ------------------------------------------------------------
    double mem_req = 3.0 * N * N * sizeof(int)
                     / (1024.0 * 1024.0 * 1024.0);

    printf("=== OpenMP Matrix Multiplication (N=%d) ===\n", N);
    printf("Requested RAM: %.4f GB\n", mem_req);
    printf("Threads available: %d\n", omp_get_max_threads());

    // ------------------------------------------------------------
    // 2. Allocate memory dynamically on the heap
    // Using 1D arrays to represent 2D matrices
    // ------------------------------------------------------------
    int *A = (int*)malloc((size_t)N * N * sizeof(int));
    int *B = (int*)malloc((size_t)N * N * sizeof(int));
    int *C = (int*)malloc((size_t)N * N * sizeof(int));

    // Check for allocation failure
    if (!A || !B || !C) {
        printf("Memory Allocation Failed!\n");
        return 1;
    }

    // ------------------------------------------------------------
    // 3. Initialize matrices A and B
    // Parallelized for faster initialization
    // ------------------------------------------------------------
    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)N * N; i++) {
        A[i] = 1;
        B[i] = 1;
    }

    printf("Initialization Done. Computing...\n");

    // ------------------------------------------------------------
    // 4. Matrix Multiplication (PARALLEL)
    // C = A x B
    //
    // Parallelization strategy:
    // - Parallelize the outer loop (rows of C)
    // - Each thread computes a different row
    // - No race conditions (each thread writes to unique memory)
    // ------------------------------------------------------------
    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {          // rows of C
        for (int j = 0; j < N; j++) {      // columns of C
            long long sum = 0;             // local variable (thread-private)
            for (int k = 0; k < N; k++) {  // dot product
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = (int)sum;
        }
    }

    double end = omp_get_wtime();

    // ------------------------------------------------------------
    // 5. Output results
    // ------------------------------------------------------------
    printf("Success! Time: %.4f seconds\n", end - start);
    printf("Result Check: C[0][0] = %d\n", C[0]);

    // ------------------------------------------------------------
    // 6. Cleanup (free heap memory)
    // ------------------------------------------------------------
    free(A);
    free(B);
    free(C);

    return 0;
}

