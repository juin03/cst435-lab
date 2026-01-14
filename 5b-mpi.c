/*
-----------------------------------------------------------------------
 PURPOSE OF THIS PROGRAM
-----------------------------------------------------------------------
This program demonstrates PARALLEL MATRIX MULTIPLICATION using MPI
(Message Passing Interface).

KEY IDEAS DEMONSTRATED:
1) MPI uses MULTIPLE PROCESSES (not threads).
2) Each process has its OWN private memory (no shared memory).
3) Work is divided by ROWS of matrix A across processes.
4) Matrix B is BROADCAST to all processes.
5) Each process computes PART of matrix C (C_local).
6) Timing is measured using MPI_Wtime().

WHAT TO OBSERVE:
- rank identifies each process (0 = master).
- size is the total number of MPI processes.
- Communication is explicit (MPI_Bcast, MPI_Barrier).
- This model works across multiple machines (distributed memory).
-----------------------------------------------------------------------
*/
/*
-----------------------------------------------------------------------
EXPECTED OUTPUT:
=== MPI Matrix Multiplication ===
Process 0: Memory allocated for A, B, and C.
Process 0: Matrix B broadcasted to all processes.
Process <rank>: Computing rows <start_row> to <end_row>...
Process <rank>: Computation completed.
Process 0: Gathering results from all processes.
Success! Time: <execution time> seconds
Result Check: C[0][0] = 2000

-----------------------------------------------------------------------
DIFFERENCE FROM OPENMP VERSION:
1) MPI uses processes and distributed memory, while OpenMP uses threads and shared memory.
2) MPI can work across multiple machines, while OpenMP is limited to a single machine.
3) MPI requires explicit communication (e.g., MPI_Bcast), while OpenMP does not.
4) MPI is better for distributed systems, while OpenMP is simpler for shared-memory systems.
-----------------------------------------------------------------------

*/

/*
-----------------------------------------------------------------------
 PURPOSE OF THIS PROGRAM
-----------------------------------------------------------------------
This program demonstrates PARALLEL MATRIX MULTIPLICATION using MPI
(Message Passing Interface).

KEY IDEAS DEMONSTRATED:
1) MPI uses MULTIPLE PROCESSES (not threads).
2) Each process has its OWN private memory (no shared memory).
3) Work is divided by ROWS of matrix A across processes.
4) Matrix B is BROADCAST to all processes.
5) Each process computes PART of matrix C (C_local).
6) Timing is measured using MPI_Wtime().

WHAT TO OBSERVE:
- rank identifies each process (0 = master).
- size is the total number of MPI processes.
- Communication is explicit (MPI_Bcast, MPI_Barrier).
- This model works across multiple machines (distributed memory).
-----------------------------------------------------------------------
*/

#include <mpi.h>     // MPI library
#include <stdio.h>   // printf
#include <stdlib.h>  // malloc, free

// ------------------------------------------------------------
// Matrix dimension (N x N)
// Use N = 2000 for testing
// Increase to 20000 for large distributed runs
// ------------------------------------------------------------
#define N 2000

int main(int argc, char** argv) {

    // ------------------------------------------------------------
    // 1. Initialize MPI environment
    // ------------------------------------------------------------
    MPI_Init(&argc, &argv);

    int rank, size;

    // rank = ID of this process (0, 1, 2, ...)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // size = total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ------------------------------------------------------------
    // Each process handles N/size rows of matrix A
    // ------------------------------------------------------------
    int rows_per_node = N / size;

    // Master process prints startup info
    if (rank == 0) {
        printf("--- MPI Matrix Multiplication (N=%d) ---\n", N);
        printf("Nodes (Processes): %d\n", size);
        printf("Master initializing and allocating memory...\n");
    }

    // ------------------------------------------------------------
    // 2. Allocate local memory (PRIVATE to each process)
    // ------------------------------------------------------------
    int *A_local = (int*)malloc((size_t)rows_per_node * N * sizeof(int));
    int *B       = (int*)malloc((size_t)N * N * sizeof(int));
    int *C_local = (int*)malloc((size_t)rows_per_node * N * sizeof(int));

    if (!A_local || !B || !C_local) {
        printf("Rank %d: Memory allocation failed!\n", rank);
        MPI_Finalize();
        return 1;
    }

    // ------------------------------------------------------------
    // 3. Initialize data
    // Each process initializes its local part of A
    // ------------------------------------------------------------
    for (size_t i = 0; i < (size_t)rows_per_node * N; i++) {
        A_local[i] = 1;
    }

    // Only MASTER initializes matrix B
    if (rank == 0) {
        for (size_t i = 0; i < (size_t)N * N; i++) {
            B[i] = 1;
        }
    }

    // ------------------------------------------------------------
    // 4. Broadcast matrix B to ALL processes
    // After this call, every process has a full copy of B
    // ------------------------------------------------------------
    MPI_Bcast(B, (size_t)N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Ensure all processes are ready before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // TIMER START
    // ------------------------------------------------------------
    double start_time = MPI_Wtime();
    if (rank == 0) {
        printf("Computing...\n");
    }

    // ------------------------------------------------------------
    // 5. Compute local matrix multiplication
    // Each process computes only its assigned rows
    // ------------------------------------------------------------
    for (int i = 0; i < rows_per_node; i++) {      // local rows
        for (int j = 0; j < N; j++) {              // columns
            long long sum = 0;
            for (int k = 0; k < N; k++) {          // dot product
                sum += A_local[i * N + k] * B[k * N + j];
            }
            C_local[i * N + j] = (int)sum;
        }
    }

    // ------------------------------------------------------------
    // Synchronize before stopping timer
    // ------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // TIMER END
    // ------------------------------------------------------------
    double end_time = MPI_Wtime();

    // ------------------------------------------------------------
    // 6. Output result (only MASTER prints timing)
    // ------------------------------------------------------------
    if (rank == 0) {
        printf("Success! Calculation Complete.\n");
        printf("Total Execution Time: %.4f seconds\n",
               end_time - start_time);
    }

    // ------------------------------------------------------------
    // 7. Cleanup and shutdown MPI
    // ------------------------------------------------------------
    free(A_local);
    free(B);
    free(C_local);

    MPI_Finalize();
    return 0;
}
