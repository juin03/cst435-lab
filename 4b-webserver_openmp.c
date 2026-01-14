/*
-----------------------------------------------------------------------
 PURPOSE OF THIS PROGRAM
-----------------------------------------------------------------------
This program simulates a web server handling 5 client requests using
OpenMP (shared-memory parallel programming).

KEY IDEAS DEMONSTRATED:
1) OpenMP creates a team of threads automatically using:
      #pragma omp parallel num_threads(5)

2) All threads share the SAME global cache structure (shared memory).

3) The cache update is a CRITICAL SECTION, protected using:
      #pragma omp critical
   This ensures only ONE thread updates the shared cache at a time
   (prevents race conditions / lost updates).

WHAT TO OBSERVE:
- Each thread has a unique thread id: omp_get_thread_num()
- Reading/processing happens in parallel
- Cache update happens one-at-a-time inside the critical section
- Final cache values should be views=5 and hits=5

-----------------------------------------------------------------------
DIFFERENCE FROM PTHREADS VERSION:
1) OpenMP abstracts thread creation and management, reducing boilerplate code.
2) Synchronization is handled using `#pragma omp critical`, which is simpler than mutexes.
3) OpenMP is ideal for parallel loops and tasks, while Pthreads offer more fine-grained control.
4) Pthreads require explicit thread lifecycle management (create, join, etc.).
-----------------------------------------------------------------------

EXPECTED OUTPUT:
=============================================
 Web Server Simulation (OpenMP Critical)
=============================================

[Server] Initial cache: views=0, hits=0, last=none

[Server] Creating 5 threads for 5 clients...

[Thread 0] Client 1 requested: /home
[Thread 0] Client 1: Reading data...
[Thread 1] Client 2 requested: /about
[Thread 1] Client 2: Reading data...
[Thread 0] Client 1: Processing request...
[Thread 0] Client 1: Entered CRITICAL SECTION
[Thread 0] Client 1: Updated cache - views=1, hits=1, last=/home
[Thread 0] Client 1: Exiting CRITICAL SECTION
[Thread 0] Client 1: Response sent

[Thread 1] Client 2: Processing request...
[Thread 1] Client 2: Entered CRITICAL SECTION
[Thread 1] Client 2: Updated cache - views=2, hits=2, last=/about
[Thread 1] Client 2: Exiting CRITICAL SECTION
[Thread 1] Client 2: Response sent

... (similar output for threads 2, 3, and 4) ...

[Server] All threads completed

=============================================
 Final Results
=============================================
[Server] Final cache: views=5, hits=5, last=/home
[Server] Expected: views=5, hits=5
[Server] Status: SUCCESS - All updates recorded correctly!
=============================================
*/

#include <stdio.h>     // printf
#include <stdlib.h>    // general utilities
#include <unistd.h>    // usleep
#include <string.h>    // strcpy
#include <omp.h>       // OpenMP functions + directives

// -------------------------------------------------------------------
// Shared cache structure - accessible by all threads
// This simulates a web server cache that stores:
//  - page_views: total pages viewed
//  - cache_hits: number of cache hits (simplified: always hit here)
//  - last_page: last page that was accessed
// -------------------------------------------------------------------
typedef struct {
    int page_views;
    int cache_hits;
    char last_page[50];
} ServerCache;

// Global shared cache (shared memory among OpenMP threads)
ServerCache cache = {0, 0, "none"};

int main() {

    // Pages that 5 "clients" will request (index matches thread id)
    const char* pages[] = {"/home", "/about", "/products", "/contact", "/home"};

    printf("=============================================\n");
    printf(" Web Server Simulation (OpenMP Critical)\n");
    printf("=============================================\n\n");

    printf("[Server] Initial cache: views=%d, hits=%d, last=%s\n\n",
           cache.page_views, cache.cache_hits, cache.last_page);

    printf("[Server] Creating 5 threads for 5 clients...\n\n");

    // ----------------------------------------------------------------
    // OpenMP PARALLEL REGION
    // - Creates a team of 5 threads
    // - Each thread executes the code inside this block
    // - Threads are automatically joined at the end of the block
    // ----------------------------------------------------------------
    #pragma omp parallel num_threads(5)
    {
        // ------------------------------------------------------------
        // Thread identification
        // omp_get_thread_num() returns thread id: 0..(num_threads-1)
        // ------------------------------------------------------------
        int tid = omp_get_thread_num();
        int client_id = tid + 1;          // map thread 0..4 -> client 1..5
        const char* page = pages[tid];    // each thread picks its page

        printf("[Thread %d] Client %d requested: %s\n", tid, client_id, page);

        // ---------------------------
        // STEP 1: Read client data
        // ---------------------------
        printf("[Thread %d] Client %d: Reading data...\n", tid, client_id);
        usleep(50000); // 50 ms simulated read time

        // ---------------------------
        // STEP 2: Process request
        // ---------------------------
        printf("[Thread %d] Client %d: Processing request...\n", tid, client_id);
        usleep(100000); // 100 ms simulated processing time

        // ------------------------------------------------------------
        // STEP 3: Update shared cache (CRITICAL SECTION)
        // #pragma omp critical ensures mutual exclusion:
        // ONLY ONE thread can execute this block at a time.
        // ------------------------------------------------------------
        #pragma omp critical
        {
            printf("[Thread %d] Client %d: Entered CRITICAL SECTION\n", tid, client_id);

            // ---- CRITICAL SECTION START ----
            cache.page_views++;              // shared counter update
            cache.cache_hits++;              // shared counter update
            strcpy(cache.last_page, page);   // shared string update
            // ---- CRITICAL SECTION END ----

            printf("[Thread %d] Client %d: Updated cache - views=%d, hits=%d, last=%s\n",
                   tid, client_id, cache.page_views, cache.cache_hits, cache.last_page);

            printf("[Thread %d] Client %d: Exiting CRITICAL SECTION\n", tid, client_id);
        }
        // Critical lock is automatically released here

        // ---------------------------
        // STEP 4: Send response
        // ---------------------------
        printf("[Thread %d] Client %d: Response sent\n\n", tid, client_id);
    }

    printf("[Server] All threads completed\n\n");

    // Final results
    printf("=============================================\n");
    printf(" Final Results\n");
    printf("=============================================\n");
    printf("[Server] Final cache: views=%d, hits=%d, last=%s\n",
           cache.page_views, cache.cache_hits, cache.last_page);
    printf("[Server] Expected: views=5, hits=5\n");

    if (cache.page_views == 5 && cache.cache_hits == 5) {
        printf("[Server] Status: SUCCESS - All updates recorded correctly!\n");
    } else {
        printf("[Server] Status: ERROR - Some updates were lost!\n");
    }

    printf("=============================================\n");

    return 0;
}
