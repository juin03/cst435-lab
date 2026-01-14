/*
-----------------------------------------------------------------------
 PURPOSE OF THIS PROGRAM
-----------------------------------------------------------------------
This program simulates a simple web server handling multiple clients
using Pthreads.

Key ideas demonstrated:
1) Multiple threads run at the same time (each thread = one client request).
2) All threads SHARE a global "cache" structure (shared memory).
3) Updating shared data must be protected using a MUTEX.
4) The CRITICAL SECTION is where the shared cache is updated.
5) pthread_mutex_lock() ensures only ONE thread updates the cache at a time,
   preventing race conditions / lost updates.

WHAT TO OBSERVE:
- Each thread prints when it ACQUIRES and RELEASES the cache lock.
- Final cache values should equal the number of clients (5).

-----------------------------------------------------------------------
DIFFERENCE FROM OPENMP VERSION:
1) Pthreads require explicit thread creation (`pthread_create`) and joining (`pthread_join`).
2) Mutex (`pthread_mutex_lock`) is used to protect the critical section.
3) More control over thread lifecycle but requires more boilerplate code.
4) OpenMP abstracts thread management and synchronization, making it simpler for parallel loops.
-----------------------------------------------------------------------

EXPECTED OUTPUT:
=============================================
 Web Server Simulation (Pthreads + Mutex)
=============================================

[Server] Mutex initialized
[Server] Initial cache: views=0, hits=0, last=none

[Server] Creating threads for 5 clients...

[Thread] Client 1 requested: /home
[Thread] Client 1: Reading data...
[Thread] Client 2 requested: /about
[Thread] Client 2: Reading data...
[Thread] Client 1: Processing request...
[Thread] Client 1: Cache lock ACQUIRED
[Thread] Client 1: Updated cache - views=1, hits=1, last=/home
[Thread] Client 1: Cache lock RELEASED
[Thread] Client 1: Response sent

[Thread] Client 2: Processing request...
[Thread] Client 2: Cache lock ACQUIRED
[Thread] Client 2: Updated cache - views=2, hits=2, last=/about
[Thread] Client 2: Cache lock RELEASED
[Thread] Client 2: Response sent

... (similar output for clients 3, 4, and 5) ...

[Server] All threads completed
[Server] Mutex destroyed

=============================================
 Final Results
=============================================
[Server] Final cache: views=5, hits=5, last=/home
[Server] Expected: views=5, hits=5
[Server] Status: SUCCESS - All updates recorded correctly!
=============================================
*/


#include <stdio.h>      // printf
#include <stdlib.h>     // exit (not used here but common)
#include <unistd.h>     // usleep
#include <pthread.h>    // pthread_create, pthread_join, mutex
#include <string.h>     // strcpy

// -------------------------------------------------------------------
// SHARED CACHE STRUCTURE (shared by ALL threads)
// This simulates a web server cache:
//  - page_views: total pages viewed
//  - cache_hits: number of cache hits (simplified as always hit here)
//  - last_page: last page requested (string)
// -------------------------------------------------------------------
typedef struct {
    int page_views;
    int cache_hits;
    char last_page[50];
} ServerCache;

// Global shared cache (shared memory for all threads in this process)
ServerCache cache = {0, 0, "none"};

// -------------------------------------------------------------------
// MUTEX DECLARATION
// This lock protects updates to the shared cache.
// -------------------------------------------------------------------
pthread_mutex_t cache_lock;

// -------------------------------------------------------------------
// REQUEST STRUCTURE (each thread gets one Request)
// client_id: which client is making the request
// page: which page they requested
// -------------------------------------------------------------------
typedef struct {
    int client_id;
    char page[50];
} Request;

// -------------------------------------------------------------------
// Thread function: handles ONE client request
// Each thread simulates a web server request pipeline:
//  1) Read client data (simulated delay)
//  2) Process request (simulated delay)
//  3) Update shared cache (CRITICAL SECTION protected by mutex)
//  4) Send response (simulated)
// -------------------------------------------------------------------
void* handle_request(void* arg) {

    // Convert void* back into a Request*
    Request* req = (Request*)arg;

    printf("[Thread] Client %d requested: %s\n", req->client_id, req->page);

    // ---------------------------
    // STEP 1: Read client data
    // ---------------------------
    printf("[Thread] Client %d: Reading data...\n", req->client_id);
    usleep(50000); // 50 ms delay to simulate reading

    // ---------------------------
    // STEP 2: Process request
    // ---------------------------
    printf("[Thread] Client %d: Processing request...\n", req->client_id);
    usleep(100000); // 100 ms delay to simulate CPU processing

    // ------------------------------------------------------------
    // STEP 3: Update shared cache (CRITICAL SECTION)
    // Only ONE thread can enter this section at a time.
    // ------------------------------------------------------------
    pthread_mutex_lock(&cache_lock);
    printf("[Thread] Client %d: Cache lock ACQUIRED\n", req->client_id);

    // ---- CRITICAL SECTION START ----
    cache.page_views++;                 // update shared counter
    cache.cache_hits++;                 // update shared counter
    strcpy(cache.last_page, req->page); // update shared string
    // ---- CRITICAL SECTION END ----

    printf("[Thread] Client %d: Updated cache - views=%d, hits=%d, last=%s\n",
           req->client_id, cache.page_views, cache.cache_hits, cache.last_page);

    // Release the lock so other threads can update the cache
    pthread_mutex_unlock(&cache_lock);
    printf("[Thread] Client %d: Cache lock RELEASED\n", req->client_id);

    // ---------------------------
    // STEP 4: Send response
    // ---------------------------
    printf("[Thread] Client %d: Response sent\n\n", req->client_id);

    return NULL;
}

int main() {

    printf("=============================================\n");
    printf(" Web Server Simulation (Pthreads + Mutex)\n");
    printf("=============================================\n\n");

    // ------------------------------------------------------------
    // MUTEX INITIALIZATION
    // Must be done before any thread uses the lock.
    // ------------------------------------------------------------
    pthread_mutex_init(&cache_lock, NULL);

    printf("[Server] Mutex initialized\n");
    printf("[Server] Initial cache: views=%d, hits=%d, last=%s\n\n",
           cache.page_views, cache.cache_hits, cache.last_page);

    // We simulate 5 clients -> 5 threads
    pthread_t threads[5];

    // Each client has its own request data (safe: each thread reads its own struct)
    Request requests[5] = {
        {1, "/home"},
        {2, "/about"},
        {3, "/products"},
        {4, "/contact"},
        {5, "/home"}
    };

    printf("[Server] Creating threads for 5 clients...\n\n");

    // ------------------------------------------------------------
    // CREATE THREADS
    // Each thread handles one request concurrently.
    // ------------------------------------------------------------
    for (int i = 0; i < 5; i++) {
        pthread_create(&threads[i], NULL, handle_request, &requests[i]);
    }

    // ------------------------------------------------------------
    // JOIN THREADS
    // Main thread waits until all client threads finish.
    // ------------------------------------------------------------
    for (int i = 0; i < 5; i++) {
        pthread_join(threads[i], NULL);
    }

    // ------------------------------------------------------------
    // MUTEX DESTRUCTION
    // Clean up lock after all threads are done.
    // ------------------------------------------------------------
    pthread_mutex_destroy(&cache_lock);

    printf("[Server] All threads completed\n");
    printf("[Server] Mutex destroyed\n\n");

    // Final results
    printf("=============================================\n");
    printf(" Final Results\n");
    printf("=============================================\n");

    printf("[Server] Final cache: views=%d, hits=%d, last=%s\n",
           cache.page_views, cache.cache_hits, cache.last_page);
    printf("[Server] Expected: views=5, hits=5\n");

    // Check if all increments were recorded correctly
    if (cache.page_views == 5 && cache.cache_hits == 5) {
        printf("[Server] Status: SUCCESS - All updates recorded correctly!\n");
    } else {
        printf("[Server] Status: ERROR - Some updates were lost!\n");
    }

    printf("=============================================\n");

    return 0;
}

