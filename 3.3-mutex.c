/*
----------------------------------------------------------------
 PURPOSE OF THIS PROGRAM
----------------------------------------------------------------
This program FIXES the race condition using a MUTEX.

Key idea:
• A mutex enforces MUTUAL EXCLUSION.
• Only ONE thread can enter the critical section at a time.

WHAT TO OBSERVE CAREFULLY:
1. Threads must request the lock before accessing balance.
2. Even with network delay (usleep), correctness is preserved.
3. Other threads WAIT instead of interfering.
4. Final balance is CORRECT.

This demonstrates the CORRECT way to protect shared data.
----------------------------------------------------------------

EXPECTED OUTPUT:
[System] Initial Balance: $1000
ATM 1: Requesting access...
ATM 1: Access granted. Checking balance...
ATM 1: Dispensed $100. New Balance: $900
ATM 2: Requesting access...
ATM 2: Access granted. Checking balance...
ATM 2: Dispensed $100. New Balance: $800
[System] Final Balance: $800 (Expected: 800)
----------------------------------------------------------------
*/

#include <stdio.h>      // printf
#include <stdlib.h>     // exit
#include <unistd.h>     // usleep
#include <pthread.h>    // pthreads

// -------------------------------------------------
// Shared Bank Account (still shared memory)
// -------------------------------------------------
int account_balance = 1000;

// -------------------------------------------------
// Mutex (Lock) to protect critical section
// -------------------------------------------------
pthread_mutex_t vault_lock;

// -------------------------------------------------
// Thread function: ATM withdrawal with locking
// -------------------------------------------------
void* perform_withdrawal(void* arg) {

    int id = *((int*)arg);
    int withdraw_amount = 100;

    printf("ATM %d: Requesting access...\n", id);

    // -------------------------------------------------
    // LOCK START
    // Only ONE thread can pass this line
    // -------------------------------------------------
    pthread_mutex_lock(&vault_lock);

    printf("ATM %d: Access granted. Checking balance...\n", id);

    int local_balance = account_balance;

    // Even with delay, data is safe
    usleep(100000);

    if (local_balance >= withdraw_amount) {
        local_balance = local_balance - withdraw_amount;
        account_balance = local_balance;
        printf("ATM %d: Dispensed $%d. New Balance: $%d\n",
               id, withdraw_amount, account_balance);
    } else {
        printf("ATM %d: Insufficient funds.\n", id);
    }

    // -------------------------------------------------
    // LOCK END
    // Other threads may now proceed
    // -------------------------------------------------
    pthread_mutex_unlock(&vault_lock);

    return NULL;
}

int main() {

    pthread_t t1, t2;
    int id1 = 1, id2 = 2;

    // Initialize the mutex
    pthread_mutex_init(&vault_lock, NULL);

    printf("[System] Initial Balance: $%d\n", account_balance);

    pthread_create(&t1, NULL, perform_withdrawal, &id1);
    pthread_create(&t2, NULL, perform_withdrawal, &id2);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    // Destroy the mutex
    pthread_mutex_destroy(&vault_lock);

    // Final balance is now CORRECT
    printf("[System] Final Balance: $%d (Expected: 800)\n",
           account_balance);

    return 0;
}

