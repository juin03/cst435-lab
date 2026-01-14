/*
----------------------------------------------------------------
 PURPOSE OF THIS PROGRAM
----------------------------------------------------------------
This program demonstrates a RACE CONDITION using THREADS.

Key idea:
• Threads SHARE memory.
• If two threads access shared data without synchronization,
  incorrect results can occur.

WHAT TO OBSERVE CAREFULLY:
1. Both ATM threads read the SAME initial balance.
2. A deliberate delay (usleep) simulates network/database lag.
3. Context switching occurs during this delay.
4. Both threads write back the SAME updated value.
5. Final balance is WRONG → this is called a LOST UPDATE.

This shows WHY synchronization is required in multithreading.
----------------------------------------------------------------

EXPECTED OUTPUT:
[System] Initial Balance: $1000
ATM 1: Checking balance...
ATM 2: Checking balance...
ATM 1: Dispensed $100. New Balance: $900
ATM 2: Dispensed $100. New Balance: $900
[System] Final Balance: $900 (Expected: 800)
----------------------------------------------------------------
*/

#include <stdio.h>      // printf
#include <stdlib.h>     // exit
#include <unistd.h>     // usleep
#include <pthread.h>    // pthreads

// -------------------------------------------------
// Shared Bank Account (GLOBAL variable)
// Shared by ALL threads → source of race condition
// -------------------------------------------------
int account_balance = 1000;

// -------------------------------------------------
// Thread function: Simulates an ATM withdrawal
// -------------------------------------------------
void* perform_withdrawal(void* arg) {

    int id = *((int*)arg);        // ATM ID (1 or 2)
    int withdraw_amount = 100;

    printf("ATM %d: Checking balance...\n", id);

    // -------------------------------
    // 1. READ (CRITICAL STEP)
    // Both threads may read SAME value
    // -------------------------------
    int local_balance = account_balance;

    // -------------------------------
    // 2. SIMULATE LATENCY
    // Forces a context switch here
    // -------------------------------
    usleep(100000); // 100 ms delay

    // -------------------------------
    // 3. CHECK AND WRITE
    // This write is NOT protected
    // -------------------------------
    if (local_balance >= withdraw_amount) {
        local_balance = local_balance - withdraw_amount;
        account_balance = local_balance;  // Write back
        printf("ATM %d: Dispensed $%d. New Balance: $%d\n",
               id, withdraw_amount, account_balance);
    } else {
        printf("ATM %d: Insufficient funds.\n", id);
    }

    return NULL;
}

int main() {

    pthread_t t1, t2;
    int id1 = 1, id2 = 2;

    printf("[System] Initial Balance: $%d\n", account_balance);

    // -------------------------------------------------
    // Two ATMs operate at the SAME TIME
    // -------------------------------------------------
    pthread_create(&t1, NULL, perform_withdrawal, &id1);
    pthread_create(&t2, NULL, perform_withdrawal, &id2);

    // Wait for both ATMs to finish
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    // Final balance is WRONG due to race condition
    printf("[System] Final Balance: $%d (Expected: 800)\n",
           account_balance);

    return 0;
}
