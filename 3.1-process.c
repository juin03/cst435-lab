/*
------------------------------------------------------------
 PURPOSE OF THIS PROGRAM
------------------------------------------------------------
This program demonstrates a VERY IMPORTANT concept in
Operating Systems:

    👉 Processes created using fork() DO NOT share memory.

Even though the parent and child processes appear to work
on the same variable (account.balance), they are actually
working on SEPARATE COPIES of memory.

KEY THINGS TO OBSERVE:
1. The child process modifies the account balance.
2. The parent process does NOT see that modification.
3. Both processes print the address of account.balance,
   which may LOOK the same but belongs to different
   address spaces.
4. This shows PROCESS ISOLATION and COPY-ON-WRITE behavior.

This is why processes are SAFE but heavier than threads.
------------------------------------------------------------

EXPECTED OUTPUT:
[Main] Initial Balance: $1000
 [Branch A] Processing service fee of $50...
 [Branch A] Local Ledger Balance: $950 (Address: 0x...)
[Main] Final HQ Balance: $1000 (Address: 0x...)
------------------------------------------------------------
*/


#include <stdio.h>      // For printf()
#include <stdlib.h>     // For exit()
#include <unistd.h>     // For fork()
#include <sys/types.h>  // For pid_t
#include <sys/wait.h>   // For wait()

// ------------------------------
// Structure representing a bank account
// ------------------------------
typedef struct {
    int account_id;    // Unique account identifier
    int balance;       // Account balance
} BankAccount;

// ------------------------------
// Global account variable
// This variable will be COPIED when fork() is called
// ------------------------------
BankAccount account = {101, 1000}; // Initial balance = $1000

int main() {

    // ------------------------------
    // This runs BEFORE fork()
    // Only ONE process exists here
    // ------------------------------
    printf("[Main] Initial Balance: $%d\n", account.balance);

    // ------------------------------
    // fork() creates a NEW process
    // After this call:
    //  - Parent and child BOTH execute from here
    // pid_t is the process ID type
    // ------------------------------
    pid_t pid = fork();

    // ------------------------------
    // Error handling: fork failed
    // ------------------------------
    if (pid < 0) {
        perror("Fork failed");
        exit(1);
    }

    // ------------------------------
    // CHILD PROCESS
    // pid == 0 means we are inside child
    // ------------------------------
    else if (pid == 0) {

        // Child acts like "Branch A"
        printf(" [Branch A] Processing service fee of $50...\n");

        // Modify CHILD'S COPY of account
        // This does NOT affect the parent
        account.balance = account.balance - 50;

        // Print child's local balance and memory address
        printf(" [Branch A] Local Ledger Balance: $%d (Address: %p)\n",
               account.balance, &account.balance);

        // Child process exits here
        exit(0);
    }

    // ------------------------------
    // PARENT PROCESS
    // pid > 0 means we are inside parent
    // ------------------------------
    else {

        // Parent waits for child to finish
        // Ensures ordered output
        wait(NULL);

        // Parent prints its own copy of account
        // Balance remains unchanged
        printf("[Main] Final HQ Balance: $%d (Address: %p)\n",
               account.balance, &account.balance);
    }

    return 0;
}
