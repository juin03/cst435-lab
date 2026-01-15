# Purpose:
# This Python program compares multithreading and multiprocessing for processing large CSV files.
# It demonstrates the performance difference between the two approaches.
# Key Features:
# - Uses ThreadPoolExecutor for multithreading.
# - Uses multiprocessing.Pool for multiprocessing.
# - Processes CSV data in chunks and performs classification tasks.
# - Measures execution time and reports CPU core/thread usage.

# Expected Output:
# - Logs for each chunk processed, including thread/process ID, core ID, and duration.
# - Total processing time for multithreading and multiprocessing.
# - Example:
#   🧵 [Thread] Data Chunk ID: 1 ---> CPU Core ID: 2
#    ℹ Identity Info: PID:12345 | TID:67890
#    ⏱ Time Consumed: 0.1234s
#    📊 Classification Result: {'Male - Small': 100, 'Female - Large': 200}
#   ⚙️ [Process] Data Chunk ID: 2 ---> CPU Core ID: 3
#    ℹ Identity Info: PID:12346 | TID:N/A
#    ⏱ Time Consumed: 0.2345s
#    📊 Classification Result: {'Male - Small': 150, 'Female - Large': 250}

import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing
import time
import os
import threading
from datetime import datetime

# Try to import psutil to get the real CPU core ID
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ================= Configuration Parameters =================
FILENAME = "shopping_behavior_updated.csv"
CHUNK_SIZE = 1000


def get_core_id():
    """Get CPU core ID specifically"""
    if HAS_PSUTIL:
        try:
            p = psutil.Process()
            return p.cpu_num()
        except:
            return "N/A"
    return "N/A (psutil not installed)"


def get_thread_info():
    """Get PID and TID"""
    pid = os.getpid()
    tid = threading.get_ident()
    return f"PID:{pid} | TID:{tid}"


def process_chunk(chunk_data):
    """
    General processing logic
    """
    chunk_id, chunk_df = chunk_data
    start_time = time.time()

    # 1. Get core ID
    core_id = get_core_id()

    # 2. Data cleaning
    chunk_df.columns = [c.strip() for c in chunk_df.columns]

    local_counts = pd.Series(dtype=int)

    # 3. Classification logic
    if 'Size' in chunk_df.columns and 'Gender' in chunk_df.columns:
        chunk_df['Size_Group'] = np.where(
            chunk_df['Size'].isin(['S', 'M']),
            'Small',
            'Large'
        )
        chunk_df['Group_Key'] = (
            chunk_df['Gender'] + ' - ' + chunk_df['Size_Group']
        )
        local_counts = chunk_df['Group_Key'].value_counts()

    duration = time.time() - start_time

    return {
        'counts': local_counts,
        'chunk_id': chunk_id,
        'duration': duration,
        'core_id': core_id,
        'thread_info': get_thread_info()
    }


def run_threading():
    print(f"\n{'='*20} Method 1: Concurrent Futures {'='*20}")
    start_time = time.time()
    total_counts = pd.Series(dtype=int)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        reader = pd.read_csv(FILENAME, chunksize=CHUNK_SIZE)
        futures = []

        for i, chunk in enumerate(reader):
            futures.append(executor.submit(process_chunk, (i + 1, chunk)))

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            total_counts = total_counts.add(res['counts'], fill_value=0)

            print(f"🧵 [Thread] Data Chunk ID: {res['chunk_id']} ---> CPU Core ID: {res['core_id']}")
            print(f" ℹ Identity Info: {res['thread_info']}")
            print(f" ⏱ Time Consumed: {res['duration']:.4f}s")
            print(f" 📊 Classification Result: {res['counts'].to_dict()}")
            print("-" * 50)

    end_time = time.time()
    return end_time - start_time


def run_multiprocessing():
    print(f"\n{'='*20} Method 2: Multiprocessing {'='*20}")
    start_time = time.time()
    total_counts = pd.Series(dtype=int)

    chunks = []
    reader = pd.read_csv(FILENAME, chunksize=CHUNK_SIZE)
    for i, chunk in enumerate(reader):
        chunks.append((i + 1, chunk))

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(process_chunk, chunks)

    for res in results:
        total_counts = total_counts.add(res['counts'], fill_value=0)

        print(f"⚙️ [Process] Data Chunk ID: {res['chunk_id']} ---> CPU Core ID: {res['core_id']}")
        print(f" ℹ Identity Info: {res['thread_info']}")
        print(f" ⏱ Time Consumed: {res['duration']:.4f}s")
        print(f" 📊 Classification Result: {res['counts'].to_dict()}")
        print("-" * 50)

    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    if os.path.exists(FILENAME):
        # 1. Run multithreading
        time_thread = run_threading()

        # 2. Run multiprocessing
        time_process = run_multiprocessing()

        # 3. Final comparison
        print(f"\n{'='*20} 📌 Final Comparison Result {'='*20}")
        print(f"1. Concurrent Futures Total Time : {time_thread:.4f} seconds")
        print(f"2. Multiprocessing Total Time    : {time_process:.4f} seconds")
    else:
        print(f"❌ File not found: {FILENAME}")
