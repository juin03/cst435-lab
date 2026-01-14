import os
import time
import random
import threading
from typing import List
from datetime import datetime

# Configuration Parameters
TEST_FILES_DIR = "test_files"
FILE_NAMES = [f"file_{i}.txt" for i in range(1, 6)]  # 5 test files
PROCESS_DELAY_RANGE = (1, 4)  # Simulate processing time for each file (1-4 seconds)

def create_test_files() -> None:
    """Create test files (simulating user-uploaded files)."""
    # Create directory (if it doesn't exist)
    if not os.path.exists(TEST_FILES_DIR):
        os.makedirs(TEST_FILES_DIR)

    # Generate content for each test file (random size text)
    for file_name in FILE_NAMES:
        file_path = os.path.join(TEST_FILES_DIR, file_name)
        # Randomly generate 100-1000 lines of text
        lines = [f"Test line {i} for {file_name}\n" for i in range(random.randint(100, 1000))]
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    print(f"✅ Created {len(FILE_NAMES)} test files.")

def process_file(file_path: str) -> None:
    """Simulate file processing logic, printing detailed time and thread info."""
    file_name = os.path.basename(file_path)

    # --- New: Get current thread info and start time ---
    current_thread = threading.current_thread()
    thread_id = threading.get_ident()  # Get Thread ID (integer identifier)
    start_timestamp = time.time()  # Get start timestamp
    # Format time (HH:MM:SS.mmm)
    start_time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"🚀 [Start] File: {file_name} | Thread ID: {thread_id} | Time: {start_time_str}")

    # Simulate time-consuming operation (e.g., format conversion, scanning)
    process_delay = random.uniform(*PROCESS_DELAY_RANGE)
    time.sleep(process_delay)

    # Actual I/O logic: Count characters and words
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        char_count = len(content)
        word_count = len(content.split())

    # --- New: Get end time and calculate total duration ---
    end_timestamp = time.time()
    end_time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    actual_duration = end_timestamp - start_timestamp  # Calculate actual execution duration

    # Generate processing report (including new ID and time info)
    report = (
        f"\n📝 File Processing Report - {file_name}\n"
        f"🧵 Thread ID : {thread_id} ({current_thread.name})\n"
        f"⏱ Start Time: {start_time_str}\n"
        f"🏁 End Time : {end_time_str}\n"
        f"⏳ Duration : {actual_duration:.4f}s (Simulated delay: {process_delay:.2f}s)\n"
        f"📊 Stats : {char_count} chars, {word_count} words\n"
        "-------------------------"
    )

    # Print report (in real scenarios, this might be logged to a DB or file)
    print(report)

def single_thread_process(files: List[str]) -> None:
    """Process files using a single thread (Sequentially)."""
    print(f"\n{'='*20} Single-Threaded Processing (Main Thread ID: {threading.get_ident()}) {'='*20}")
    start_time = time.time()

    # Process each file sequentially
    for file_path in files:
        process_file(file_path)

    total_time = time.time() - start_time
    print(f"\n✅ Single-thread processing complete! Total time: {total_time:.2f}s")

def multi_thread_process(files: List[str]) -> None:
    """Process files using multiple threads (Concurrently)."""
    print(f"\n{'='*20} Multi-Threaded Processing (Main Thread ID: {threading.get_ident()}) {'='*20}")
    start_time = time.time()

    # Create a list to hold the threads
    threads = []
    for file_path in files:
        # Create an independent thread for each file
        thread = threading.Thread(
            target=process_file,
            args=(file_path,),
            name=f"Worker-{os.path.basename(file_path)}"
        )
        threads.append(thread)
        thread.start()  # Start the thread

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    total_time = time.time() - start_time
    print(f"\n✅ Multi-thread processing complete! Total time: {total_time:.2f}s")

if __name__ == "__main__":
    # 1. Create test files
    create_test_files()

    # 2. Get paths for all test files
    file_paths = [os.path.join(TEST_FILES_DIR, name) for name in FILE_NAMES]

    # 3. Single-threaded processing
    single_thread_process(file_paths)

    # 4. Multi-threaded processing
    multi_thread_process(file_paths)

    # Optional: Clean up test files
    # for file_path in file_paths:
    #     os.remove(file_path)
    # if os.path.exists(TEST_FILES_DIR):
    #     os.rmdir(TEST_FILES_DIR)
