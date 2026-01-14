# https://colab.research.google.com/drive/1lOt_cTtCSNuXdHb0Qz-WcgWnFTp8HVwn?usp=sharing#scrollTo=61I7Tdao02yj
# See this yaoxiang for more details
# Purpose:
# This Python program benchmarks matrix multiplication on both CPU and GPU using PyTorch.
# It demonstrates the performance difference between CPU and GPU for large matrix operations.
# Key Features:
# - Uses PyTorch for matrix operations.
# - Measures execution time for different matrix sizes.
# - Visualizes the performance comparison using matplotlib.

import torch
import time
import matplotlib.pyplot as plt

# 1. Check if GPU is available
if not torch.cuda.is_available():
    print("⚠ Warning: No GPU detected! Please enable GPU in Runtime settings.")
    device_name = "CPU Only"
else:
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU Detected: {device_name}")

def benchmark_matmul(size, device_type='cpu'):
    """
    Function to benchmark matrix multiplication.
    size: Matrix size (N x N)
    device_type: 'cpu' or 'cuda'
    """
    # Prepare Data: Create two random large matrices (N x N)
    # float32 is the standard precision for deep learning
    A = torch.randn(size, size, dtype=torch.float32)
    B = torch.randn(size, size, dtype=torch.float32)

    # Move data to the specified device
    device = torch.device(device_type)
    A = A.to(device)
    B = B.to(device)

    # Warm-up: GPU has initialization overhead on the first run.
    if device_type == 'cuda':
        _ = torch.matmul(A, B)
        torch.cuda.synchronize() # Wait for warm-up to finish

    # Start Timing
    start_time = time.time()

    # Perform matrix multiplication (C = A * B)
    C = torch.matmul(A, B)

    # CRITICAL: GPU execution is asynchronous.
    # Force CPU to wait for GPU to finish to get accurate timing.
    if device_type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

# --- Start Experiment ---
# Define different matrix sizes: from small to large
matrix_sizes = [2000, 5000, 8000, 10000]
cpu_times = []
gpu_times = []

print(f"\n--- Benchmark Started (Device: {device_name}) ---")
print(f"{'Matrix Size':<15} | {'CPU Time (s)':<15} | {'GPU Time (s)':<15} | {'Speedup':<10}")
print("-" * 65)

for size in matrix_sizes:
    # Test CPU (Limit max size to prevent freezing)
    if size > 10000:
        print(f"Skipping CPU for size {size} (too slow)")
        cpu_t = float('inf')
    else:
        cpu_t = benchmark_matmul(size, 'cpu')

    # Test GPU
    gpu_t = benchmark_matmul(size, 'cuda')

    # Record data
    cpu_times.append(cpu_t)
    gpu_times.append(gpu_t)
    speedup = cpu_t / gpu_t
    print(f"{size}x{size:<9} | {cpu_t:.4f} | {gpu_t:.4f} | {speedup:.1f}x")

# --- Visualize Results ---
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, cpu_times, marker='o', label='CPU (Intel Xeon)',
         color='red', linestyle='--')
plt.plot(matrix_sizes, gpu_times, marker='x', label=f'GPU ({device_name})',
         color='green', linewidth=2)
plt.title('CPU vs GPU Benchmark: Matrix Multiplication')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Time (Seconds) - Lower is Better')
plt.legend()
plt.grid(True)
plt.show()