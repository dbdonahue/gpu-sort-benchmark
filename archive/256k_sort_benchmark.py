import time
import numpy as np
import cupy as cp

SIZE_MB = 512
N = SIZE_MB * 1024 * 1024 // 2

print(f"Creating {N:,} random uint16 numbers ({SIZE_MB} MiB)...")
rng = np.random.default_rng(12345)
a = rng.integers(0, 65536, size=N, dtype=np.uint16)

print(f"System array size: {a.nbytes / 1024**2:.1f} MiB")
print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
print()

# --------------------
# CPU: NumPy sort
# --------------------
cpu = a.copy()

print("Running CPU NumPy sort...")
t0 = time.perf_counter()
cpu.sort()
cpu_time = time.perf_counter() - t0

print(f"CPU NumPy sort:       {cpu_time:.3f} seconds")
print()

# --------------------
# GPU warm-up
# --------------------
warm = cp.array([5, 3, 4, 1, 2], dtype=cp.uint16)
warm.sort()
cp.cuda.Stream.null.synchronize()
del warm

# --------------------
# Host -> GPU transfer
# --------------------
print("Copying 512 MiB to RTX 3080...")
t0 = time.perf_counter()
gpu = cp.asarray(a)
cp.cuda.Stream.null.synchronize()
to_gpu_time = time.perf_counter() - t0

print(f"RAM -> GPU:           {to_gpu_time:.3f} seconds")

# --------------------
# GPU sort only
# --------------------
print("Running RTX 3080 sort...")
cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()

gpu.sort()

cp.cuda.Stream.null.synchronize()
gpu_sort_time = time.perf_counter() - t0

print(f"GPU sort only:        {gpu_sort_time:.3f} seconds")

# --------------------
# GPU -> Host transfer
# --------------------
print("Copying result back to system RAM...")
t0 = time.perf_counter()
gpu_result = cp.asnumpy(gpu)
cp.cuda.Stream.null.synchronize()
from_gpu_time = time.perf_counter() - t0

print(f"GPU -> RAM:           {from_gpu_time:.3f} seconds")

gpu_total = to_gpu_time + gpu_sort_time + from_gpu_time

print()
print("=" * 52)
print("RESULTS")
print("=" * 52)
print(f"CPU NumPy sort:       {cpu_time:8.3f} sec")
print(f"GPU transfer in:      {to_gpu_time:8.3f} sec")
print(f"GPU sort only:        {gpu_sort_time:8.3f} sec")
print(f"GPU transfer out:     {from_gpu_time:8.3f} sec")
print(f"GPU end-to-end:       {gpu_total:8.3f} sec")
print()

if gpu_sort_time > 0:
    print(f"GPU vs CPU sort-only: {cpu_time/gpu_sort_time:8.2f}x")

if gpu_total > 0:
    print(f"GPU vs CPU total:     {cpu_time/gpu_total:8.2f}x")

print()
print("Checking results...")
print("CPU sorted correctly:",
      bool(np.all(cpu[:-1] <= cpu[1:])))
print("GPU matches CPU:",
      bool(np.array_equal(cpu, gpu_result)))
