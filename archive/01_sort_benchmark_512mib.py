# Reconstructed alpha version from the September 2026 benchmark session.
# Fixed-size 512 MiB uint16 CPU/GPU comparison.

import time
import numpy as np
import cupy as cp

size_mib = 512
n = (size_mib * 1024 * 1024) // 2
rng = np.random.default_rng(12345)
a = rng.integers(0, 65536, size=n, dtype=np.uint16)

cpu = a.copy()
t0 = time.perf_counter()
cpu.sort()
cpu_time = time.perf_counter() - t0

t0 = time.perf_counter()
gpu = cp.asarray(a)
cp.cuda.Stream.null.synchronize()
h2d = time.perf_counter() - t0

start = cp.cuda.Event(); end = cp.cuda.Event()
start.record(); gpu.sort(); end.record(); end.synchronize()
gpu_sort = cp.cuda.get_elapsed_time(start, end) / 1000.0

t0 = time.perf_counter()
out = cp.asnumpy(gpu)
cp.cuda.Stream.null.synchronize()
d2h = time.perf_counter() - t0

total = h2d + gpu_sort + d2h
print(f"CPU NumPy sort: {cpu_time:.3f} s")
print(f"RAM -> GPU: {h2d:.3f} s")
print(f"GPU sort only: {gpu_sort:.3f} s")
print(f"GPU -> RAM: {d2h:.3f} s")
print(f"GPU end-to-end: {total:.3f} s")
print(f"GPU matches CPU: {np.array_equal(cpu, out)}")
