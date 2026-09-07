# Reconstructed alpha version from the September 2026 benchmark session.
# GPU-only scaling test for large uint16 datasets.

import gc
import time
import numpy as np
import cupy as cp

SIZES_GIB = [4.0, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
rng = np.random.default_rng(12345)

for size_gib in SIZES_GIB:
    size_bytes = int(size_gib * 1024**3)
    n = size_bytes // 2
    print(f"\n=== {size_gib:.1f} GiB / {n:,} uint16 ===")

    a = rng.integers(0, 65536, size=n, dtype=np.uint16)

    t0 = time.perf_counter()
    gpu = cp.asarray(a)
    cp.cuda.Stream.null.synchronize()
    h2d = time.perf_counter() - t0

    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record(); gpu.sort(); end.record(); end.synchronize()
    sort_time = cp.cuda.get_elapsed_time(start, end) / 1000.0

    t0 = time.perf_counter()
    out = cp.asnumpy(gpu)
    cp.cuda.Stream.null.synchronize()
    d2h = time.perf_counter() - t0

    total = h2d + sort_time + d2h
    valid = bool(np.all(out[:-1] <= out[1:]))

    print(f"H2D:   {h2d:.3f} s")
    print(f"Sort:  {sort_time:.3f} s")
    print(f"D2H:   {d2h:.3f} s")
    print(f"Total: {total:.3f} s")
    print(f"Sort throughput:  {size_gib / sort_time:.2f} GiB/s")
    print(f"Total throughput: {size_gib / total:.2f} GiB/s")
    print(f"Valid: {valid}")

    del a, gpu, out
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
