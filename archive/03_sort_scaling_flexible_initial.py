# Reconstructed alpha version from the September 2026 benchmark session.
# Initial flexible-size variant used for exact boundary probes.

import argparse
import gc
import time
import numpy as np
import cupy as cp


def parse_size(text):
    s = text.lower()
    if s.endswith("gib"):
        size_bytes = int(float(s[:-3]) * 1024**3)
    elif s.endswith("mib"):
        size_bytes = int(float(s[:-3]) * 1024**2)
    elif s.endswith("el"):
        return int(s[:-2])
    elif s.endswith("b"):
        size_bytes = int(s[:-1])
    else:
        raise ValueError("Use GiB, MiB, B, or el")
    return size_bytes // 2


parser = argparse.ArgumentParser()
parser.add_argument("--size", nargs="+", required=True)
args = parser.parse_args()

rng = np.random.default_rng(12345)

for requested in args.size:
    n = parse_size(requested)
    size_gib = (n * 2) / 1024**3
    print(f"\n{requested}: {n:,} uint16 ({size_gib:.9f} GiB)")

    host = rng.integers(0, 65536, size=n, dtype=np.uint16)

    t0 = time.perf_counter()
    gpu = cp.asarray(host)
    cp.cuda.Stream.null.synchronize()
    h2d = time.perf_counter() - t0

    free_before_sort, _ = cp.cuda.runtime.memGetInfo()

    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record(); gpu.sort(); end.record(); end.synchronize()
    sort_time = cp.cuda.get_elapsed_time(start, end) / 1000.0

    t0 = time.perf_counter()
    result = cp.asnumpy(gpu)
    cp.cuda.Stream.null.synchronize()
    d2h = time.perf_counter() - t0

    print(f"H2D {h2d:.3f} s; sort {sort_time:.3f} s; D2H {d2h:.3f} s")
    print(f"CUDA free before sort: {free_before_sort / 1024**3:.3f} GiB")
    print(f"Valid: {bool(np.all(result[:-1] <= result[1:]))}")

    del host, gpu, result
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
