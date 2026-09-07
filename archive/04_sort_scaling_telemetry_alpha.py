# Reconstructed alpha version from the September 2026 benchmark session.
# Adds Windows/CUDA/NVIDIA memory telemetry and optional no-copy-back validation.

import argparse
import ctypes
import gc
import subprocess
import time
import numpy as np
import cupy as cp


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def parse_size(text):
    s = text.lower()
    if s.endswith("gib"):
        return int(float(s[:-3]) * 1024**3) // 2
    if s.endswith("mib"):
        return int(float(s[:-3]) * 1024**2) // 2
    if s.endswith("el"):
        return int(s[:-2])
    raise ValueError("Use GiB, MiB, or el")


def show_telemetry():
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    print(f"CUDA: {(total_mem-free_mem)/1024**3:.3f} GiB used, {free_mem/1024**3:.3f} GiB free")

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    used = status.ullTotalPhys - status.ullAvailPhys
    print(f"RAM:  {used/1024**3:.3f} GiB used, {status.ullAvailPhys/1024**3:.3f} GiB available")

    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10
        )
        print("nvidia-smi MiB used,free,total:", p.stdout.strip())
    except Exception:
        print("nvidia-smi telemetry unavailable")


def gpu_sorted(gpu):
    chunk = 16 * 1024 * 1024
    previous = None
    for start in range(0, gpu.size, chunk):
        part = gpu[start:min(start + chunk, gpu.size)]
        first = int(part[0].get())
        if previous is not None and previous > first:
            return False
        if part.size > 1 and not bool(cp.all(part[:-1] <= part[1:]).get()):
            return False
        previous = int(part[-1].get())
    return True


parser = argparse.ArgumentParser()
parser.add_argument("--size", nargs="+", required=True)
parser.add_argument("--telemetry", action="store_true")
parser.add_argument("--no-copy-back", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(12345)

for requested in args.size:
    n = parse_size(requested)
    print(f"\n=== {requested}: {n:,} uint16 ===")
    if args.telemetry:
        show_telemetry()

    host = rng.integers(0, 65536, size=n, dtype=np.uint16)

    t0 = time.perf_counter()
    gpu = cp.asarray(host)
    cp.cuda.Stream.null.synchronize()
    h2d = time.perf_counter() - t0

    del host
    gc.collect()

    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record(); gpu.sort(); end.record(); end.synchronize()
    sort_time = cp.cuda.get_elapsed_time(start, end) / 1000.0

    if args.no_copy_back:
        d2h = None
        valid = gpu_sorted(gpu)
    else:
        t0 = time.perf_counter()
        result = cp.asnumpy(gpu)
        cp.cuda.Stream.null.synchronize()
        d2h = time.perf_counter() - t0
        valid = bool(np.all(result[:-1] <= result[1:]))
        del result

    print(f"H->GPU: {h2d:.3f} s")
    print(f"GPU sort: {sort_time:.3f} s")
    print("GPU->H: SKIP" if d2h is None else f"GPU->H: {d2h:.3f} s")
    print(f"Valid: {valid}")

    del gpu
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
