import time
import gc
import numpy as np
import cupy as cp

SIZES_MIB = [512, 1024, 2048, 3072]

gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()

print("GPU:", gpu_name)
print("Testing:", ", ".join(f"{x} MiB" for x in SIZES_MIB))
print()

rng = np.random.default_rng(12345)
results = []

# GPU warmup
warm = cp.array([5, 3, 4, 1, 2], dtype=cp.uint16)
warm.sort()
cp.cuda.Stream.null.synchronize()
del warm
cp.get_default_memory_pool().free_all_blocks()

for size_mib in SIZES_MIB:

    n = size_mib * 1024 * 1024 // 2
    gib = size_mib / 1024

    print("=" * 64)
    print(f"{size_mib:,} MiB ({gib:.1f} GiB)")
    print(f"{n:,} uint16 numbers")
    print("=" * 64)

    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    print(
        f"GPU memory before test: "
        f"{free_mem / 1024**3:.2f} GiB free / "
        f"{total_mem / 1024**3:.2f} GiB total"
    )

    print("Generating random data...")
    t0 = time.perf_counter()
    a = rng.integers(0, 65536, size=n, dtype=np.uint16)
    generation_time = time.perf_counter() - t0
    print(f"Generation:            {generation_time:8.3f} sec")

    # --------------------------------------------------
    # CPU sort
    # --------------------------------------------------

    print("Making CPU working copy...")
    cpu = a.copy()

    print("Running NumPy CPU sort...")
    t0 = time.perf_counter()
    cpu.sort()
    cpu_time = time.perf_counter() - t0

    print(f"CPU NumPy sort:        {cpu_time:8.3f} sec")

    # --------------------------------------------------
    # RAM -> GPU
    # --------------------------------------------------

    print("Copying to RTX 3080...")

    try:
        cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()
        gpu = cp.asarray(a)
        cp.cuda.Stream.null.synchronize()
        to_gpu_time = time.perf_counter() - t0

        print(f"RAM -> GPU:            {to_gpu_time:8.3f} sec")

        # Original host input no longer needed
        del a
        gc.collect()

        # --------------------------------------------------
        # GPU sort -- CUDA event timing
        # --------------------------------------------------

        print("Running GPU sort...")

        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()

        gpu.sort()

        end_event.record()
        end_event.synchronize()

        gpu_sort_time = (
            cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
        )

        print(f"GPU sort only:         {gpu_sort_time:8.3f} sec")

        # --------------------------------------------------
        # GPU -> RAM
        # --------------------------------------------------

        print("Copying sorted array back to RAM...")

        t0 = time.perf_counter()
        gpu_result = cp.asnumpy(gpu)
        cp.cuda.Stream.null.synchronize()
        from_gpu_time = time.perf_counter() - t0

        print(f"GPU -> RAM:            {from_gpu_time:8.3f} sec")

        gpu_total = to_gpu_time + gpu_sort_time + from_gpu_time

        print("Validating...")
        matches = np.array_equal(cpu, gpu_result)

        cpu_throughput = gib / cpu_time
        gpu_sort_throughput = gib / gpu_sort_time
        gpu_total_throughput = gib / gpu_total

        print()
        print(f"CPU sort:              {cpu_time:8.3f} sec")
        print(f"GPU sort only:         {gpu_sort_time:8.3f} sec")
        print(f"GPU end-to-end:        {gpu_total:8.3f} sec")
        print()
        print(f"CPU throughput:        {cpu_throughput:8.2f} GiB/sec")
        print(f"GPU sort throughput:   {gpu_sort_throughput:8.2f} GiB/sec")
        print(f"GPU total throughput:  {gpu_total_throughput:8.2f} GiB/sec")
        print()
        print(f"GPU sort-only speedup: {cpu_time/gpu_sort_time:8.2f}x")
        print(f"GPU total speedup:     {cpu_time/gpu_total:8.2f}x")
        print(f"Results match:         {matches}")

        results.append((
            size_mib,
            n,
            cpu_time,
            to_gpu_time,
            gpu_sort_time,
            from_gpu_time,
            gpu_total,
            cpu_time / gpu_total,
            matches
        ))

        del gpu
        del gpu_result
        del cpu

    except cp.cuda.memory.OutOfMemoryError:
        print()
        print("*** GPU OUT OF MEMORY at this size ***")
        del cpu

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    print()

print()
print("=" * 105)
print("FINAL SCALING RESULTS")
print("=" * 105)

print(
    f"{'Size':>8} "
    f"{'Elements':>16} "
    f"{'CPU':>9} "
    f"{'H->GPU':>9} "
    f"{'GPU sort':>9} "
    f"{'GPU->H':>9} "
    f"{'GPU total':>10} "
    f"{'Speedup':>9}"
)

for r in results:
    size_mib, n, cpu_t, h2d, gpu_t, d2h, total_t, speedup, matches = r

    print(
        f"{size_mib:7,d}M "
        f"{n:16,d} "
        f"{cpu_t:8.3f}s "
        f"{h2d:8.3f}s "
        f"{gpu_t:8.3f}s "
        f"{d2h:8.3f}s "
        f"{total_t:9.3f}s "
        f"{speedup:8.2f}x"
    )
