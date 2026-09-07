import argparse
import time
import gc
import numpy as np
import cupy as cp

parser = argparse.ArgumentParser(
    description="Benchmark large uint16 sorts on CPU and RTX GPU."
)

parser.add_argument(
    "--cpu-sort",
    action="store_true",
    help="Also run and time NumPy CPU sorting. Off by default."
)

args = parser.parse_args()

SIZES_MIB = [
    4096,   # 4.0 GiB
    5120,   # 5.0 GiB
    6144,   # 6.0 GiB
    6656,   # 6.5 GiB
    7168,   # 7.0 GiB
    7680,   # 7.5 GiB
    8192,   # 8.0 GiB
    8704,   # 8.5 GiB
]

gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()

print()
print("GPU:", gpu_name)
print("CPU sorting:", "ENABLED" if args.cpu_sort else "DISABLED")
print()

rng = np.random.default_rng(12345)
results = []

print("Warming up GPU...")

warm = cp.array([5, 3, 4, 1, 2], dtype=cp.uint16)
warm.sort()
cp.cuda.Stream.null.synchronize()

del warm

cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

print("Warmup complete.")
print()

for size_mib in SIZES_MIB:

    n = size_mib * 1024 * 1024 // 2
    gib = size_mib / 1024

    print("=" * 72)
    print(f"TEST: {gib:.1f} GiB")
    print(f"{size_mib:,} MiB")
    print(f"{n:,} uint16 numbers")
    print("=" * 72)

    free_mem, total_mem = cp.cuda.runtime.memGetInfo()

    free_gib = free_mem / 1024**3
    total_gib = total_mem / 1024**3

    print(
        f"GPU memory before test: "
        f"{free_gib:.2f} GiB free / "
        f"{total_gib:.2f} GiB total"
    )

    if gib >= free_gib:
        print()
        print(
            f"STOPPING: Dataset is {gib:.2f} GiB but only "
            f"{free_gib:.2f} GiB of GPU memory is currently free."
        )
        break

    a = None
    cpu = None
    gpu = None
    gpu_result = None

    try:

        print("Generating random data...")

        t0 = time.perf_counter()

        a = rng.integers(
            0,
            65536,
            size=n,
            dtype=np.uint16
        )

        generation_time = time.perf_counter() - t0

        print(
            f"Generation:             "
            f"{generation_time:9.3f} sec"
        )

        cpu_time = None

        if args.cpu_sort:

            print("Making CPU working copy...")

            cpu = a.copy()

            print("Running NumPy CPU sort...")

            t0 = time.perf_counter()

            cpu.sort()

            cpu_time = time.perf_counter() - t0

            print(
                f"CPU NumPy sort:         "
                f"{cpu_time:9.3f} sec"
            )

        print("Copying data to RTX 3080...")

        cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()

        gpu = cp.asarray(a)

        cp.cuda.Stream.null.synchronize()

        to_gpu_time = time.perf_counter() - t0

        print(
            f"RAM -> GPU:             "
            f"{to_gpu_time:9.3f} sec"
        )

        del a
        a = None

        gc.collect()

        free_before_sort, _ = cp.cuda.runtime.memGetInfo()

        print(
            f"GPU free before sort:   "
            f"{free_before_sort / 1024**3:9.3f} GiB"
        )

        print("Running GPU sort...")

        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()

        gpu.sort()

        end_event.record()
        end_event.synchronize()

        gpu_sort_time = (
            cp.cuda.get_elapsed_time(
                start_event,
                end_event
            ) / 1000.0
        )

        print(
            f"GPU sort only:          "
            f"{gpu_sort_time:9.3f} sec"
        )

        print("Copying sorted result back to RAM...")

        t0 = time.perf_counter()

        gpu_result = cp.asnumpy(gpu)

        cp.cuda.Stream.null.synchronize()

        from_gpu_time = time.perf_counter() - t0

        print(
            f"GPU -> RAM:             "
            f"{from_gpu_time:9.3f} sec"
        )

        print("Validating result...")

        if args.cpu_sort:

            matches = np.array_equal(
                cpu,
                gpu_result
            )

            validation_text = (
                f"Matches CPU result:     {matches}"
            )

        else:

            sorted_ok = bool(
                np.all(
                    gpu_result[:-1]
                    <= gpu_result[1:]
                )
            )

            validation_text = (
                f"Sorted correctly:       {sorted_ok}"
            )

            matches = sorted_ok

        gpu_total = (
            to_gpu_time
            + gpu_sort_time
            + from_gpu_time
        )

        gpu_sort_throughput = gib / gpu_sort_time
        gpu_total_throughput = gib / gpu_total

        print()
        print("-" * 72)

        if args.cpu_sort:
            print(
                f"CPU sort:               "
                f"{cpu_time:9.3f} sec"
            )

        print(
            f"RAM -> GPU:             "
            f"{to_gpu_time:9.3f} sec"
        )

        print(
            f"GPU sort:               "
            f"{gpu_sort_time:9.3f} sec"
        )

        print(
            f"GPU -> RAM:             "
            f"{from_gpu_time:9.3f} sec"
        )

        print(
            f"GPU end-to-end:         "
            f"{gpu_total:9.3f} sec"
        )

        print()
        print(
            f"GPU sort throughput:    "
            f"{gpu_sort_throughput:9.2f} GiB/sec"
        )

        print(
            f"GPU total throughput:   "
            f"{gpu_total_throughput:9.2f} GiB/sec"
        )

        if args.cpu_sort:

            print(
                f"GPU total speedup:      "
                f"{cpu_time / gpu_total:9.2f}x"
            )

        print(validation_text)

        results.append({
            "gib": gib,
            "h2d": to_gpu_time,
            "gpu": gpu_sort_time,
            "d2h": from_gpu_time,
            "total": gpu_total,
            "cpu": cpu_time,
            "ok": matches,
        })

    except cp.cuda.memory.OutOfMemoryError:

        print()
        print("*" * 72)
        print(
            f"GPU OUT OF MEMORY while processing "
            f"{gib:.1f} GiB."
        )
        print(
            "Stopping here rather than attempting "
            "still-larger datasets."
        )
        print("*" * 72)

        break

    except MemoryError:

        print()
        print("*" * 72)
        print(
            f"SYSTEM RAM allocation failed at "
            f"{gib:.1f} GiB."
        )
        print("Stopping benchmark.")
        print("*" * 72)

        break

    finally:

        if a is not None:
            del a

        if cpu is not None:
            del cpu

        if gpu is not None:
            del gpu

        if gpu_result is not None:
            del gpu_result

        gc.collect()

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        try:
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass

        print()

print()
print("=" * 96)
print("FINAL SCALING RESULTS")
print("=" * 96)

if args.cpu_sort:

    print(
        f"{'Size':>8} "
        f"{'CPU':>10} "
        f"{'H->GPU':>10} "
        f"{'GPU sort':>10} "
        f"{'GPU->H':>10} "
        f"{'GPU total':>11} "
        f"{'Speedup':>10}"
    )

    for r in results:

        print(
            f"{r['gib']:7.1f}G "
            f"{r['cpu']:9.3f}s "
            f"{r['h2d']:9.3f}s "
            f"{r['gpu']:9.3f}s "
            f"{r['d2h']:9.3f}s "
            f"{r['total']:10.3f}s "
            f"{r['cpu'] / r['total']:9.2f}x"
        )

else:

    print(
        f"{'Size':>8} "
        f"{'H->GPU':>10} "
        f"{'GPU sort':>10} "
        f"{'GPU->H':>10} "
        f"{'GPU total':>11} "
        f"{'Sort GiB/s':>12}"
    )

    for r in results:

        print(
            f"{r['gib']:7.1f}G "
            f"{r['h2d']:9.3f}s "
            f"{r['gpu']:9.3f}s "
            f"{r['d2h']:9.3f}s "
            f"{r['total']:10.3f}s "
            f"{r['gib'] / r['gpu']:11.2f}"
        )

print()
print("Benchmark complete.")
