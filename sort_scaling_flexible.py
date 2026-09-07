import argparse
import ctypes
import gc
import importlib.util
import os
import re
import subprocess
import sys
import time
from decimal import Decimal, ROUND_HALF_UP


# ======================================================================
# BASIC CONFIGURATION
#
# Keep this section free of NumPy/CuPy dependencies so --help can work
# even when this script is launched with the wrong Python interpreter.
# ======================================================================

BYTES_PER_ELEMENT = 2       # uint16
VALIDATION_CHUNK_ELEMENTS = 16 * 1024 * 1024


# ======================================================================
# ENVIRONMENT DETECTION
# ======================================================================

def missing_benchmark_modules():

    missing = []

    for module_name in ("numpy", "cupy"):

        try:
            found = importlib.util.find_spec(module_name)
        except Exception:
            found = None

        if found is None:
            missing.append(module_name)

    return missing


def expected_environment_command(help_only=False):

    base = (
        r".\gpu-sort-test\Scripts\python.exe "
        r".\sort_scaling_flexible.py"
    )

    if help_only:
        return base + " --help"

    return base


YELLOW = "\033[93m" if sys.stdout.isatty() else ""
RED = "\033[91m" if sys.stdout.isatty() else ""
RESET = "\033[0m" if sys.stdout.isatty() else ""

def make_environment_warning():

    missing = missing_benchmark_modules()

    if not missing:
        return ""

    return f"""

{YELLOW}ENVIRONMENT WARNING
-------------------{RESET}

This script is currently running under:

    {sys.executable}

The following benchmark module(s) are not available in this Python
environment:

    {", ".join(missing)}

The benchmark itself will NOT run correctly from this Python interpreter.

You previously created a dedicated environment containing NumPy, CuPy,
CUDA support, and the other benchmark dependencies.

Use this command instead:

    {expected_environment_command(help_only=True)}

For an actual benchmark, use for example:

    {expected_environment_command()} --telemetry --size 4GiB 8GiB

The help text itself is being displayed successfully because the program
delays importing NumPy and CuPy until after command-line processing.
"""


# ======================================================================
# HELP TEXT
# ======================================================================

HELP_EPILOG = r"""
SIZE SYNTAX
-----------

Binary units -- normally recommended for RAM and GPU-memory testing:

    10GiB
    3.75GiB
    1024MiB
    256KiB

Full binary-unit names also work:

    10 gibibytes
    1024 mebibytes
    256 kibibytes

Decimal / approximate storage units:

    10GB
    10 gigabytes
    1024MB
    1024 megabytes
    256KB
    256 kilobytes

The difference matters:

    10 GiB = 10,737,418,240 bytes
    10 GB  = 10,000,000,000 bytes

For GPU VRAM experiments, GiB is usually the best unit.

Exact byte counts:

    4294967296B
    4294967296 bytes

Exact uint16 element counts:

    2147483648el
    2147483648 elements


MULTIPLE DATASET SIZES
----------------------

Supply as many sizes as desired:

    --size 1GiB 2GiB 4GiB 8GiB

Units may also be separate words:

    --size 10 gigabytes 12 gigabytes 16 gigabytes


OPTIONS
-------

--size SIZE [SIZE ...]

    Required.

    Specifies one or more dataset sizes.

--cpu-sort

    Also creates a host-side copy and sorts it with NumPy.

    This is OFF by default.

    Be careful with very large datasets because this requires another
    dataset-sized block of system RAM.

--telemetry

    Shows memory telemetry before every test, including:

      * CUDA-reported GPU memory
      * NVIDIA dedicated VRAM from nvidia-smi
      * Windows dedicated GPU-memory usage
      * Windows shared GPU/system-memory usage
      * physical system RAM used / available / total
      * Windows commit usage and commit limit

--no-copy-back

    Does not copy the complete sorted array from the GPU back into
    ordinary system RAM.

    Instead, the sorted result is validated directly on the GPU in
    manageable chunks.

    This is useful for very large capacity tests such as 16 GiB or
    32 GiB on a computer with 48 GiB of physical RAM.

--help, -h

    Displays this manual.

    Help works even if the script was accidentally launched outside the
    special gpu-sort-test Python environment.


FINAL TABLE COLUMNS
-------------------

Requested

    The dataset size exactly as requested on the command line.

GiB

    The actual resulting dataset size measured in binary GiB.

Elements

    Number of uint16 (two-byte) numbers in the dataset.

CPU sort

    NumPy CPU sorting time.

    Appears only if --cpu-sort was specified.

H->GPU

    Host-to-GPU transfer time.

    H means "host": ordinary CPU-accessible system RAM.

    This measures how long it takes to move the unsorted array from
    system RAM into GPU-accessible memory.

GPU sort

    The CUDA/GPU sorting time itself.

    This excludes the initial host-to-GPU transfer and, normally, the
    final GPU-to-host copy.

GPU->H

    GPU-to-host transfer time.

    This is the time required to copy the completed sorted array back
    into ordinary system RAM.

    Shows SKIP when --no-copy-back is specified.

Total

    Sum of the measured GPU-path operations.

    Normal test:

        H->GPU + GPU sort + GPU->H

    With --no-copy-back:

        H->GPU + GPU sort

Speedup

    CPU-sort time divided by GPU-path time.

    Appears only with --cpu-sort.

Status

    Indicates whether the result validated successfully or identifies
    the stage at which an error occurred.


EXAMPLES
--------

Show this help:

    .\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py --help

Normal GPU test:

    .\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py --size 4GiB 8GiB

Show memory telemetry:

    .\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py --telemetry --size 8GiB 10GiB 12GiB

Enable CPU comparison:

    .\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py --cpu-sort --size 512MiB 1GiB

Large capacity probe without copying the entire result back:

    .\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py --telemetry --no-copy-back --size 16GiB 32GiB

Decimal / approximate units:

    .\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py --size 10 gigabytes 1024 megabytes 256 kilobytes
"""


# ======================================================================
# SIZE PARSER
# ======================================================================

NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")

COMBINED_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$"
)

UNIT_FACTORS = {

    # Binary
    "gib": 1024 ** 3,
    "gibibyte": 1024 ** 3,
    "gibibytes": 1024 ** 3,

    "mib": 1024 ** 2,
    "mebibyte": 1024 ** 2,
    "mebibytes": 1024 ** 2,

    "kib": 1024,
    "kibibyte": 1024,
    "kibibytes": 1024,

    # Decimal
    "gb": 1000 ** 3,
    "g": 1000 ** 3,
    "gigabyte": 1000 ** 3,
    "gigabytes": 1000 ** 3,

    "mb": 1000 ** 2,
    "m": 1000 ** 2,
    "megabyte": 1000 ** 2,
    "megabytes": 1000 ** 2,

    "kb": 1000,
    "k": 1000,
    "kilobyte": 1000,
    "kilobytes": 1000,

    # Bytes
    "b": 1,
    "byte": 1,
    "bytes": 1,
}

ELEMENT_UNITS = {
    "el",
    "els",
    "element",
    "elements",
}


def make_size_spec(value_text, unit_text, original):

    value = Decimal(value_text)
    unit = unit_text.lower()

    if unit in ELEMENT_UNITS:

        if value != value.to_integral_value():
            raise ValueError(
                "Element counts must be whole numbers."
            )

        elements = int(value)
        byte_count = elements * BYTES_PER_ELEMENT

    elif unit in UNIT_FACTORS:

        raw_bytes = (
            value * Decimal(UNIT_FACTORS[unit])
        )

        byte_count = int(
            raw_bytes.to_integral_value(
                rounding=ROUND_HALF_UP
            )
        )

        # uint16 requires an even number of bytes.
        remainder = byte_count % BYTES_PER_ELEMENT

        if remainder:
            byte_count -= remainder

        elements = (
            byte_count // BYTES_PER_ELEMENT
        )

    else:

        raise ValueError(
            f"Unknown unit '{unit_text}'."
        )

    if elements <= 0:
        raise ValueError(
            "Dataset must contain at least one uint16 value."
        )

    return {
        "requested": original,
        "bytes": byte_count,
        "elements": elements,
    }


def parse_size_tokens(tokens):

    results = []
    i = 0

    while i < len(tokens):

        token = tokens[i].strip()

        combined = COMBINED_RE.match(token)

        if combined:

            try:

                results.append(
                    make_size_spec(
                        combined.group(1),
                        combined.group(2),
                        token
                    )
                )

            except ValueError as e:

                raise argparse.ArgumentTypeError(
                    f"{token}: {e}"
                )

            i += 1
            continue

        # Also accepts:
        #
        #     10 gigabytes

        if (
            NUMBER_RE.match(token)
            and i + 1 < len(tokens)
        ):

            unit_text = tokens[i + 1].strip()

            try:

                results.append(
                    make_size_spec(
                        token,
                        unit_text,
                        token + " " + unit_text
                    )
                )

            except ValueError as e:

                raise argparse.ArgumentTypeError(
                    f"{token} {unit_text}: {e}"
                )

            i += 2
            continue

        raise argparse.ArgumentTypeError(
            f"Cannot understand size '{token}'. "
            f"Use --help for examples."
        )

    return results


# ======================================================================
# ARGUMENT PARSER
#
# IMPORTANT:
# NumPy and CuPy have still NOT been imported at this point.
# ======================================================================

environment_warning = ""

if (
    "--help" in sys.argv[1:]
    or "-h" in sys.argv[1:]
):
    environment_warning = make_environment_warning()


parser = argparse.ArgumentParser(
    description=(
        "Flexible uint16 CPU/GPU sorting benchmark "
        "with optional Windows memory telemetry."
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=HELP_EPILOG + environment_warning,
)

parser.add_argument(
    "--size",
    nargs="+",
    metavar="SIZE",
    help=(
        "One or more dataset sizes. Examples: "
        "4GiB 10GB 1024MiB 256 kilobytes "
        "2147483648el"
    ),
)

parser.add_argument(
    "--cpu-sort",
    action="store_true",
    help=(
        "Also perform a NumPy CPU sort."
    ),
)

parser.add_argument(
    "--telemetry",
    action="store_true",
    help=(
        "Display Windows/CUDA/NVIDIA memory telemetry "
        "before each test."
    ),
)

parser.add_argument(
    "--no-copy-back",
    action="store_true",
    help=(
        "Do not copy the complete result back to "
        "system RAM."
    ),
)


args = parser.parse_args()


# --size is required for actual benchmarking, but deliberately NOT marked
# required in argparse so --help can always work cleanly.

if args.size is None:

    parser.error(
        "--size is required when running a benchmark. "
        "Use --help for usage information."
    )


try:
    size_specs = parse_size_tokens(args.size)

except argparse.ArgumentTypeError as e:
    parser.error(str(e))


# ======================================================================
# NOW LOAD THE BENCHMARK DEPENDENCIES
# ======================================================================

try:

    import numpy as np

except ModuleNotFoundError:

    print()
    print("=" * 78)
    print("BENCHMARK ENVIRONMENT NOT ACTIVE")
    print("=" * 78)
    print()
    print("NumPy is not available in the current Python environment.")
    print()
    print("Current Python:")
    print()
    print(f"    {sys.executable}")
    print()
    print("Use the benchmark environment instead:")
    print()
    print(
        r"    .\gpu-sort-test\Scripts\python.exe "
        r".\sort_scaling_flexible.py --help"
    )
    print()
    print("Example benchmark:")
    print()
    print(
        r"    .\gpu-sort-test\Scripts\python.exe "
        r".\sort_scaling_flexible.py "
        r"--telemetry --size 4GiB 8GiB"
    )
    print()

    sys.exit(2)


try:

    import cupy as cp

except ModuleNotFoundError:

    print()
    print("=" * 78)
    print("BENCHMARK ENVIRONMENT NOT ACTIVE")
    print("=" * 78)
    print()
    print("CuPy is not available in the current Python environment.")
    print()
    print("Current Python:")
    print()
    print(f"    {sys.executable}")
    print()
    print("Use the benchmark environment instead:")
    print()
    print(
        r"    .\gpu-sort-test\Scripts\python.exe "
        r".\sort_scaling_flexible.py --help"
    )
    print()
    print("Example benchmark:")
    print()
    print(
        r"    .\gpu-sort-test\Scripts\python.exe "
        r".\sort_scaling_flexible.py "
        r"--telemetry --size 4GiB 8GiB"
    )
    print()

    sys.exit(2)


DTYPE = np.uint16


# ======================================================================
# WINDOWS SYSTEM MEMORY TELEMETRY
# ======================================================================

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


def get_windows_memory():

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)

    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(
        ctypes.byref(status)
    )

    if not ok:
        return None

    return {
        "phys_total":
            status.ullTotalPhys,

        "phys_available":
            status.ullAvailPhys,

        "phys_used":
            status.ullTotalPhys
            - status.ullAvailPhys,

        "commit_limit":
            status.ullTotalPageFile,

        "commit_used":
            status.ullTotalPageFile
            - status.ullAvailPageFile,
    }


# ======================================================================
# NVIDIA-SMI TELEMETRY
# ======================================================================

def get_nvidia_smi_memory():

    try:

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu="
                "memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )

        line = result.stdout.strip().splitlines()[0]

        parts = [
            x.strip()
            for x in line.split(",")
        ]

        return {
            "used_mib": float(parts[0]),
            "free_mib": float(parts[1]),
            "total_mib": float(parts[2]),
        }

    except Exception:
        return None


# ======================================================================
# WINDOWS GPU MEMORY COUNTERS
# ======================================================================

def get_windows_gpu_memory():

    ps_command = r"""
$ErrorActionPreference = 'Stop'

$c = Get-Counter -Counter `
'\GPU Adapter Memory(*)\Dedicated Usage',`
'\GPU Adapter Memory(*)\Shared Usage'

$d = (
    $c.CounterSamples |
    Where-Object { $_.Path -like '*\Dedicated Usage' } |
    Measure-Object -Property CookedValue -Sum
).Sum

$s = (
    $c.CounterSamples |
    Where-Object { $_.Path -like '*\Shared Usage' } |
    Measure-Object -Property CookedValue -Sum
).Sum

Write-Output "$d|$s"
"""

    try:

        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                ps_command,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )

        line = result.stdout.strip().splitlines()[-1]

        dedicated, shared = line.split("|")

        return {
            "dedicated": float(dedicated),
            "shared": float(shared),
        }

    except Exception:
        return None


def gib(value):
    return value / (1024 ** 3)


def print_telemetry():

    print()
    print("MEMORY TELEMETRY BEFORE TEST")
    print("-" * 78)

    try:

        free_mem, total_mem = (
            cp.cuda.runtime.memGetInfo()
        )

        print(
            f"CUDA GPU memory:       "
            f"{gib(total_mem - free_mem):8.3f} GiB used, "
            f"{gib(free_mem):8.3f} GiB free, "
            f"{gib(total_mem):8.3f} GiB total"
        )

    except Exception:

        print(
            "CUDA GPU memory:       unavailable"
        )

    nv = get_nvidia_smi_memory()

    if nv:

        print(
            f"NVIDIA dedicated VRAM: "
            f"{nv['used_mib'] / 1024:8.3f} GiB used, "
            f"{nv['free_mib'] / 1024:8.3f} GiB free, "
            f"{nv['total_mib'] / 1024:8.3f} GiB total"
        )

    else:

        print(
            "NVIDIA dedicated VRAM: unavailable"
        )

    wg = get_windows_gpu_memory()

    if wg:

        print(
            f"Windows GPU dedicated: "
            f"{gib(wg['dedicated']):8.3f} GiB used"
        )

        print(
            f"Windows GPU shared:    "
            f"{gib(wg['shared']):8.3f} GiB system RAM"
        )

    else:

        print(
            "Windows GPU counters:  unavailable"
        )

    wm = get_windows_memory()

    if wm:

        print(
            f"System physical RAM:   "
            f"{gib(wm['phys_used']):8.3f} GiB used, "
            f"{gib(wm['phys_available']):8.3f} GiB available, "
            f"{gib(wm['phys_total']):8.3f} GiB total"
        )

        print(
            f"Windows commit:        "
            f"{gib(wm['commit_used']):8.3f} GiB used / "
            f"{gib(wm['commit_limit']):8.3f} GiB limit"
        )

    else:

        print(
            "System memory:         unavailable"
        )

    print("-" * 78)
    print()


# ======================================================================
# VALIDATION
# ======================================================================

def is_host_sorted_chunked(array):

    if array.size < 2:
        return True

    previous_last = None

    for start in range(
        0,
        array.size,
        VALIDATION_CHUNK_ELEMENTS
    ):

        end = min(
            start + VALIDATION_CHUNK_ELEMENTS,
            array.size
        )

        chunk = array[start:end]

        if previous_last is not None:

            if previous_last > int(chunk[0]):
                return False

        if chunk.size > 1:

            if not bool(
                np.all(
                    chunk[:-1] <= chunk[1:]
                )
            ):
                return False

        previous_last = int(chunk[-1])

    return True


def is_gpu_sorted_chunked(array):

    if array.size < 2:
        return True

    previous_last = None

    for start in range(
        0,
        array.size,
        VALIDATION_CHUNK_ELEMENTS
    ):

        end = min(
            start + VALIDATION_CHUNK_ELEMENTS,
            array.size
        )

        chunk = array[start:end]

        first_value = int(
            chunk[0].get()
        )

        if previous_last is not None:

            if previous_last > first_value:
                return False

        if chunk.size > 1:

            ok = bool(
                cp.all(
                    chunk[:-1] <= chunk[1:]
                ).get()
            )

            if not ok:
                return False

        previous_last = int(
            chunk[-1].get()
        )

    return True


def arrays_equal_chunked(a, b):

    if a.size != b.size:
        return False

    for start in range(
        0,
        a.size,
        VALIDATION_CHUNK_ELEMENTS
    ):

        end = min(
            start + VALIDATION_CHUNK_ELEMENTS,
            a.size
        )

        if not np.array_equal(
            a[start:end],
            b[start:end]
        ):
            return False

    return True


# ======================================================================
# INITIALIZATION
# ======================================================================

gpu_props = (
    cp.cuda.runtime.getDeviceProperties(0)
)

gpu_name = (
    gpu_props["name"].decode()
)

print()
print("=" * 82)
print("FLEXIBLE UINT16 SORT BENCHMARK")
print("=" * 82)
print("Python:", sys.executable)
print("GPU:", gpu_name)
print(
    "CPU sorting:",
    "ENABLED" if args.cpu_sort else "DISABLED"
)
print(
    "Copy result back:",
    "NO" if args.no_copy_back else "YES"
)
print(
    "Memory telemetry:",
    "ENABLED" if args.telemetry else "DISABLED"
)
print("Element type: uint16")
print("Bytes per element:", BYTES_PER_ELEMENT)
print("Number of tests:", len(size_specs))
print()


# ======================================================================
# CUDA WARMUP
# ======================================================================

print("Warming up CUDA...")

warm = cp.array(
    [5, 3, 4, 1, 2],
    dtype=cp.uint16
)

warm.sort()
cp.cuda.Stream.null.synchronize()

del warm

cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

print("Warmup complete.")
print()


# ======================================================================
# BENCHMARK LOOP
# ======================================================================

rng = np.random.default_rng(12345)
results = []


for test_number, spec in enumerate(
    size_specs,
    start=1
):

    requested = spec["requested"]
    byte_count = spec["bytes"]
    n = spec["elements"]

    size_gib = (
        byte_count / (1024 ** 3)
    )

    size_mib = (
        byte_count / (1024 ** 2)
    )

    a = None
    cpu = None
    gpu = None
    gpu_result = None

    cpu_time = None
    to_gpu_time = None
    gpu_sort_time = None
    from_gpu_time = None

    stage = "initialization"

    print()
    print("=" * 82)
    print(
        f"TEST {test_number}/{len(size_specs)}: "
        f"{requested}"
    )
    print("=" * 82)

    print(
        f"Actual size:           "
        f"{size_gib:12.6f} GiB"
    )

    print(
        f"                       "
        f"{size_mib:12.3f} MiB"
    )

    print(
        f"Bytes:                 "
        f"{byte_count:12,d}"
    )

    print(
        f"uint16 elements:       "
        f"{n:12,d}"
    )

    if args.telemetry:
        print_telemetry()

    free_mem, total_mem = (
        cp.cuda.runtime.memGetInfo()
    )

    print(
        f"CUDA memory reported:  "
        f"{gib(free_mem):12.3f} GiB free / "
        f"{gib(total_mem):.3f} GiB total"
    )

    if byte_count > free_mem:

        print()
        print(
            "NOTE: Dataset exceeds CUDA's currently "
            "reported free VRAM."
        )

        print(
            "      The allocation will still be attempted."
        )


    try:

        # --------------------------------------------------------------
        # Generate host data
        # --------------------------------------------------------------

        stage = "host random-data generation"

        print()
        print("Generating random data...")

        t0 = time.perf_counter()

        a = rng.integers(
            0,
            65536,
            size=n,
            dtype=DTYPE
        )

        generation_time = (
            time.perf_counter() - t0
        )

        print(
            f"Generation:             "
            f"{generation_time:12.3f} sec"
        )


        # --------------------------------------------------------------
        # Optional CPU sort
        # --------------------------------------------------------------

        if args.cpu_sort:

            stage = "CPU working-copy allocation"

            print(
                "Making CPU sort copy..."
            )

            cpu = a.copy()

            stage = "CPU NumPy sort"

            print(
                "Running NumPy CPU sort..."
            )

            t0 = time.perf_counter()

            cpu.sort()

            cpu_time = (
                time.perf_counter() - t0
            )

            print(
                f"CPU NumPy sort:         "
                f"{cpu_time:12.3f} sec"
            )


        # --------------------------------------------------------------
        # Host -> GPU
        # --------------------------------------------------------------

        stage = "host-to-GPU transfer"

        print(
            "Copying dataset to GPU..."
        )

        cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()

        gpu = cp.asarray(a)

        cp.cuda.Stream.null.synchronize()

        to_gpu_time = (
            time.perf_counter() - t0
        )

        print(
            f"H -> GPU:               "
            f"{to_gpu_time:12.3f} sec"
        )


        del a
        a = None

        gc.collect()


        free_before_sort, _ = (
            cp.cuda.runtime.memGetInfo()
        )

        print(
            f"CUDA free before sort:  "
            f"{gib(free_before_sort):12.3f} GiB"
        )


        # --------------------------------------------------------------
        # GPU sort
        # --------------------------------------------------------------

        stage = "GPU sort"

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
            )
            / 1000.0
        )

        print(
            f"GPU sort only:          "
            f"{gpu_sort_time:12.3f} sec"
        )


        # --------------------------------------------------------------
        # Result handling
        # --------------------------------------------------------------

        if args.no_copy_back:

            stage = "GPU validation"

            print(
                "Validating sorted result "
                "directly on GPU..."
            )

            valid = is_gpu_sorted_chunked(
                gpu
            )

        else:

            stage = "GPU-to-host transfer"

            print(
                "Copying sorted result back "
                "to host RAM..."
            )

            t0 = time.perf_counter()

            gpu_result = cp.asnumpy(gpu)

            cp.cuda.Stream.null.synchronize()

            from_gpu_time = (
                time.perf_counter() - t0
            )

            print(
                f"GPU -> H:               "
                f"{from_gpu_time:12.3f} sec"
            )

            stage = "result validation"

            print(
                "Validating result..."
            )

            if args.cpu_sort:

                valid = arrays_equal_chunked(
                    cpu,
                    gpu_result
                )

            else:

                valid = is_host_sorted_chunked(
                    gpu_result
                )


        # --------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------

        if from_gpu_time is None:

            gpu_total = (
                to_gpu_time
                + gpu_sort_time
            )

        else:

            gpu_total = (
                to_gpu_time
                + gpu_sort_time
                + from_gpu_time
            )

        sort_throughput = (
            size_gib / gpu_sort_time
        )

        total_throughput = (
            size_gib / gpu_total
        )

        print()
        print("-" * 82)

        if args.cpu_sort:

            print(
                f"CPU sort:               "
                f"{cpu_time:12.3f} sec"
            )

        print(
            f"H -> GPU:               "
            f"{to_gpu_time:12.3f} sec"
        )

        print(
            f"GPU sort:               "
            f"{gpu_sort_time:12.3f} sec"
        )

        if from_gpu_time is None:

            print(
                "GPU -> H:                       SKIPPED"
            )

        else:

            print(
                f"GPU -> H:               "
                f"{from_gpu_time:12.3f} sec"
            )

        print(
            f"Measured total:         "
            f"{gpu_total:12.3f} sec"
        )

        print()

        print(
            f"GPU sort throughput:    "
            f"{sort_throughput:12.3f} GiB/sec"
        )

        print(
            f"Path throughput:        "
            f"{total_throughput:12.3f} GiB/sec"
        )

        if args.cpu_sort:

            print(
                f"GPU path speedup:       "
                f"{cpu_time / gpu_total:12.2f}x"
            )

        print(
            f"Sorted correctly:       {valid}"
        )

        results.append({
            "requested": requested,
            "gib": size_gib,
            "elements": n,
            "cpu": cpu_time,
            "h2d": to_gpu_time,
            "sort": gpu_sort_time,
            "d2h": from_gpu_time,
            "total": gpu_total,
            "status":
                "OK" if valid else "BAD RESULT",
        })


    except cp.cuda.memory.OutOfMemoryError as e:

        print()
        print("*" * 82)
        print("CUDA OUT OF MEMORY")
        print(f"Failure stage: {stage}")
        print(
            f"Requested dataset: "
            f"{size_gib:.6f} GiB"
        )
        print()
        print(str(e))
        print("*" * 82)

        results.append({
            "requested": requested,
            "gib": size_gib,
            "elements": n,
            "cpu": cpu_time,
            "h2d": to_gpu_time,
            "sort": gpu_sort_time,
            "d2h": from_gpu_time,
            "total": None,
            "status":
                f"CUDA OOM during {stage}",
        })


    except cp.cuda.runtime.CUDARuntimeError as e:

        status = getattr(
            e,
            "status",
            None
        )

        print()
        print("*" * 82)
        print("CUDA RUNTIME ERROR")
        print(f"Failure stage: {stage}")
        print(f"CUDA status: {status}")
        print(str(e))
        print("*" * 82)

        results.append({
            "requested": requested,
            "gib": size_gib,
            "elements": n,
            "cpu": cpu_time,
            "h2d": to_gpu_time,
            "sort": gpu_sort_time,
            "d2h": from_gpu_time,
            "total": None,
            "status":
                f"CUDA error {status} during {stage}",
        })

        if status != 2:

            print()
            print(
                "Stopping because this was not "
                "an ordinary CUDA allocation failure."
            )

            break


    except MemoryError as e:

        print()
        print("*" * 82)
        print(
            "SYSTEM RAM / COMMIT ALLOCATION FAILURE"
        )
        print(f"Failure stage: {stage}")
        print(str(e))
        print("*" * 82)

        results.append({
            "requested": requested,
            "gib": size_gib,
            "elements": n,
            "cpu": cpu_time,
            "h2d": to_gpu_time,
            "sort": gpu_sort_time,
            "d2h": from_gpu_time,
            "total": None,
            "status":
                f"HOST OOM during {stage}",
        })


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

        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

        try:
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

        try:
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass

        gc.collect()


# ======================================================================
# FINAL TABLE
# ======================================================================

print()
print()
print("=" * 126)
print("FINAL RESULTS")
print("=" * 126)


def fmt(value):

    return (
        f"{value:.3f}s"
        if value is not None
        else "SKIP"
    )


if args.cpu_sort:

    print(
        f"{'Requested':>18} "
        f"{'GiB':>10} "
        f"{'Elements':>18} "
        f"{'CPU sort':>10} "
        f"{'H->GPU':>10} "
        f"{'GPU sort':>10} "
        f"{'GPU->H':>10} "
        f"{'Total':>10} "
        f"{'Speedup':>10} "
        f"Status"
    )

else:

    print(
        f"{'Requested':>18} "
        f"{'GiB':>10} "
        f"{'Elements':>18} "
        f"{'H->GPU':>10} "
        f"{'GPU sort':>10} "
        f"{'GPU->H':>10} "
        f"{'Total':>10} "
        f"Status"
    )


for r in results:

    if args.cpu_sort:

        speedup = (
            f"{r['cpu'] / r['total']:.2f}x"
            if (
                r["cpu"] is not None
                and r["total"] is not None
            )
            else "-"
        )

        print(
            f"{r['requested']:>18} "
            f"{r['gib']:9.6f}G "
            f"{r['elements']:18,d} "
            f"{fmt(r['cpu']):>10} "
            f"{fmt(r['h2d']):>10} "
            f"{fmt(r['sort']):>10} "
            f"{fmt(r['d2h']):>10} "
            f"{fmt(r['total']):>10} "
            f"{speedup:>10} "
            f"{r['status']}"
        )

    else:

        print(
            f"{r['requested']:>18} "
            f"{r['gib']:9.6f}G "
            f"{r['elements']:18,d} "
            f"{fmt(r['h2d']):>10} "
            f"{fmt(r['sort']):>10} "
            f"{fmt(r['d2h']):>10} "
            f"{fmt(r['total']):>10} "
            f"{r['status']}"
        )


print()
print("Benchmark complete.")

