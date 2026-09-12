# GPU Sort Benchmark

A Windows/CUDA benchmark project exploring how to sort very large datasets efficiently on a desktop GPU — and when the best "sort" is not a comparison sort at all.

The project began by characterizing CuPy sorting performance on an **NVIDIA GeForce RTX 3080 (10 GB)** under Windows 11/WDDM. It now extends that work toward a **32 GiB external-file sorting benchmark** for values constrained to the integer domain **0–65,535**, stored as either `uint16` or exact-integer `float64` values.

> **Key result so far:** on the test system, a CuPy `uint16` chunk of approximately **3.870 GiB** gave the best practical full round-trip behavior among the sizes tested. For the constrained 0–65,535 domain, however, a GPU histogram/counting-sort design should avoid comparison sorting and general merging entirely.

For the full benchmark history, WDDM observations, algorithm notes, validation plan, and 32 GiB external-sort design, see **[MORE_DETAILS_README.md](MORE_DETAILS_README.md)**.

---

## What this repository investigates

The project separates several costs that are often collapsed into a single GPU benchmark number:

- host-to-GPU transfer
- GPU sorting time
- GPU-to-host transfer
- VRAM/workspace pressure
- Windows WDDM shared-memory behavior
- chunk-size effects
- external-file I/O
- merge/reconstruction cost
- independent validation

That distinction matters. On the RTX 3080 test system, the GPU sort kernel can be far faster than the complete data path around it.

---

## Test platform

The principal development and benchmark machine is:

```text
CPU:        Intel Core i9-11900K
Cores:      8 cores / 16 threads
Memory:     48 GB DDR4-3200
GPU:        NVIDIA GeForce RTX 3080
VRAM:       10 GB
OS:         Windows 11
CUDA:       13.x
Python:     3.14.x
GPU stack:  CuPy
```

Windows uses **WDDM**, so workloads can spill into shared system memory rather than failing exactly at nominal VRAM capacity. That can keep very large workloads alive, but often at a substantial performance cost.

---

## Current benchmark: flexible CuPy GPU sort

The current working benchmark is:

```text
sort_scaling_flexible.py
```

It supports:

- flexible sizes such as `4GiB`, `1024MiB`, exact bytes, and element counts
- optional CPU sorting
- optional CUDA/Windows memory telemetry
- optional GPU-to-host copy-back
- on-GPU validation when copy-back is disabled
- controlled testing beyond reported CUDA free VRAM
- graceful handling of allocation failures

Earlier experimental versions are retained in `archive/`.

### Example

```powershell
..\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py --size 3.870GiB
```

With telemetry:

```powershell
..\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py --telemetry --size 3.870GiB
```

Sort and validate on the GPU without copying the result back:

```powershell
..\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py --no-copy-back --size 3.870GiB
```

---

## Chunk-size result

A precision sweep around 4 GiB found a practical sweet spot near **3.87 GiB** for CuPy `uint16` sorting on this machine.

Final full-round-trip comparison:

| Chunk size | Mean H→GPU | Mean GPU sort | Mean GPU→H | Mean total |
|---:|---:|---:|---:|---:|
| **3.870 GiB** | **~0.956 s** | **~0.182 s** | **~0.790 s** | **~1.927 s** |
| 3.890 GiB | ~0.982 s | ~0.182 s | ~0.801 s | ~1.965 s |
| 3.875 GiB | ~0.991 s | ~0.182 s | ~0.803 s | ~1.977 s |

The important observation is that the **sort time itself was essentially identical** across the finalists. The difference came primarily from transfer/residency behavior.

This is a machine-specific operating point, not a universal CUDA constant.

---

# 32 GiB external sorting experiment

The next stage of the project treats the input as a file supplied by another party rather than data generated inside the benchmark.

Proposed workload:

```text
Input size:     32 GiB
Input order:    unsorted/random
Logical values: exact integers 0 through 65,535
Representations: uint16 and float64
Output:         fully sorted file
Validation:     independent correctness checks
```

A 32 GiB file contains:

```text
uint16:  17,179,869,184 values
float64:  4,294,967,296 values
```

The two representations carry the same logical keyspace but very different numbers of elements per byte.

---

## Algorithm 1: GPU histogram / counting sort

Because there are only **65,536 possible keys**, the entire dataset can be represented exactly by 65,536 counters.

Using `uint64` counters:

```text
65,536 × 8 bytes = 512 KiB
```

The planned external-file pipeline is:

```text
32 GiB input file
        │
        ▼
read bounded chunk
        │
        ▼
copy to GPU
        │
        ▼
validate + histogram
        │
        ▼
accumulate 65,536 uint64 counters
        │
       ...
        │
        ▼
reconstruct values in ascending order
        │
        ▼
32 GiB sorted output file
```

No per-chunk comparison sort is required.

No k-way merge is required.

Once all chunks have contributed to the histogram, the histogram is an exact representation of the sorted multiset.

For the `float64` case, every input value is expected to be an exact integer-valued double such as `12345.0`; it can therefore be validated and mapped losslessly to the same 16-bit logical key.

---

## Algorithm 2: conventional chunk sort + external merge

For comparison, the project also plans a conventional external sorting path:

```text
input file
    │
    ▼
~3.870 GiB chunk
    │
    ▼
GPU sort
    │
    ▼
write sorted run
    │
   ...
    │
    ▼
merge sorted runs
    │
    ▼
final sorted file
```

The merge should use compiled/native or GPU-oriented code rather than a Python element-by-element loop, so the comparison remains meaningful.

The expected minimum storage traffic highlights the structural difference:

```text
Histogram path:
    read input       32 GiB
    write output     32 GiB
    -----------------------
    minimum          64 GiB

Chunk-sort + merge:
    read input       32 GiB
    write runs       32 GiB
    read runs        32 GiB
    write output     32 GiB
    -----------------------
    minimum         128 GiB
```

---

## Radix connection

The histogram approach can also be viewed as the terminal form of radix partitioning for this specific problem.

A 16-bit key could be partitioned in multiple radix passes, but once each possible key has its own bucket there are exactly 65,536 buckets — effectively the histogram itself.

Because equal values do not need to retain individual identity, counting them is sufficient.

---

## Validation strategy

For an externally supplied 32 GiB `float64` file, the planned implementation should verify at minimum:

```text
file size == 32 GiB
element count == 4,294,967,296
all values are finite
all values are in [0, 65,535]
all values are exact integers
sum(histogram) == element count
```

The output can then be independently checked for:

```text
correct byte size
correct element count
monotonic nondecreasing order
same histogram as input
```

Independent validation should be reported separately from the transformation timing.

---

## Why this project is interesting

The goal is not merely to demonstrate that GPUs are fast. It is to examine the engineering decisions that determine whether GPU acceleration actually helps end to end:

- how much does PCIe dominate the workload?
- what does WDDM do when VRAM is oversubscribed?
- where is the useful chunk-size range?
- when is a constrained keyspace more important than the nominal datatype?
- can counting/histogramming eliminate the merge entirely?
- when does SSD throughput become the real bottleneck?

In the constrained 0–65,535 case, the likely fastest solution is interesting precisely because it may never invoke a conventional sorting algorithm.

---

## Repository layout

```text
gpu-sort-benchmark/
│
├── sort_scaling_flexible.py
├── README.md
├── MORE_DETAILS_README.md
├── results/
│   └── benchmark output files
└── archive/
    ├── earlier benchmark versions
    └── README.md
```

---

## Status

The flexible GPU sorting benchmark and chunk-size characterization are implemented.

The 32 GiB external histogram/counting-sort and conventional external-merge comparison are the next implementation phase described by this repository.

---

## License

No license has been selected yet. Add an explicit license before relying on this repository for public redistribution or reuse.
