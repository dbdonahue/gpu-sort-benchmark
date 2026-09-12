# GPU Sort Benchmark — More Details

[← Back to the main README](README.md)

This document contains the longer engineering notes behind the public project overview: benchmark methodology, measured chunk behavior, WDDM observations, the planned 32 GiB external-file experiment, and the reasoning behind the histogram/counting-sort design.

GPU-accelerated sorting experiments for very large, constrained-keyspace datasets on Windows 11 using Python, CuPy, CUDA, and an NVIDIA RTX 3080.

This repository explores two related questions:

1. **How fast can a large array be sorted on the GPU when it must be processed in chunks?**
2. **How much faster can the problem become when the data domain is known in advance?**

The motivating workload is a large dataset whose values are restricted to the integer range **0 through 65,535**. The values may be stored as either `uint16` or `float64`. In the `float64` case, the values are still exact integers represented as doubles (for example, `12345.0`).

For that constrained domain, a conventional comparison sort is not necessarily the best algorithm. A histogram/counting-sort approach can represent the entire dataset exactly with only 65,536 counters.

---

## Current Status

The repository currently contains benchmark code used to characterize GPU sorting behavior, chunk sizes, PCIe transfer cost, Windows WDDM memory behavior, and practical VRAM limits.

The current benchmark script is:

```text
sort_scaling_flexible.py
```

It supports:

- arbitrary input sizes such as `4GiB`, `1024MiB`, exact bytes, and exact element counts
- optional CPU sorting
- optional GPU/Windows memory telemetry
- optional copy-back to host memory
- validation of sorted output
- testing beyond nominal CUDA free VRAM
- graceful reporting of allocation failures
- operation from a dedicated Python/CuPy virtual environment

Earlier experimental versions are preserved under:

```text
archive/
```

---

## Test System

The principal benchmark system used during development is:

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

Because Windows uses WDDM, CUDA workloads may use shared system memory when dedicated VRAM is insufficient. This can allow allocations larger than nominal free VRAM, but performance can fall sharply once residency and paging begin.

---

# Part I: GPU Chunk-Sort Benchmark

## Why Chunking Is Necessary

Large datasets can exceed practical VRAM capacity even when the raw input appears to fit.

GPU sorting algorithms usually require temporary workspace in addition to the input array. On Windows, WDDM residency behavior can also introduce substantial shared-memory use.

For this system, a precision sweep found that a chunk size near **3.87 GiB** gave the best practical full round-trip behavior for `uint16` sorting with CuPy.

The current production candidate is:

```text
3.870 GiB
≈ 4,155,380,858 bytes
≈ 2,077,690,429 uint16 values
```

The exact optimum is not treated as a universal hardware constant. It is an empirically useful operating point for this particular GPU, driver, CUDA stack, Windows memory manager, and benchmark.

---

## Representative Results

For approximately 3.87 GiB chunks, a typical complete path is approximately:

```text
Host -> GPU     ~0.9–1.0 s
GPU sort        ~0.18 s
GPU -> Host     ~0.8 s
------------------------
Total           ~1.9 s
```

The GPU sort itself is extremely fast relative to transfer time. Once chunk sizes become large, **PCIe transfer and memory residency dominate the end-to-end cost**.

A final comparison among the strongest candidates showed approximately:

| Chunk size | Mean H->GPU | Mean GPU sort | Mean GPU->H | Mean total |
|---:|---:|---:|---:|---:|
| 3.870 GiB | ~0.956 s | ~0.182 s | ~0.790 s | ~1.927 s |
| 3.890 GiB | ~0.982 s | ~0.182 s | ~0.801 s | ~1.965 s |
| 3.875 GiB | ~0.991 s | ~0.182 s | ~0.803 s | ~1.977 s |

These numbers should be treated as machine-specific measurements, not general CUDA performance claims.

---

## Example Usage

Run the benchmark using the dedicated virtual environment:

```powershell
..\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py --size 3.870GiB
```

Run with telemetry:

```powershell
..\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py --telemetry --size 3.870GiB
```

Run without copying the sorted array back to host RAM:

```powershell
..\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py --no-copy-back --size 3.870GiB
```

Run several sizes:

```powershell
..\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py `
    --size 3.870GiB 3.875GiB 3.890GiB
```

Display help:

```powershell
..\gpu-sort-test\Scripts\python.exe .\sort_scaling_flexible.py --help
```

---

# Part II: 32-GiB External Sorting Experiment

The next stage of the project is a 32-GiB external-file sorting benchmark.

The proposed test case is:

```text
Input size:     32 GiB
Input type:     float64 or uint16
Logical values: integers 0 through 65,535
Input order:    random / unsorted
Output:         fully sorted file
Validation:     exact value preservation and monotonic ordering
```

For `float64`, 32 GiB contains:

```text
4,294,967,296 values
```

For `uint16`, 32 GiB contains:

```text
17,179,869,184 values
```

---

# Algorithm A: GPU Histogram / Counting Sort

For this constrained keyspace, this is expected to be the fastest algorithm.

There are only:

```text
65,536 possible values
```

so the complete multiset can be represented by:

```text
65,536 x uint64 counters
```

which requires only:

```text
512 KiB
```

of histogram storage.

The algorithm is:

```text
32-GiB input file
        |
        v
read bounded chunk
        |
        v
copy chunk to GPU
        |
        v
validate values
        |
        v
GPU histogram
        |
        v
accumulate into global 65,536-bin histogram
        |
        v
discard input chunk
        |
       ...
        |
        v
reconstruct sorted output sequentially
        |
        v
32-GiB sorted output file
```

No per-chunk comparison sort is required.

No general merge is required.

The histogram is already an exact representation of the sorted result.

---

## Why This Is Equivalent to the Terminal Form of Radix Partitioning

A 16-bit logical key can be radix-partitioned by bits.

For example:

```text
high 8 bits -> 256 buckets
low 8 bits  -> 256 sub-buckets
```

After enough radix passes, every possible 16-bit value has its own bucket.

At that point, the structure is equivalent to a 65,536-bin histogram.

Because equal values are interchangeable for this workload, there is no need to preserve individual element identity inside a bucket. Counting each key is sufficient.

For this specific domain, **counting sort is effectively the simplest final form of radix partitioning**.

---

# Algorithm B: Conventional GPU Chunk Sort + Merge

A second implementation is intended as a comparison baseline.

The planned architecture is:

```text
32-GiB input file
        |
        v
read ~3.870-GiB chunk
        |
        v
GPU sort
        |
        v
write sorted temporary run
        |
       ...
        |
        v
multiple sorted runs
        |
        v
k-way / tiled merge
        |
        v
32-GiB sorted output file
```

This is a conventional external sorting design.

The purpose is not to intentionally cripple the comparison. The merge should be implemented using compiled native code or a GPU-oriented tiled/merge-path approach rather than a Python element-by-element loop.

---

## Expected Storage Traffic

For the histogram approach:

```text
read input        32 GiB
write output      32 GiB
------------------------
minimum traffic   64 GiB
```

For conventional chunk sort plus external merge:

```text
read input            32 GiB
write sorted runs     32 GiB
read sorted runs      32 GiB
write final output    32 GiB
----------------------------
minimum traffic      128 GiB
```

This difference is one of the primary reasons the histogram approach is expected to win when the keyspace is known and small.

---

# float64 vs uint16

The repository intentionally compares two physical representations of the same logical values.

## `uint16`

```text
2 bytes per value
range: 0 through 65,535
```

This is the natural representation for the problem.

## `float64`

```text
8 bytes per value
logical values still restricted to exact integers 0.0 through 65535.0
```

The `float64` experiment measures the cost of carrying the same logical information in a much wider representation.

For histogramming, a `float64` value can be validated and mapped losslessly to a 16-bit logical key before incrementing its bin.

---

# File-Oriented Operation

The external-sort version is designed around a file supplied by another party.

The implementation should not require the entire 32-GiB input to reside in RAM.

A bounded-buffer pipeline is preferable:

```text
Disk -> Buffer A -> GPU histogram
Disk -> Buffer B -> GPU histogram
```

Double buffering may allow disk I/O and GPU work to overlap.

The optimal histogram chunk size may differ substantially from the 3.870-GiB chunk-sort optimum because histogramming requires far less GPU workspace.

---

# Validation

The file-oriented implementation should validate the input and output independently.

For a 32-GiB `float64` input file, useful checks include:

```text
input file size == 32 GiB
element count == 4,294,967,296
all values are finite
all values are >= 0
all values are <= 65,535
all values are exact integers
sum(histogram) == element count
```

The output is sorted by construction when generated directly from the histogram.

Additional validation may include:

```text
output file size == 32 GiB
output element count == input element count
output values are monotonically nondecreasing
output histogram == input histogram
```

Independent output validation should be timed separately from the transformation itself.

---

# Benchmarking Principles

This project tries to distinguish several different costs rather than reporting only one headline number:

```text
input generation or file read
host -> GPU transfer
GPU computation
GPU -> host transfer
temporary-file writes
merge cost
output reconstruction
output-file write
validation
```

Where possible, results should include both:

- **kernel / algorithm time**
- **true end-to-end elapsed time**

This avoids presenting a fast GPU kernel as though PCIe, disk I/O, allocation, WDDM residency, and reconstruction were free.

---

# Windows / WDDM Notes

On Windows, the RTX 3080 is managed through WDDM rather than a dedicated compute-only driver model.

Observed behavior includes:

- successful CUDA workloads larger than nominal free VRAM
- significant use of Windows shared GPU memory
- sharp performance degradation once oversubscription becomes large
- allocation failures during large GPU sorts
- pinned-memory allocation failures at extreme transfer sizes

These effects are part of the benchmark rather than incidental noise. They materially affect real end-to-end performance on a Windows desktop system.

---

# Repository Layout

Current and planned organization:

```text
gpu-sort-benchmark/
|
|-- sort_scaling_flexible.py
|-- README.md
|-- MORE_DETAILS_README.md
|-- LICENSE
|-- results/
|   `-- benchmark output files
|
`-- archive/
    |-- earlier benchmark versions
    `-- README.md
```

Additional 32-GiB external-sort programs may be added as the histogram and merge implementations are developed.

---

# Goals

This project is intended to answer practical engineering questions rather than demonstrate one predetermined result:

- Where is the practical GPU chunk-size sweet spot?
- How much of GPU sorting time is actually PCIe transfer time?
- What happens when Windows WDDM oversubscribes VRAM?
- How does `uint16` compare with `float64` for the same logical keyspace?
- How much faster is histogram/counting sort than conventional chunk sort plus merge?
- Is CPU reconstruction faster than GPU reconstruction when the destination is a host-side file?
- At what point does storage bandwidth become the limiting factor?

---

# Caveats

Results in this repository are specific to the tested hardware and software environment.

Performance may change with:

- GPU model
- VRAM capacity
- PCIe generation and lane width
- CUDA version
- CuPy version
- NVIDIA driver version
- Windows WDDM behavior
- system RAM
- storage device
- power state
- background GPU activity

The benchmark should therefore be treated as an engineering characterization of a particular system, not as a universal ranking of algorithms or hardware.

---

# License

This project is licensed under the **BSD Zero Clause License (0BSD)**. See [LICENSE](LICENSE).

0BSD allows use, copying, modification, and redistribution for any purpose, with or without fee, and does not impose a downstream attribution requirement.
