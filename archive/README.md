# Archived benchmark drafts

These files preserve earlier alpha stages of the GPU sorting benchmark.

The archive files are reconstructed from the development history and benchmark behavior discussed during this project. They preserve the design progression but are not guaranteed to be byte-for-byte copies of every intermediate local file.

Current maintained program: `../sort_scaling_flexible.py`

Archived stages:

- `01_sort_benchmark_512mib.py` - fixed 512 MiB CPU versus GPU benchmark.
- `02_sort_scaling_large.py` - GPU scaling tests from 4.0 through 8.5 GiB.
- `03_sort_scaling_flexible_initial.py` - flexible size parser and exact element-count boundary tests.
- `04_sort_scaling_telemetry_alpha.py` - memory telemetry and no-copy-back capacity testing before the later help and environment-detection improvements.

The exact 2^31-element tests showed no timing discontinuity at that boundary. Later tests also demonstrated successful sorting above nominal free VRAM under Windows WDDM, which motivated the telemetry work in the maintained version.
