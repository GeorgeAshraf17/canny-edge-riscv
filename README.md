# Canny Edge Detection on RISC-V (RVV 1.0)

![Language](https://img.shields.io/badge/language-C%2B%2B17-blue)
![Architecture](https://img.shields.io/badge/arch-RISC--V%20rv64gcv-orange)
![Build](https://img.shields.io/badge/build-Makefile-lightgrey)

A scalar-to-vector implementation of the **Canny edge detection** pipeline in C++17, cross-compiled for the **RISC-V RV64GCV** ISA and executed under **QEMU user-mode emulation**. The project tracks the full optimization journey: a clean scalar baseline, a compiler optimization-level sweep, and a hand-written **RVV intrinsics** pass on the two hottest pipeline stages.

> **Course:** Embedded Systems — Dr. Omar Ahmed Nasr
> **Institution:** Cairo University — Faculty of Engineering, EECE Department

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Repository Structure](#repository-structure)
- [Environment](#environment)
- [Build & Run](#build--run)
- [Testing](#testing)
- [Compiler Optimization Sweep](#compiler-optimization-sweep)
- [RVV Intrinsic Optimization](#rvv-intrinsic-optimization)
- [Benchmark Results](#benchmark-results)
- [Known Limitations](#known-limitations)
- [Makefile Targets](#makefile-targets)
- [Team](#team)
- [License](#license)

---


## Pipeline Overview

The implementation covers the gradient-computation half of Canny edge detection — Gaussian smoothing through gradient direction — with each stage individually timed:

```
Input Image (raw 8-bit grayscale)
      │
      ▼
┌─────────────────────────┐
│  1. Gaussian Blur        │  5×5 kernel — noise suppression
│     scalar + RVV         │  scalar: int32 accumulator, /273
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  2. Sobel Gradients      │  3×3 Kx, Ky → Gx, Gy  (int16, SoA)
│     scalar + RVV         │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  3. Gradient Magnitude   │  L1 (|Gx|+|Gy|) and L2 (√(Gx²+Gy²))
│                          │  normalized to [0, 255]
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  4. Gradient Direction   │  Quantized to 4 buckets: 0°, 45°, 90°, 135°
└──────────┬──────────────┘
           ▼
Output: blurred / magnitude / direction maps (.raw)
```

`src/main.cpp` drives all four stages back-to-back, times each one independently (`clock_gettime`, averaged over 100 runs), and writes `out_blurred.raw`, `out_magnitude.raw`, and `out_direction.raw`.

---

## Repository Structure

```
.
├── include/
│   ├── gaussian.h           # gaussian_blur<PixelT, AccumT, KernelT>() template
│   ├── gaussian_rvv.h        # gaussian_blur_rvv()
│   ├── sobel.h                # sobel() — Gx, Gy (int16, SoA)
│   ├── sobel_rvv.h           # sobel_rvv()
│   ├── magnitude.h           # magnitude_l1(), magnitude_l2()
│   ├── direction.h            # compute_direction() — integer cross-multiplication, no atan2
│   └── image_io.h             # load_image() / save_image() — headerless raw grayscale
│
├── src/
│   ├── gaussian.cpp
│   ├── gaussian_rvv.cpp      # RVV intrinsics — separable horizontal/vertical pass
│   ├── sobel.cpp
│   ├── sobel_rvv.cpp         # RVV intrinsics
│   ├── magnitude.cpp
│   ├── direction.cpp
│   ├── image_io.cpp
│   └── main.cpp                # Entry point: runs pipeline, times stages, saves outputs
│
├── tests/
│   ├── test_pipeline.cpp                       # GoogleTest suite (host, g++)
│   ├── test_gaussian_rvv_equivalence.cpp        # Scalar vs RVV output diff (Gaussian)
│   ├── test_sobel_rvv_equivalence.cpp           # Scalar vs RVV output diff (Sobel)
│   └── test_helpers.h                            # Synthetic image generators
│
├── docs/
│   └── benchmarks.md           # Full scalar-vs-RVV writeup, methodology, and findings
│
├── Optimization Results/
│   ├── before_after.png        # Per-stage scalar vs RVV bar chart
│   ├── pipeline_pie.png        # Pipeline time breakdown (scalar baseline)
│   ├── timing_padded.txt        # Raw timing data
│   ├── speedup_target.txt       # Stage-by-stage speedup summary
│   ├── timing_parser.py
│   ├── plot3_before_after.py
│   └── plot_pipeline_pie.py
│
├── compiler_sweep_results.txt   # -O0/-O2/-O3/-Os/-Ofast binary size + timing sweep
├── profiling_results.txt
├── Makefile
└── .gitignore
```

---

## Environment

| Component | Version / Notes |
|-----------|------------------|
| Host compiler | `g++` (system default) |
| RISC-V compiler | `riscv64-linux-gnu-g++` |
| QEMU | user-mode, `qemu-riscv64` |
| Target arch | `rv64gcv` — RV64I + M + A + F + D + C + **V (RVV 1.0)** |
| GoogleTest | installed at `~/googletest-install` |
| Python (optional) | 3.x + `numpy`, `matplotlib` — only needed for the plotting scripts in `Optimization Results/` |

### Installing the toolchain

```bash
sudo apt update
sudo apt install -y gcc-riscv64-linux-gnu g++-riscv64-linux-gnu qemu-user

# GoogleTest, built from source
git clone --depth 1 https://github.com/google/googletest
cd googletest && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/googletest-install -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && make install
```

**Verify:**
```bash
riscv64-linux-gnu-g++ --version
qemu-riscv64 --version
```

---

## Build & Run

### 1. Clone

```bash
git clone https://github.com/GeorgeAshraf17/canny-edge-riscv.git
cd canny-edge-riscv
```

### 2. Provide a test image

The pipeline expects a headerless raw 8-bit grayscale file (`width × height` bytes). Place one at `test_image.raw` in the project root, or point the binary at any file of your own with matching dimensions.

### 3. Build

```bash
# Scalar baseline (Gaussian + Sobel both scalar)
make canny_rv

# RVV build (Gaussian vectorized; see Known Limitations re: Sobel)
make canny_rvv
```

### 4. Run under QEMU

```bash
make run        # runs build_rv/canny_rv
make run_rvv    # runs build_rv/canny_rvv
```

Both targets invoke:
```bash
qemu-riscv64 -cpu rv64,v=true,vlen=256 -L /usr/riscv64-linux-gnu \
    ./build_rv/<binary> test_image.raw 256 256
```

Output is a per-stage timing table printed to stdout, plus three `.raw` files written to the working directory.

---

## Testing

Unit tests run natively on the host (no QEMU needed) via GoogleTest:

```bash
make test
```

This covers Gaussian symmetry/impulse-response, Sobel zero-gradient-on-uniform-input and directional-edge cases, L1/L2 magnitude sanity checks, and direction quantization — see `tests/test_pipeline.cpp`.

RVV-vs-scalar equivalence tests exist separately (`tests/test_gaussian_rvv_equivalence.cpp`, `tests/test_sobel_rvv_equivalence.cpp`) but are **not** wired into the `make test` target; see [Known Limitations](#known-limitations) for what running them directly reveals.

---

## Compiler Optimization Sweep

```bash
make sweep       # builds canny_O0, canny_O2, canny_O3, canny_Os, canny_Ofast
make run_sweep   # runs each under QEMU and prints timing for every level
```

Results (binary size per level, plus full timing) are recorded in `compiler_sweep_results.txt` and `profiling_results.txt`.

---

## RVV Intrinsic Optimization

Two stages were hand-vectorized with RVV 1.0 intrinsics (`<riscv_vector.h>`), gated behind preprocessor flags so the scalar build is unaffected:

- **`gaussian_blur_rvv()`** (`src/gaussian_rvv.cpp`) — separable 5×5 filter: a vectorized horizontal pass (`e8m1`, widened to 16-bit) followed by a vectorized vertical pass (`e16m2`, widened to 32-bit for accumulation).
- **`sobel_rvv()`** (`src/sobel_rvv.cpp`) — vectorized 3×3 gradient computation.

```bash
make canny_rvv   # builds with -DUSE_RVV_GAUSSIAN
make run_rvv     # runs it under QEMU, vlen=256
```

> **Note:** the default `canny_rvv` target only enables `USE_RVV_GAUSSIAN` — see [Known Limitations](#known-limitations) for why Sobel still runs scalar in this build, and how to force the full-RVV path.

---

## Benchmark Results

Full methodology, raw tables for 256×256 / 512×512 / 1024×1024, and analysis are in [`docs/benchmarks.md`](docs/benchmarks.md). Summary at 256×256, `vlen=256`:

| Stage | Scalar (ms) | RVV: Gaussian only (ms) | Speedup |
|-------|------------:|--------------------------:|--------:|
| Gaussian blur | 12.74 | 3.90 | **3.27×** |
| Sobel | 4.20 | 1.06† | — |
| Magnitude (L1) | 0.61 | 0.67 | 0.91× |
| Direction | 0.97 | 0.85 | 1.14× |
| **Total** | **18.52** | **6.47** | **2.86×** |

**Headline takeaway:** Gaussian blur is the only stage with a clean, structural win from RVV (separable filtering + vectorization). Magnitude is *slower* under `USE_RVV_GAUSSIAN` because it switches to a two-pass (max-reduction, then normalize) implementation, trading one pass for two. Sobel's full-RVV numbers (see `docs/benchmarks.md`) are slower than scalar under QEMU emulation — plausibly because the kernel does too little arithmetic per pixel for vectorization overhead to pay off in an emulated environment; this would need re-measuring on real hardware to draw silicon-level conclusions.

---

## Known Limitations

These are documented (not hidden) because they came directly out of running the equivalence tests in `tests/`:

- **The default `canny_rvv` build target does not actually vectorize Sobel.** The Makefile compiles `sobel_rvv.cpp` into the binary but never defines `USE_RVV_SOBEL`, so `main.cpp` falls through to the scalar `sobel()` at runtime. The "Sobel" row in the table above is the scalar kernel, not the RVV one.
- **`gaussian_blur_rvv` is not bit-exact with the scalar reference.** At 256×256, 64,499 of 65,536 pixels differ. The vertical pass approximates `/273` as `(sum * 240) >> 16`, which is close but introduces rounding error beyond what fixed-point approximation alone would explain.
- **`sobel_rvv`'s vertical gradient (`gy`) comes out sign-flipped** relative to the scalar version on every mismatching pixel (e.g. scalar `gy=8` vs. RVV `gy=-8`), which propagates through direction and the final magnitude map (64,968 / 65,536 pixels differ end-to-end at 256×256).

See `docs/benchmarks.md` for the full investigation, including how to reproduce the divergence with `-DUSE_RVV_SOBEL`.

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make test` | Build and run the GoogleTest suite on host |
| `make canny_rv` | Cross-compile the scalar RISC-V binary |
| `make canny_rvv` | Cross-compile the RVV binary (`-DUSE_RVV_GAUSSIAN`) |
| `make run` | Run `canny_rv` under QEMU (`vlen=256`) |
| `make run_rvv` | Run `canny_rvv` under QEMU (`vlen=256`) |
| `make sweep` | Build at `-O0`, `-O2`, `-O3`, `-Os`, `-Ofast`; print binary sizes |
| `make run_sweep` | Run every sweep build under QEMU and print timings |
| `make clean` | Remove `build_host/` and `build_rv/` |

## Team

| Name | ID |
|------|----|
| George Ashraf "Leader" | 91240244 |
| Sara Rezk | 91240340 |
| Ramez Reda | 91240286 |
| Refaat Shokry | 91240293 |
| Ahmed Emad | 91240102 |

---
### Runtime arguments

All binaries take the same three positional arguments:

```bash
<binary> <input.raw> <width> <height>
```

---

## License

his project is licensed under the [MIT License](LICENSE).
