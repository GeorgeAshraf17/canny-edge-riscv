# Performance Benchmarks: Scalar vs. RVV Intrinsics

This document compares the scalar (baseline) and RVV-intrinsic-optimized implementations of the Canny edge detection pipeline, measured on RISC-V (`rv64gcv`) under QEMU user-mode emulation with `vlen=256`.

## Methodology

- **Hardware/emulation:** `qemu-riscv64 -cpu rv64,v=true,vlen=256`
- **Compiler:** `riscv64-linux-gnu-g++`, `-march=rv64gcv`
- **Build flags:** scalar baseline at `-O2`; RVV builds at `-O3` (matching the project `Makefile` targets `canny_rv` and `canny_rvv`)
- **Timing:** each pipeline stage is timed independently with `clock_gettime(CLOCK_MONOTONIC, ...)`, averaged over 100 runs per image, as implemented in `src/main.cpp`
- **Test images:** synthetic random 8-bit grayscale images at 256×256, 512×512, and 1024×1024
- **Variants compared:**
  - **Scalar** — `make canny_rv` (all stages scalar)
  - **RVV (Gaussian only)** — `make canny_rvv` (the project's current default RVV target: vectorizes Gaussian blur only; Sobel stays scalar)
  - **RVV (Gaussian + Sobel)** — a full-RVV build (`-DUSE_RVV_GAUSSIAN -DUSE_RVV_SOBEL`) that also enables `sobel_rvv.cpp`, included here to show the effect of vectorizing every stage

Raw numbers below are reproducible with:
```bash
make canny_rv && make canny_rvv
qemu-riscv64 -cpu rv64,v=true,vlen=256 -L /usr/riscv64-linux-gnu ./build_rv/canny_rv  <img>.raw <w> <h>
qemu-riscv64 -cpu rv64,v=true,vlen=256 -L /usr/riscv64-linux-gnu ./build_rv/canny_rvv <img>.raw <w> <h>
```

## Results — 256×256

| Stage          | Scalar (ms) | RVV: Gaussian only (ms) | Speedup | RVV: Gaussian+Sobel (ms) | Speedup |
|----------------|------------:|-------------------------:|--------:|---------------------------:|--------:|
| Gaussian blur  |      10.04  |                     6.14 |  1.64×  |                       6.21 |  1.62×  |
| Sobel          |       3.47  |                     0.79 |  4.39× *|                       4.55 |  0.76×  |
| Magnitude (L1) |       0.63  |                     2.65 |  0.24×  |                       2.66 |  0.24×  |
| Direction      |       0.79  |                     0.81 |  0.97×  |                       0.79 |  1.00×  |
| **Total**      |  **15.02**  |                **10.39** |**1.45×**|                  **14.32** |**1.05×**|

\* Sobel shows as "fast" in the Gaussian-only column only because it is still running the scalar implementation there — see note below.

## Results — 512×512

| Stage          | Scalar (ms) | RVV: Gaussian only (ms) | RVV: Gaussian+Sobel (ms) |
|----------------|------------:|-------------------------:|---------------------------:|
| Gaussian blur  |      40.80  |                    24.62 |                      25.04 |
| Sobel          |      13.54  |                     3.28 |                      18.05 |
| Magnitude (L1) |       2.39  |                    10.79 |                      10.85 |
| Direction      |       3.17  |                     3.16 |                       3.20 |
| **Total**      |  **59.90**  |                **41.86** |                  **57.14** |

## Results — 1024×1024

| Stage          | Scalar (ms) | RVV: Gaussian only (ms) | RVV: Gaussian+Sobel (ms) |
|----------------|------------:|-------------------------:|---------------------------:|
| Gaussian blur  |     162.51  |                    96.43 |                      98.14 |
| Sobel          |      56.33  |                    12.64 |                      70.64 |
| Magnitude (L1) |       9.88  |                    42.16 |                      41.86 |
| Direction      |      13.18  |                    12.72 |                      12.87 |
| **Total**      | **241.90**  |               **163.94** |                 **223.51** |

Timings scale linearly with pixel count across all three sizes, as expected for a fixed-radius filter pipeline.

## Observations

**Gaussian blur is consistently ~1.6× faster with RVV.** The separable 5×5 filter (`gaussian_rvv.cpp`) splits the convolution into a vectorized horizontal pass (`e8m1`, widened to 16-bit) and a vectorized vertical pass (`e16m2`, widened to 32-bit for the accumulation), which is the right structural change — separable filtering is O(n) per pixel instead of O(n²) — and the intrinsic version captures that gain on top of vectorization.

**The current `canny_rvv` build target does not vectorize Sobel.** `Makefile`'s `canny_rvv` recipe compiles `sobel_rvv.cpp` but never defines `USE_RVV_SOBEL`, so `main.cpp`'s `#ifdef USE_RVV_SOBEL` falls through to the scalar `sobel()` at runtime. This is why Sobel appears "fast" in the default RVV build — it's simply running the unmodified scalar code. Enabling `USE_RVV_SOBEL` (full-RVV column above) shows the vectorized Sobel kernel is actually **slower than scalar** at every image size tested (e.g. 18.0 ms vs. 13.5 ms at 512×512). This is plausible under QEMU emulation: the 3×3 Sobel kernel does very little arithmetic per pixel, so the fixed overhead of widening/narrowing conversions and emulated vector instruction decode can outweigh the benefit of processing multiple pixels per instruction. This would be worth re-measuring on real RVV hardware (or a cycle-accurate simulator) before drawing conclusions about silicon performance.

**Magnitude is also slower in the RVV build, but for a different, structural reason.** `magnitude_l1()` switches to a vectorized max-reduction + normalization implementation whenever `USE_RVV_GAUSSIAN` is defined (see `src/magnitude.cpp`), even though that's a separate optimization from Gaussian vectorization. That implementation requires two full passes over the image (one to find the global max gradient magnitude, one to normalize against it) versus one pass in the scalar version, which is the likely source of the ~4× slowdown seen here, independent of any RVV-vs-scalar instruction cost.

**Correctness: the RVV outputs are not bit-exact with scalar.** The repo's own equivalence tests (`tests/test_gaussian_rvv_equivalence.cpp`, `tests/test_sobel_rvv_equivalence.cpp`) are not currently wired into `make test`, and running them directly surfaces two issues:
- `gaussian_blur_rvv` diverges from the scalar reference on the majority of pixels (64,499 / 65,536 at 256×256). The vertical pass rescales by `* 240 >> 16` instead of dividing by 273 directly, which is a fixed-point approximation of `1/273` — close, but not identical, and the divergence is large enough that it likely also includes a logic difference beyond rounding.
- `sobel_rvv`'s vertical gradient (`gy`) comes out **sign-flipped** relative to scalar on every mismatching pixel (e.g. `gy_scalar=8, gy_rvv=-8`), which propagates into the direction stage as well as the final magnitude map (64,968 / 65,536 pixels differ end-to-end).

These are correctness bugs, not just numerical noise, and should be fixed before relying on the RVV path for production output — see [Recommendations](#recommendations).

## Recommendations

1. **Fix `sobel_rvv`'s `gy` sign.** Compare the vertical-kernel term ordering against the scalar reference (`sobel.cpp`); the vectorized version appears to compute `bottom_row - top_row` where the scalar computes `top_row - bottom_row` (or vice versa).
2. **Fix or re-derive the Gaussian RVV normalization.** Either use a constant that exactly reproduces `/ 273`, or accept the approximation deliberately and update the equivalence test to check a small tolerance instead of bit-exact equality.
3. **Wire `test_gaussian_rvv_equivalence` and `test_sobel_rvv_equivalence` into `make test`** so future RVV changes can't silently break numerical correctness.
4. **Decide whether to ship Sobel RVV at all.** Given it's slower than scalar in this emulated environment, either leave `canny_rvv` as Gaussian-only (current behavior, just needs the bugs above fixed) or re-benchmark on real hardware before enabling `USE_RVV_SOBEL` by default.
5. **Re-run this benchmark on real RVV-capable hardware** (e.g. a SiFive board or QEMU with `-icount`/cycle modeling) since QEMU user-mode timing reflects host-CPU emulation cost, not target silicon cycles, and instruction-level overheads (like Sobel's) may look very different on real vector units.
