#include "magnitude.h"
#include <cmath>
#include <cstdint>
#include <algorithm>

#ifdef USE_RVV_GAUSSIAN
#include <riscv_vector.h>

// Hand-written RVV implementation — kept for reference and for real RVV
// hardware (where vector dispatch genuinely is cheap relative to scalar).
// On QEMU rv64gcv emulation this is empirically ~3-4x SLOWER than the plain
// scalar magnitude_l1() below for this workload size, so magnitude_l1() does
// NOT call this function. See the note above magnitude_l1() for details.
void compute_magnitude_rvv(const int16_t* gx, const int16_t* gy, uint8_t* magnitude, int width, int height) {
    int total_pixels = width * height;

    // ── PASS 1 (scalar): find global max ──────────────────────────────────────
    // Under QEMU the dominant cost for vector code is per-instruction dispatch
    // overhead (~14 ns each), not the specific operation.  A plain scalar loop
    // here costs zero vector dispatches and is JIT-compiled very efficiently,
    // so this pass is essentially free compared to any vector alternative.
    int16_t max_val = 0;
    for (int i = 0; i < total_pixels; i++) {
        int16_t ag = gx[i] < 0 ? (int16_t)(-gx[i]) : gx[i];
        int16_t ab = gy[i] < 0 ? (int16_t)(-gy[i]) : gy[i];
        int16_t m  = ag + ab;
        if (m > max_val) max_val = m;
    }
    if (max_val == 0) max_val = 1;

    // Fixed-point reciprocal: inv_scale = floor((255 << 16) / max_val)
    // Result: (mag * inv_scale) >> 16  ≈  (mag * 255) / max_val
    //
    // Overflow safety: mag ≤ max_val, so product ≤ max_val × (255<<16 / max_val)
    //                  = 255 × 65536 = 16,711,680  <<  INT32_MAX.
    //
    // Replaces vdiv_vx_i32m4 (integer vector divide ≈ 30–100× slower than vmul
    // on hardware; under QEMU it also dispatches as a costly emulated operation).
    int32_t inv_scale = (int32_t)((255u << 16) / (uint32_t)max_val);

    // ── PASS 2 (RVV): compute L1 magnitude + normalize in a single pass ───────
    // No temp buffer → no heap allocation in the hot loop (called 100× in bench).
    //
    // LMUL=4 here, not LMUL=2: under QEMU each vector instruction pays a fixed
    // per-dispatch emulation cost regardless of how many lanes it processes, so
    // the loop trip count (not the element count) is what dominates runtime.
    // e16m2 only gives VLMAX = 2*VLEN/16 = 32 elements/iter at vlen=256, forcing
    // 2048 loop iterations × ~11 vector ops ≈ 22k dispatches for a 256×256
    // image. e16m4 doubles VLMAX to 64 elements/iter (1024 iterations), roughly
    // halving total dispatches for the same work. m4 is also the ceiling here:
    // the i16→i32 widen below doubles LMUL, and m8 is the max legal group, so
    // m4 is as large as we can go before the widen step would overflow it.
    int x = 0;
    while (x < total_pixels) {
        size_t vl = __riscv_vsetvl_e16m4(total_pixels - x);

        vint16m4_t v_gx     = __riscv_vle16_v_i16m4(&gx[x], vl);
        vint16m4_t v_gy     = __riscv_vle16_v_i16m4(&gy[x], vl);

        // |gx| + |gy|
        vint16m4_t v_abs_gx = __riscv_vmax_vv_i16m4(v_gx, __riscv_vneg_v_i16m4(v_gx, vl), vl);
        vint16m4_t v_abs_gy = __riscv_vmax_vv_i16m4(v_gy, __riscv_vneg_v_i16m4(v_gy, vl), vl);
        vint16m4_t v_mag    = __riscv_vadd_vv_i16m4(v_abs_gx, v_abs_gy, vl);

        // Widen i16→i32 (m4 → m8), multiply by fixed-point scale, decode with >>16
        vint32m8_t v_mag32  = __riscv_vwcvt_x_x_v_i32m8(v_mag, vl);
        vint32m8_t v_scaled = __riscv_vmul_vx_i32m8(v_mag32, inv_scale, vl);

        // Narrow 32→16→8 (m8 → m4 → m2), same vnsra+vncvt+vreinterpret pattern as sobel_rvv.cpp
        vint16m4_t v_16 = __riscv_vnsra_wx_i16m4(v_scaled, 16, vl);
        vuint8m2_t v_8  = __riscv_vreinterpret_v_i8m2_u8m2(
                              __riscv_vncvt_x_x_w_i8m2(v_16, vl));

        __riscv_vse8_v_u8m2(&magnitude[x], v_8, vl);
        x += vl;
    }
}
#endif // USE_RVV_GAUSSIAN

// NOTE: magnitude_l1 always uses the scalar path below, even in RVV builds.
// Benchmarked on this target (QEMU rv64gcv, vlen=256): the hand-written RVV
// version in compute_magnitude_rvv() is ~3-4x SLOWER than plain scalar code,
// regardless of LMUL (verified m2 vs m4: no meaningful difference). Root
// cause: QEMU's vector emulation cost here scales with elements processed,
// not instruction count, so low-arithmetic-intensity, memory-bound ops like
// this (one load, abs, add, multiply, shift per pixel) can't amortize the
// per-element vector overhead. Sobel/Gaussian win with RVV because they do
// far more compute per loaded byte. compute_magnitude_rvv() is kept below
// for reference / real-hardware use, but is intentionally not called here.
void magnitude_l1(const int16_t* gx, const int16_t* gy, uint8_t* magnitude, int width, int height) {
    int total_pixels = width * height;
    int16_t max_val = 0;

    for (int i = 0; i < total_pixels; i++) {
        int16_t mag = std::abs(gx[i]) + std::abs(gy[i]);
        if (mag > max_val) max_val = mag;
    }
    if (max_val == 0) max_val = 1;

    for (int i = 0; i < total_pixels; i++) {
        int16_t mag = std::abs(gx[i]) + std::abs(gy[i]);
        magnitude[i] = static_cast<uint8_t>((mag * 255) / max_val);
    }
}

// L2 (Euclidean) magnitude — scalar only, exercised by test_pipeline.cpp
void magnitude_l2(const int16_t* gx, const int16_t* gy, uint8_t* magnitude, int width, int height) {
    int total_pixels = width * height;
    float max_val = 0.0f;

    for (int i = 0; i < total_pixels; i++) {
        float mag = std::sqrt(static_cast<float>(gx[i]) * gx[i] + static_cast<float>(gy[i]) * gy[i]);
        if (mag > max_val) max_val = mag;
    }
    if (max_val == 0.0f) max_val = 1.0f;

    for (int i = 0; i < total_pixels; i++) {
        float mag = std::sqrt(static_cast<float>(gx[i]) * gx[i] + static_cast<float>(gy[i]) * gy[i]);
        magnitude[i] = static_cast<uint8_t>((mag * 255.0f) / max_val);
    }
}
