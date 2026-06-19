#include "magnitude.h"
#include <cmath>
#include <cstdint>
#include <algorithm>

#ifdef USE_RVV_GAUSSIAN
#include <riscv_vector.h>

// Your high-performance RVV vector reduction routine
// Only compiled (and only ever needs to be compiled) when targeting the
// RISC-V V extension toolchain with -DUSE_RVV_GAUSSIAN -- this is the
// fix: previously <riscv_vector.h> was included and this function was
// defined unconditionally, which broke host (g++) builds of `make test`
// since riscv_vector.h doesn't exist outside a RISC-V V cross-compiler.
void compute_magnitude_rvv(const int16_t* gx, const int16_t* gy, uint8_t* magnitude, int width, int height) {
    int total_pixels = width * height;
    int x = 0;

    vint16m1_t v_max_init = __riscv_vmv_v_x_i16m1(0, 1);
    vint16m1_t v_global_max = v_max_init;

    // PASS 1: Vectorized Absolute L1 Magnitude Calculation + Max Element Reduction
    while (x < total_pixels) {
        size_t vl = __riscv_vsetvl_e16m2(total_pixels - x);

        vint16m2_t v_gx = __riscv_vle16_v_i16m2(&gx[x], vl);
        vint16m2_t v_gy = __riscv_vle16_v_i16m2(&gy[x], vl);

        vint16m2_t v_abs_gx = __riscv_vmax_vv_i16m2(v_gx, __riscv_vneg_v_i16m2(v_gx, vl), vl);
        vint16m2_t v_abs_gy = __riscv_vmax_vv_i16m2(v_gy, __riscv_vneg_v_i16m2(v_gy, vl), vl);

        vint16m2_t v_mag = __riscv_vadd_vv_i16m2(v_abs_gx, v_abs_gy, vl);
        v_global_max = __riscv_vredmax_vs_i16m2_i16m1(v_mag, v_global_max, vl);

        x += vl;
    }

    int16_t max_val = __riscv_vmv_x_s_i16m1_i16(v_global_max);
    if (max_val == 0) max_val = 1;

    // PASS 2: Normalize and Write to 8-bit Output Map
    x = 0;
    while (x < total_pixels) {
        size_t vl = __riscv_vsetvl_e16m2(total_pixels - x);

        vint16m2_t v_gx = __riscv_vle16_v_i16m2(&gx[x], vl);
        vint16m2_t v_gy = __riscv_vle16_v_i16m2(&gy[x], vl);

        vint16m2_t v_abs_gx = __riscv_vmax_vv_i16m2(v_gx, __riscv_vneg_v_i16m2(v_gx, vl), vl);
        vint16m2_t v_abs_gy = __riscv_vmax_vv_i16m2(v_gy, __riscv_vneg_v_i16m2(v_gy, vl), vl);
        vint16m2_t v_mag = __riscv_vadd_vv_i16m2(v_abs_gx, v_abs_gy, vl);

        vint32m4_t v_scaled = __riscv_vwmul_vx_i32m4(v_mag, 255, vl);
        vint32m4_t v_norm = __riscv_vdiv_vx_i32m4(v_scaled, max_val, vl);

        vint16m2_t v_16 = __riscv_vnsra_wx_i16m2(v_norm, 0, vl);
        vuint8m1_t v_8  = __riscv_vreinterpret_v_i8m1_u8m1(__riscv_vncvt_x_x_w_i8m1(v_16, vl));

        __riscv_vse8_v_u8m1(&magnitude[x], v_8, vl);
        x += vl;
    }
}
#endif // USE_RVV_GAUSSIAN

// Classical scalar fallback matching the exact naming convention expected by main.cpp
void magnitude_l1(const int16_t* gx, const int16_t* gy, uint8_t* magnitude, int width, int height) {
#ifdef USE_RVV_GAUSSIAN
    // If compiling for RVV execution, dynamically run our optimized reduction pipeline
    compute_magnitude_rvv(gx, gy, magnitude, width, height);
#else
    // Foundational baseline scalar fallback
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
#endif
}

// L2 (Euclidean) magnitude: sqrt(gx^2 + gy^2), normalized to [0, 255].
// Declared in magnitude.h and exercised by tests/test_pipeline.cpp's
// MagnitudeTest suite, but previously had no implementation anywhere in
// the repo -- this was a silent linker-error bug (undefined reference)
// that would only surface once the riscv_vector.h host-build bug above
// was also fixed and `make test` actually got far enough to link.
// Scalar-only: no RVV variant exists for L2 yet, and main.cpp never
// calls this stage, so there's no performance-critical path depending
// on it being vectorized.
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
