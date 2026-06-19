#include "magnitude.h"
#include <cmath>
#include <cstdint>
#include <algorithm>

#ifdef USE_RVV_GAUSSIAN
#include <riscv_vector.h>

void compute_magnitude_rvv(const int16_t* gx, const int16_t* gy, uint8_t* magnitude, int width, int height) {
    int total_pixels = width * height;
    int x = 0;

    // Temp buffer: store Pass 1 magnitudes so Pass 2 does ONE load instead of
    // reloading gx+gy and recomputing abs/add (was wasted double-work before).
    int16_t* mag_buf = new int16_t[total_pixels];

    vint16m1_t v_global_max = __riscv_vmv_v_x_i16m1(0, 1);

    // PASS 1: compute L1 magnitude, store to mag_buf, reduce to global max
    while (x < total_pixels) {
        size_t vl = __riscv_vsetvl_e16m2(total_pixels - x);

        vint16m2_t v_gx = __riscv_vle16_v_i16m2(&gx[x], vl);
        vint16m2_t v_gy = __riscv_vle16_v_i16m2(&gy[x], vl);

        vint16m2_t v_abs_gx = __riscv_vmax_vv_i16m2(v_gx, __riscv_vneg_v_i16m2(v_gx, vl), vl);
        vint16m2_t v_abs_gy = __riscv_vmax_vv_i16m2(v_gy, __riscv_vneg_v_i16m2(v_gy, vl), vl);
        vint16m2_t v_mag    = __riscv_vadd_vv_i16m2(v_abs_gx, v_abs_gy, vl);

        __riscv_vse16_v_i16m2(&mag_buf[x], v_mag, vl);  // store for Pass 2
        v_global_max = __riscv_vredmax_vs_i16m2_i16m1(v_mag, v_global_max, vl);

        x += vl;
    }

    int16_t max_val = __riscv_vmv_x_s_i16m1_i16(v_global_max);
    if (max_val == 0) max_val = 1;

    // Fixed-point reciprocal: inv_scale = floor((255 << 16) / max_val)
    // So that (mag * inv_scale) >> 16  =  (mag * 255) / max_val  (integer approx)
    //
    // Overflow safety: mag <= max_val, so product <= max_val * (255<<16 / max_val)
    //                  = 255 * 65536 = 16,711,680  <<  INT32_MAX (2.1B)  — safe.
    //
    // This replaces vdiv_vx_i32m4, which was the root cause of the slowdown.
    // Integer vector division is emulated element-by-element under QEMU and is
    // 30-100x slower than vmul; the compiler cannot optimize away an explicit
    // intrinsic vdiv the way it can optimize scalar / into multiply-by-reciprocal.
    int32_t inv_scale = (int32_t)((255u << 16) / (uint32_t)max_val);

    // PASS 2: normalize with multiply+shift, no vdiv anywhere
    x = 0;
    while (x < total_pixels) {
        size_t vl = __riscv_vsetvl_e16m2(total_pixels - x);

        // Single load from precomputed mag_buf (vs reloading gx+gy + recomputing)
        vint16m2_t v_mag  = __riscv_vle16_v_i16m2(&mag_buf[x], vl);

        // Widen i16 -> i32, multiply by fixed-point scale
        vint32m4_t v_mag32  = __riscv_vwcvt_x_x_v_i32m4(v_mag, vl);
        vint32m4_t v_scaled = __riscv_vmul_vx_i32m4(v_mag32, inv_scale, vl);

        // Decode fixed-point (>>16) and narrow 32->16->8.
        // Same vnsra + vncvt + vreinterpret pattern already proven in sobel_rvv.
        vint16m2_t v_16 = __riscv_vnsra_wx_i16m2(v_scaled, 16, vl);
        vuint8m1_t v_8  = __riscv_vreinterpret_v_i8m1_u8m1(
                              __riscv_vncvt_x_x_w_i8m1(v_16, vl));

        __riscv_vse8_v_u8m1(&magnitude[x], v_8, vl);
        x += vl;
    }

    delete[] mag_buf;
}
#endif // USE_RVV_GAUSSIAN

// Scalar fallback — called when NOT compiling with -DUSE_RVV_GAUSSIAN
void magnitude_l1(const int16_t* gx, const int16_t* gy, uint8_t* magnitude, int width, int height) {
#ifdef USE_RVV_GAUSSIAN
    compute_magnitude_rvv(gx, gy, magnitude, width, height);
#else
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
