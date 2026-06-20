#include "magnitude.h"
#include <cmath>
#include <cstdint>
#include <algorithm>

#ifdef USE_RVV_GAUSSIAN
#include <riscv_vector.h>


void compute_magnitude_rvv(const int16_t* gx, const int16_t* gy, uint8_t* magnitude, int width, int height) {
    int total_pixels = width * height;

    int16_t max_val = 0;
    for (int i = 0; i < total_pixels; i++) {
        int16_t ag = gx[i] < 0 ? (int16_t)(-gx[i]) : gx[i];
        int16_t ab = gy[i] < 0 ? (int16_t)(-gy[i]) : gy[i];
        int16_t m  = ag + ab;
        if (m > max_val) max_val = m;
    }
    if (max_val == 0) max_val = 1;
    int32_t inv_scale = (int32_t)((255u << 16) / (uint32_t)max_val);
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
