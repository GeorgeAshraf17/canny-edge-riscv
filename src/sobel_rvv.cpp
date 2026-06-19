#include "sobel_rvv.h"
#include <riscv_vector.h>
#include <cstdint>

static inline void sobel_pixel_scalar(const uint8_t* input, int width, int height, int x, int y, int16_t* gx_out, int16_t* gy_out) {
    int32_t sx = 0, sy = 0;
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int ny = y + ky;
            int nx = x + kx;
            uint8_t p = 0;
            if (ny >= 0 && ny < height && nx >= 0 && nx < width)
                p = input[ny * width + nx];
            if (kx == -1) sx -= (ky == 0) ? 2*p : p;
            if (kx ==  1) sx += (ky == 0) ? 2*p : p;
            if (ky == -1) sy -= (kx == 0) ? 2*p : p;
            if (ky ==  1) sy += (kx == 0) ? 2*p : p;
        }
    }
    *gx_out = (int16_t)sx;
    *gy_out = (int16_t)sy;
}

void sobel_rvv(const uint8_t* input, int16_t* gx, int16_t* gy, int width, int height) {
    for (int y = 0; y < height; y++) {
        if (y < 1 || y >= height - 1 || width < 3) {
            for (int x = 0; x < width; x++) {
                sobel_pixel_scalar(input, width, height, x, y, &gx[y * width + x], &gy[y * width + x]);
            }
            continue;
        }

        sobel_pixel_scalar(input, width, height, 0, y, &gx[y * width], &gy[y * width]);

        int x = 1;
        int x_end = width - 1;
        const uint8_t* row_top = &input[(y - 1) * width];
        const uint8_t* row_mid = &input[y * width];
        const uint8_t* row_bot = &input[(y + 1) * width];

        while (x < x_end) {
            size_t vl = __riscv_vsetvl_e8m1(x_end - x);

            vuint8m1_t c00 = __riscv_vle8_v_u8m1(&row_top[x - 1], vl);
            vuint8m1_t c01 = __riscv_vle8_v_u8m1(&row_top[x],     vl);
            vuint8m1_t c02 = __riscv_vle8_v_u8m1(&row_top[x + 1], vl);

            vuint8m1_t c10 = __riscv_vle8_v_u8m1(&row_mid[x - 1], vl);
            vuint8m1_t c12 = __riscv_vle8_v_u8m1(&row_mid[x + 1], vl);

            vuint8m1_t c20 = __riscv_vle8_v_u8m1(&row_bot[x - 1], vl);
            vuint8m1_t c21 = __riscv_vle8_v_u8m1(&row_bot[x],     vl);
            vuint8m1_t c22 = __riscv_vle8_v_u8m1(&row_bot[x + 1], vl);

            // 1. Widen from Unsigned 8-bit to Unsigned 16-bit registers (LMUL 1 -> 2)
            vuint16m2_t u00 = __riscv_vwcvtu_x_x_v_u16m2(c00, vl);
            vuint16m2_t u01 = __riscv_vwcvtu_x_x_v_u16m2(c01, vl);
            vuint16m2_t u02 = __riscv_vwcvtu_x_x_v_u16m2(c02, vl);

            vuint16m2_t u10 = __riscv_vwcvtu_x_x_v_u16m2(c10, vl);
            vuint16m2_t u12 = __riscv_vwcvtu_x_x_v_u16m2(c12, vl);

            vuint16m2_t u20 = __riscv_vwcvtu_x_x_v_u16m2(c20, vl);
            vuint16m2_t u21 = __riscv_vwcvtu_x_x_v_u16m2(c21, vl);
            vuint16m2_t u22 = __riscv_vwcvtu_x_x_v_u16m2(c22, vl);

            // 2. Reinterpret bitwise as Signed 16-bit variables safely (zero-cost instruction)
            vint16m2_t v00 = __riscv_vreinterpret_v_u16m2_i16m2(u00);
            vint16m2_t v01 = __riscv_vreinterpret_v_u16m2_i16m2(u01);
            vint16m2_t v02 = __riscv_vreinterpret_v_u16m2_i16m2(u02);

            vint16m2_t v10 = __riscv_vreinterpret_v_u16m2_i16m2(u10);
            vint16m2_t v12 = __riscv_vreinterpret_v_u16m2_i16m2(u12);

            vint16m2_t v20 = __riscv_vreinterpret_v_u16m2_i16m2(u20);
            vint16m2_t v21 = __riscv_vreinterpret_v_u16m2_i16m2(u21);
            vint16m2_t v22 = __riscv_vreinterpret_v_u16m2_i16m2(u22);

            // --- Sobel X Kernel Vector Operations ---
            vint16m2_t v_sx = __riscv_vsub_vv_i16m2(v02, v00, vl);
            vint16m2_t v_mid_x = __riscv_vsub_vv_i16m2(v12, v10, vl);
            v_sx = __riscv_vadd_vv_i16m2(v_sx, __riscv_vadd_vv_i16m2(v_mid_x, v_mid_x, vl), vl);
            v_sx = __riscv_vadd_vv_i16m2(v_sx, __riscv_vsub_vv_i16m2(v22, v20, vl), vl);

            // --- Sobel Y Kernel Vector Operations ---
            // sy = bottom_row - top_row (matches scalar: ky=+1 added, ky=-1 subtracted)
            vint16m2_t v_sy = __riscv_vsub_vv_i16m2(v20, v00, vl);
            vint16m2_t v_mid_y = __riscv_vsub_vv_i16m2(v21, v01, vl);
            v_sy = __riscv_vadd_vv_i16m2(v_sy, __riscv_vadd_vv_i16m2(v_mid_y, v_mid_y, vl), vl);
            v_sy = __riscv_vadd_vv_i16m2(v_sy, __riscv_vsub_vv_i16m2(v22, v02, vl), vl);

            __riscv_vse16_v_i16m2(&gx[y * width + x], v_sx, vl);
            __riscv_vse16_v_i16m2(&gy[y * width + x], v_sy, vl);

            x += vl;
        }

        sobel_pixel_scalar(input, width, height, width - 1, y, &gx[y * width + width - 1], &gy[y * width + width - 1]);
    }
}
