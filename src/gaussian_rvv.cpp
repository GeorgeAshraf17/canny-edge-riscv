#include "gaussian_rvv.h"
#include <riscv_vector.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

static const int16_t K1[5] = {1, 4, 6, 4, 1};

static inline int32_t gaussian_pixel_scalar(const uint8_t* input, int width, int height, int x, int y) {
    static const int16_t K[5][5] = {
        { 1,  4,  7,  4,  1},
        { 4, 16, 26, 16,  4},
        { 7, 26, 41, 26,  7},
        { 4, 16, 26, 16,  4},
        { 1,  4,  7,  4,  1}
    };
    int32_t sum = 0;
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int ny = y + ky;
            int nx = x + kx;
            uint8_t pixel = 0;
            if (ny >= 0 && ny < height && nx >= 0 && nx < width)
                pixel = input[ny * width + nx];
            sum += (int32_t)pixel * K[ky + 2][kx + 2];
        }
    }
    return sum;
}

// Horizontal-only 1x5 pass at a single column, with zero-padding outside
// the row. Used for edge columns (x=0,1,width-2,width-1) and for rows
// that Pass 1's vectorized loop doesn't touch -- every row still needs a
// CORRECT horizontal intermediate value here, because Pass 2 reads
// neighboring rows' inter[] values (including row 0/1 and height-2/height-1)
// when it computes the vertical filter for rows 2 and height-3.
static inline int16_t horiz_pass_scalar(const uint8_t* row, int width, int x) {
    int32_t sum = 0;
    for (int kx = -2; kx <= 2; kx++) {
        int nx = x + kx;
        uint8_t p = (nx >= 0 && nx < width) ? row[nx] : 0;
        sum += (int32_t)p * K1[kx + 2];
    }
    return (int16_t)sum;
}

void gaussian_blur_rvv(const uint8_t* input, uint8_t* output, int width, int height) {
    int16_t* inter = (int16_t*)aligned_alloc(64, width * height * sizeof(int16_t));
    int16_t c2 = 4, c3 = 6;

    // =========================================================================
    // PASS 1: Horizontal Vectorized 1x5 Row Filter
    // =========================================================================
    for (int y = 0; y < height; y++) {
        const uint8_t* row_in = &input[y * width];
        int16_t* row_inter = &inter[y * width];

        // Always populate a CORRECT horizontal-only intermediate for this
        // row, regardless of whether it's a top/bottom boundary row --
        // Pass 2 needs every row's inter[] values as input, even rows
        // 0/1/height-2/height-1, which it never produces final output for
        // itself but does read as neighbors.
        if (width < 5) {
            for (int x = 0; x < width; x++)
                row_inter[x] = horiz_pass_scalar(row_in, width, x);
        } else {
            row_inter[0] = horiz_pass_scalar(row_in, width, 0);
            row_inter[1] = horiz_pass_scalar(row_in, width, 1);

            int x = 2;
            int x_end = width - 2;

            while (x < x_end) {
                size_t vl = __riscv_vsetvl_e8m1(x_end - x);

                vuint8m1_t v_m2 = __riscv_vle8_v_u8m1(&row_in[x - 2], vl);
                vuint8m1_t v_m1 = __riscv_vle8_v_u8m1(&row_in[x - 1], vl);
                vuint8m1_t v_0  = __riscv_vle8_v_u8m1(&row_in[x],     vl);
                vuint8m1_t v_p1 = __riscv_vle8_v_u8m1(&row_in[x + 1], vl);
                vuint8m1_t v_p2 = __riscv_vle8_v_u8m1(&row_in[x + 2], vl);

                vint16m2_t w_m2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwcvtu_x_x_v_u16m2(v_m2, vl));
                vint16m2_t w_m1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwcvtu_x_x_v_u16m2(v_m1, vl));
                vint16m2_t w_0  = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwcvtu_x_x_v_u16m2(v_0,  vl));
                vint16m2_t w_p1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwcvtu_x_x_v_u16m2(v_p1, vl));
                vint16m2_t w_p2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwcvtu_x_x_v_u16m2(v_p2, vl));

                vint16m2_t acc = __riscv_vadd_vv_i16m2(w_m2, w_p2, vl);
                vint16m2_t sides = __riscv_vadd_vv_i16m2(w_m1, w_p1, vl);
                acc = __riscv_vadd_vv_i16m2(acc, __riscv_vsll_vx_i16m2(sides, 2, vl), vl);
                acc = __riscv_vadd_vv_i16m2(acc, __riscv_vadd_vv_i16m2(__riscv_vsll_vx_i16m2(w_0, 2, vl), __riscv_vsll_vx_i16m2(w_0, 1, vl), vl), vl);

                __riscv_vse16_v_i16m2(&row_inter[x], acc, vl);
                x += vl;
            }

            row_inter[width - 2] = horiz_pass_scalar(row_in, width, width - 2);
            row_inter[width - 1] = horiz_pass_scalar(row_in, width, width - 1);
        }

        // Rows that Pass 2 never produces output for (top/bottom boundary)
        // get their FINAL output written here directly via the exact
        // scalar 2D kernel, since Pass 2's loop range is [2, height-3].
        if (y < 2 || y >= height - 2) {
            for (int x = 0; x < width; x++) {
                int32_t s = gaussian_pixel_scalar(input, width, height, x, y);
                output[y * width + x] = (uint8_t)std::clamp(s / 273, 0, 255);
            }
        }
    }

    // =========================================================================
    // PASS 2: Vertical Vectorized 5x1 Column Filter
    // =========================================================================
    for (int y = 2; y < height - 2; y++) {
        int x = 0;
        int x_end = width;
        uint8_t* row_out = &output[y * width];

        const int16_t* r_m2 = &inter[(y - 2) * width];
        const int16_t* r_m1 = &inter[(y - 1) * width];
        const int16_t* r_0  = &inter[y * width];
        const int16_t* r_p1 = &inter[(y + 1) * width];
        const int16_t* r_p2 = &inter[(y + 2) * width];

        while (x < x_end) {
            size_t vl = __riscv_vsetvl_e16m2(x_end - x);

            vint16m2_t w_m2 = __riscv_vle16_v_i16m2(&r_m2[x], vl);
            vint16m2_t w_m1 = __riscv_vle16_v_i16m2(&r_m1[x], vl);
            vint16m2_t w_0  = __riscv_vle16_v_i16m2(&r_0[x],  vl);
            vint16m2_t w_p1 = __riscv_vle16_v_i16m2(&r_p1[x], vl);
            vint16m2_t w_p2 = __riscv_vle16_v_i16m2(&r_p2[x], vl);

            vint32m4_t acc = __riscv_vwadd_vv_i32m4(w_m2, w_p2, vl);
            vint16m2_t sides = __riscv_vadd_vv_i16m2(w_m1, w_p1, vl);
            acc = __riscv_vwmacc_vx_i32m4(acc, c2, sides, vl);
            acc = __riscv_vwmacc_vx_i32m4(acc, c3, w_0, vl);

            // Separable kernel total weight is 16 (horizontal) * 16 (vertical)
            // = 256, NOT 273 (273 is the sum of the true non-separable 5x5
            // kernel used by the scalar reference). Divide by the weight
            // sum that actually applies to the kernel being computed here.
            vint32m4_t v_shifted = __riscv_vsra_vx_i32m4(acc, 8, vl);

            v_shifted = __riscv_vmax_vx_i32m4(v_shifted, 0, vl);
            v_shifted = __riscv_vmin_vx_i32m4(v_shifted, 255, vl);

            vint16m2_t v_16 = __riscv_vnsra_wx_i16m2(v_shifted, 0, vl);
            vuint8m1_t v_8  = __riscv_vreinterpret_v_i8m1_u8m1(__riscv_vncvt_x_x_w_i8m1(v_16, vl));

            __riscv_vse8_v_u8m1(&row_out[x], v_8, vl);
            x += vl;
        }
    }

    free(inter);
}
