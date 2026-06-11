#include "gaussian.h"
#include <cstdint>
#include <algorithm>

static const int16_t GAUSS_KERNEL[5][5] = {
    { 1,  4,  7,  4,  1},
    { 4, 16, 26, 16,  4},
    { 7, 26, 41, 26,  7},
    { 4, 16, 26, 16,  4},
    { 1,  4,  7,  4,  1}
};

template<typename PixelT, typename AccumT, typename KernelT>
void gaussian_blur(const PixelT* input, PixelT* output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            AccumT sum = 0;
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;
                    PixelT pixel = 0;
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width)
                        pixel = input[ny * width + nx];
                    sum += (AccumT)pixel * GAUSS_KERNEL[ky+2][kx+2];
                }
            }
            int32_t result = sum / 273;
            output[y * width + x] = (PixelT)std::clamp(result, 0, 255);
        }
    }
}

template void gaussian_blur<uint8_t, int32_t, int16_t>(
    const uint8_t*, uint8_t*, int, int);
