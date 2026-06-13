#include "sobel.h"
#include <cstdint>

// Sobel X kernel
// -1  0  1
// -2  0  2
// -1  0  1

// Sobel Y kernel
//  1  2  1
//  0  0  0
// -1 -2 -1

void sobel(const uint8_t* input, int16_t* gx, int16_t* gy, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int32_t sx = 0, sy = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;

                    // Zero padding
                    uint8_t p = 0;
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width)
                        p = input[ny * width + nx];

                    // Sobel X
                    if (kx == -1) sx -= (ky == 0) ? 2*p : p;
                    if (kx ==  1) sx += (ky == 0) ? 2*p : p;

                    // Sobel Y
                    if (ky == -1) sy -= (kx == 0) ? 2*p : p;
                    if (ky ==  1) sy += (kx == 0) ? 2*p : p;
                }
            }

            gx[y * width + x] = (int16_t)sx;
            gy[y * width + x] = (int16_t)sy;
        }
    }
}
