#include "../include/sobel.h"
#include "../include/sobel_rvv.h"
#include <cstdio>
#include <cstdlib>

int main() {
    const int width = 256, height = 256;
    uint8_t* input = new uint8_t[width * height];
    int16_t* gx_scalar = new int16_t[width * height];
    int16_t* gy_scalar = new int16_t[width * height];
    int16_t* gx_rvv = new int16_t[width * height];
    int16_t* gy_rvv = new int16_t[width * height];

    srand(123);
    for (int i = 0; i < width * height; i++) input[i] = rand() % 256;

    sobel(input, gx_scalar, gy_scalar, width, height);
    sobel_rvv(input, gx_rvv, gy_rvv, width, height);

    int mismatches = 0;
    for (int i = 0; i < width * height; i++) {
        if (gx_scalar[i] != gx_rvv[i] || gy_scalar[i] != gy_rvv[i]) {
            mismatches++;
            if (mismatches <= 10)
                printf("Mismatch at pixel %d: gx_scalar=%d gx_rvv=%d gy_scalar=%d gy_rvv=%d\n",
                       i, gx_scalar[i], gx_rvv[i], gy_scalar[i], gy_rvv[i]);
        }
    }

    if (mismatches == 0)
        printf("PASS: RVV Sobel output matches scalar exactly (%d pixels checked)\n", width * height);
    else
        printf("FAIL: %d / %d pixels mismatched\n", mismatches, width * height);

    delete[] input;
    delete[] gx_scalar; delete[] gy_scalar;
    delete[] gx_rvv;    delete[] gy_rvv;
    return mismatches == 0 ? 0 : 1;
}
