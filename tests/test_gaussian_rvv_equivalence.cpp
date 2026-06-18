#include "../src/gaussian.h"
#include "../src/gaussian_rvv.h"
#include <cstdio>
#include <cstdlib>

int main() {
    const int width = 256, height = 256;
    uint8_t* input = new uint8_t[width * height];
    uint8_t* out_scalar = new uint8_t[width * height];
    uint8_t* out_rvv = new uint8_t[width * height];

    srand(42);
    for (int i = 0; i < width * height; i++) input[i] = rand() % 256;

    gaussian_blur<uint8_t, int32_t, int16_t>(input, out_scalar, width, height);
    gaussian_blur_rvv(input, out_rvv, width, height);

    int mismatches = 0;
    for (int i = 0; i < width * height; i++) {
        if (out_scalar[i] != out_rvv[i]) {
            mismatches++;
            if (mismatches <= 10)
                printf("Mismatch at pixel %d: scalar=%d rvv=%d\n", i, out_scalar[i], out_rvv[i]);
        }
    }

    if (mismatches == 0)
        printf("PASS: RVV Gaussian output matches scalar exactly (%d pixels checked)\n", width * height);
    else
        printf("FAIL: %d / %d pixels mismatched\n", mismatches, width * height);

    delete[] input;
    delete[] out_scalar;
    delete[] out_rvv;
    return mismatches == 0 ? 0 : 1;
}
