#include "magnitude.h"
#include <cstdint>
#include <cmath>
#include <algorithm>

// L1 norm: |Gx| + |Gy|
void magnitude_l1(const int16_t* gx, const int16_t* gy,
                  uint8_t* mag, int width, int height) {
    int n = width * height;
    
    // First pass: find max
    int32_t max_val = 0;
    for (int i = 0; i < n; i++) {
        int32_t val = abs(gx[i]) + abs(gy[i]);
        if (val > max_val) max_val = val;
    }
    
    // Second pass: normalize to [0, 255]
    for (int i = 0; i < n; i++) {
        int32_t val = abs(gx[i]) + abs(gy[i]);
        mag[i] = (uint8_t)(val * 255 / (max_val + 1));
    }
}

// L2 norm: sqrt(Gx^2 + Gy^2)
void magnitude_l2(const int16_t* gx, const int16_t* gy,
                  uint8_t* mag, int width, int height) {
    int n = width * height;
    
    // First pass: find max
    float max_val = 0;
    for (int i = 0; i < n; i++) {
        float val = sqrt((float)gx[i]*gx[i] + (float)gy[i]*gy[i]);
        if (val > max_val) max_val = val;
    }
    
    // Second pass: normalize to [0, 255]
    for (int i = 0; i < n; i++) {
        float val = sqrt((float)gx[i]*gx[i] + (float)gy[i]*gy[i]);
        mag[i] = (uint8_t)(val * 255.0f / (max_val + 1.0f));
    }
}
