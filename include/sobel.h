#pragma once
#include <cstdint>

void sobel(const uint8_t* input, int16_t* gx, int16_t* gy, int width, int height);
