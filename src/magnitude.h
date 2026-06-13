#pragma once
#include <cstdint>

void magnitude_l1(const int16_t* gx, const int16_t* gy, 
                  uint8_t* mag, int width, int height);

void magnitude_l2(const int16_t* gx, const int16_t* gy, 
                  uint8_t* mag, int width, int height);
