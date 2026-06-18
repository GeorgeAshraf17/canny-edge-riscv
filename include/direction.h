#pragma once
#include <cstdint>

// Direction codes:
// 0 = horizontal gradient   (edge is vertical,   0 degrees)
// 1 = diagonal "/"          (45 degrees)
// 2 = vertical gradient     (edge is horizontal, 90 degrees)
// 3 = diagonal "\"          (135 degrees)
void compute_direction(const int16_t* gx, const int16_t* gy,
                        uint8_t* direction, int width, int height);


