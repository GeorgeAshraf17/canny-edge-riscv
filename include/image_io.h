#pragma once
#include <cstdint>

uint8_t* load_image(const char* path, int width, int height);
void     save_image(const char* path, const uint8_t* img, int width, int height);
