#include "image_io.h"
#include <cstdlib>
#include <cstdio>

uint8_t* load_image(const char* path, int width, int height) {
    uint8_t* buf = (uint8_t*)aligned_alloc(64, width * height);
    FILE* f = fopen(path, "rb");
    fread(buf, 1, width * height, f);
    fclose(f);
    return buf;
}

void save_image(const char* path, const uint8_t* img, int width, int height) {
    FILE* f = fopen(path, "wb");
    fwrite(img, 1, width * height, f);
    fclose(f);
}
